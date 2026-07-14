# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the Dynamic-SD k=0 drafter skip.

Stubs the GPUModelRunner.propose_draft_token_ids -> drafter.propose seam:
the original SpecDecodeBaseProposer.propose is replaced with a recorder
(standing in for the expensive draft forward) before install() wraps it,
so we can assert exactly when the draft forward would have run.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

import vllm_gguf_plugin.mtp_dynamic_skip as mds


class _Recorder:
    """Stands in for the original propose (i.e. the draft forward)."""

    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, proposer, num_speculative_tokens, *args, **kwargs):
        self.calls.append(((num_speculative_tokens, *args), dict(kwargs)))
        proposer.num_speculative_tokens = num_speculative_tokens
        batch = kwargs["common_attn_metadata"].batch_size()
        return torch.zeros(batch, num_speculative_tokens, dtype=torch.int64)


@pytest.fixture
def patched(monkeypatch):
    """Install the skip wrapper around a recorder standing in for propose."""
    recorder = _Recorder()
    monkeypatch.setattr(SpecDecodeBaseProposer, "propose", recorder, raising=True)
    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_gguf_dynamic_k0_skip_patched",
        False,
        raising=False,
    )
    mds.install()
    yield recorder
    # monkeypatch restores propose; drop the flag so other tests/plugin
    # registration see a clean class.
    SpecDecodeBaseProposer._gguf_dynamic_k0_skip_patched = False


def _make_proposer(dynamic: bool) -> SpecDecodeBaseProposer:
    p = object.__new__(SpecDecodeBaseProposer)
    p.speculative_config = SimpleNamespace(
        uses_dynamic_speculative_decoding=lambda: dynamic
    )
    p.num_speculative_tokens = 1
    p._last_draft_probs = object()  # stale sentinel from a previous k>0 step
    return p


def _step_kwargs(batch_size: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return dict(
        target_token_ids=torch.randint(0, 100, (batch_size * 2,), generator=g),
        target_positions=torch.arange(batch_size * 2),
        target_hidden_states=torch.randn(batch_size * 2, 8, generator=g),
        next_token_ids=torch.randint(0, 100, (batch_size,), generator=g),
        token_indices_to_sample=None,
        common_attn_metadata=SimpleNamespace(batch_size=lambda: batch_size),
        sampling_metadata=SimpleNamespace(),
    )


def test_k0_dynamic_skips_draft_forward(patched):
    p = _make_proposer(dynamic=True)
    kwargs = _step_kwargs(batch_size=3, seed=0)
    out = SpecDecodeBaseProposer.propose(p, 0, **kwargs)

    assert patched.calls == [], "k=0 step must never reach the draft forward"
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 0)
    assert out.dtype == torch.int64
    # State mirrors upstream's k==0 path.
    assert p.num_speculative_tokens == 0
    assert p._last_draft_probs is None


def test_k_positive_unchanged(patched):
    p = _make_proposer(dynamic=True)
    kwargs = _step_kwargs(batch_size=2, seed=1)
    out = SpecDecodeBaseProposer.propose(p, 1, **kwargs)

    assert len(patched.calls) == 1
    called_args, called_kwargs = patched.calls[0]
    assert called_args[0] == 1
    # Args forwarded to the original untouched (identity, not copies).
    for key, val in kwargs.items():
        assert called_kwargs[key] is val
    assert out.shape == (2, 1)


def test_k0_static_config_falls_through(patched):
    """Without a dynamic schedule the wrapper must not intervene."""
    p = _make_proposer(dynamic=False)
    kwargs = _step_kwargs(batch_size=2, seed=2)
    SpecDecodeBaseProposer.propose(p, 0, **kwargs)
    assert len(patched.calls) == 1, "static config: upstream handles k=0 itself"


def test_transition_0_to_1_matches_always_on(patched):
    """A k=1 step after a k=0 skip must present the draft forward with
    exactly the inputs an always-on run would have produced."""
    p = _make_proposer(dynamic=True)
    step1 = _step_kwargs(batch_size=1, seed=10)
    step2 = _step_kwargs(batch_size=2, seed=11)  # batch grew -> schedule k=0
    step3 = _step_kwargs(batch_size=1, seed=12)  # batch shrank -> k=1 again

    SpecDecodeBaseProposer.propose(p, 1, **step1)
    SpecDecodeBaseProposer.propose(p, 0, **step2)
    SpecDecodeBaseProposer.propose(p, 1, **step3)

    # Only the two k>0 steps reached the forward.
    assert [c[0][0] for c in patched.calls] == [1, 1]
    # The post-transition call's inputs are identical to what the runner
    # supplied — the skip mutated nothing that feeds the next proposal.
    _, k3_kwargs = patched.calls[1]
    for key, val in step3.items():
        assert k3_kwargs[key] is val
    # And no tensor content was altered in place by the k=0 skip.
    assert torch.equal(
        step3["target_hidden_states"],
        _step_kwargs(batch_size=1, seed=12)["target_hidden_states"],
    )


def test_install_idempotent(patched):
    before = SpecDecodeBaseProposer.propose
    mds.install()
    assert SpecDecodeBaseProposer.propose is before


def test_positional_args_supported(patched):
    """The only in-tree call site uses keywords, but positional calls must
    still resolve batch size / device correctly."""
    p = _make_proposer(dynamic=True)
    kw = _step_kwargs(batch_size=4, seed=3)
    out = SpecDecodeBaseProposer.propose(
        p,
        0,
        kw["target_token_ids"],
        kw["target_positions"],
        kw["target_hidden_states"],
        kw["next_token_ids"],
        kw["token_indices_to_sample"],
        kw["common_attn_metadata"],
        kw["sampling_metadata"],
    )
    assert patched.calls == []
    assert out.shape == (4, 0)
