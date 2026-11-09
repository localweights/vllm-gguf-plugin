# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the Dynamic-SD k=0 drafter skip.

Stubs the GPUModelRunner.propose_draft_token_ids -> drafter.propose seam:
the original SpecDecodeBaseProposer.propose is replaced with a recorder
(standing in for the expensive draft forward) before install() wraps it,
so we can assert exactly when the draft forward would have run.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

import vllm_gguf_plugin.mtp_dynamic_skip as mds

# Model Runner V2 speculators only exist on vLLM >= 0.29.
_MRV2_AVAILABLE = (
    importlib.util.find_spec(
        "vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator"
    )
    is not None
)
needs_mrv2 = pytest.mark.skipif(
    not _MRV2_AVAILABLE, reason="this vLLM has no Model Runner V2 speculators"
)


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


# ---------------------------------------------------------------------------
# Model Runner V2 (default since vLLM 0.29): the drafter is a Speculator
# ---------------------------------------------------------------------------
# MRV2 does not call SpecDecodeBaseProposer at all; TargetDependentARSpeculator
# (base of MTPSpeculator) runs the draft prefill forward on a k=0 step and only
# skips sampling.  The wrapper must skip that forward identically.


class _MVR2Recorder:
    """Stands in for the original speculator propose (the draft forward)."""

    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, speculator, input_batch, *args, **kwargs):
        self.calls.append(((input_batch, *args), dict(kwargs)))
        num_reqs = input_batch.num_reqs
        return torch.zeros(
            num_reqs, getattr(speculator, "num_speculative_steps", 1), dtype=torch.int64
        )


@pytest.fixture
def mrv2_patched(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    recorder = _MVR2Recorder()
    monkeypatch.setattr(TargetDependentARSpeculator, "propose", recorder, raising=True)
    monkeypatch.setattr(
        TargetDependentARSpeculator,
        "_gguf_dynamic_k0_skip_patched",
        False,
        raising=False,
    )
    mds._install_mrv2()
    yield recorder
    TargetDependentARSpeculator._gguf_dynamic_k0_skip_patched = False


class _FakeSpeculator:
    """Instance-shaped stand-in (no engine boot)."""

    def __init__(self, dynamic: bool, num_steps: int = 2, num_reqs: int = 3):
        self.speculative_config = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: dynamic
        )
        self.num_speculative_steps = num_steps
        self.draft_tokens = torch.ones(num_reqs, num_steps, dtype=torch.int64)


def _run_mrv2(speculator, **kwargs):
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    return TargetDependentARSpeculator.propose(
        speculator, SimpleNamespace(num_reqs=3), **kwargs
    )


@needs_mrv2
def test_mrv2_k0_dynamic_skips_draft_forward(mrv2_patched):
    spec = _FakeSpeculator(dynamic=True)
    out = _run_mrv2(spec, num_speculative_tokens=0)

    assert mrv2_patched.calls == [], "k=0 step must never reach the draft forward"
    assert out.shape == (3, 2)
    assert torch.all(out == -1), "mirror upstream's k==0 draft_tokens fill"


@needs_mrv2
def test_mrv2_k_positive_unchanged(mrv2_patched):
    spec = _FakeSpeculator(dynamic=True)
    sentinel = object()
    out = _run_mrv2(spec, num_speculative_tokens=2, dummy_run=False, marker=sentinel)

    assert len(mrv2_patched.calls) == 1
    _, called_kwargs = mrv2_patched.calls[0]
    assert called_kwargs["marker"] is sentinel
    assert called_kwargs["num_speculative_tokens"] == 2
    assert out.shape == (3, 2)


@needs_mrv2
def test_mrv2_k0_static_config_falls_through(mrv2_patched):
    spec = _FakeSpeculator(dynamic=False)
    _run_mrv2(spec, num_speculative_tokens=0)
    assert len(mrv2_patched.calls) == 1, "static config: upstream handles k=0 itself"


@needs_mrv2
def test_mrv2_never_skips_dummy_or_profile_runs(mrv2_patched):
    """Cudagraph capture / memory profiling must run the real forward."""
    for flag in ("dummy_run", "is_profile"):
        mrv2_patched.calls.clear()
        spec = _FakeSpeculator(dynamic=True)
        _run_mrv2(spec, num_speculative_tokens=0, **{flag: True})
        assert len(mrv2_patched.calls) == 1, f"{flag}=True must not be skipped"


@needs_mrv2
def test_mrv2_none_k_resolves_to_static_steps(mrv2_patched):
    """num_speculative_tokens=None means "self.num_speculative_steps" (non-zero
    here), so the forward runs."""
    spec = _FakeSpeculator(dynamic=True, num_steps=2)
    _run_mrv2(spec, num_speculative_tokens=None)
    assert len(mrv2_patched.calls) == 1


@needs_mrv2
def test_mrv2_install_idempotent(mrv2_patched):
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    before = TargetDependentARSpeculator.propose
    mds._install_mrv2()
    assert TargetDependentARSpeculator.propose is before


def test_install_wires_mrv1():
    """Wiring proof: install() patches the MRV1 proposer."""
    SpecDecodeBaseProposer._gguf_dynamic_k0_skip_patched = False
    mds.install()
    assert SpecDecodeBaseProposer._gguf_dynamic_k0_skip_patched is True


@needs_mrv2
def test_install_wires_mrv2():
    """Wiring proof: install() patches the MRV2 MTP speculator base too."""
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    TargetDependentARSpeculator._gguf_dynamic_k0_skip_patched = False
    mds.install()
    assert TargetDependentARSpeculator._gguf_dynamic_k0_skip_patched is True


@needs_mrv2
def test_mrv2_positional_flags_respected(mrv2_patched):
    """A fully positional call must still resolve k and the dummy/profile flags
    (the wrapper may not assume keywords)."""
    spec = _FakeSpeculator(dynamic=True)
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    # attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
    # num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature,
    # seeds, dp_sync_state, dummy_run, skip_attn_for_dummy_run, mm_inputs,
    # is_profile, num_speculative_tokens
    positional = [None] * 16
    positional[15] = 0  # k = 0 -> would be skipped ...
    positional[14] = True  # ... but this is a profiling run (is_profile=True)
    TargetDependentARSpeculator.propose(
        spec, SimpleNamespace(num_reqs=3), *positional
    )
    assert len(mrv2_patched.calls) == 1, "positional is_profile=True must not skip"

    mrv2_patched.calls.clear()
    positional[14] = False
    TargetDependentARSpeculator.propose(
        spec, SimpleNamespace(num_reqs=3), *positional
    )
    assert mrv2_patched.calls == [], "positional k=0 step must be skipped"
