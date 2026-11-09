# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests: the GGUF MTP draft shares the target's embed_tokens/lm_head
and never allocates its own (T-4c).

Mock modules only; the real production wrappers are installed by
``mtp_enable.install()`` onto the real vLLM classes (monkeypatch restores them).
"""

import types

import pytest
import torch
import torch.nn as nn

import vllm_gguf_plugin.mtp_enable as mtp_enable

VOCAB, HID = 4096, 64


class _QuantStyle(nn.Module):
    """GGUF-quantized-style module: ``qweight`` only, no ``.weight``."""

    def __init__(self):
        super().__init__()
        self.qweight = nn.Parameter(
            torch.zeros(VOCAB, 8, dtype=torch.uint8), requires_grad=False
        )


def _gguf_cfg():
    qc = types.SimpleNamespace(get_name=lambda: "gguf")
    return types.SimpleNamespace(quant_config=qc, load_config=None)


@pytest.fixture
def installed(monkeypatch):
    """Install plugin wrappers over fakes of the Qwen3_5 MTP classes."""
    from vllm.config import speculative as spec
    from vllm.model_executor.models import qwen3_5_mtp as q
    from vllm.v1.spec_decode import llm_base_proposer as prop
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eu

    allocs = []

    def _emb(*a, **k):
        allocs.append("embed")
        return nn.Embedding(VOCAB, HID)

    def _head(*a, **k):
        allocs.append("lm_head")
        h = nn.Linear(HID, VOCAB, bias=False)
        h.tie_weights = lambda *_a, **_k: h
        return h

    class FakePredictor(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.embed_tokens = q.VocabParallelEmbedding(VOCAB, HID)

    class FakeMTP(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.model = FakePredictor(vllm_config=vllm_config)
            self.lm_head = q.ParallelLMHead(VOCAB, HID)
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

    monkeypatch.setattr(q, "VocabParallelEmbedding", _emb)
    monkeypatch.setattr(q, "ParallelLMHead", _head)
    monkeypatch.setattr(q, "Qwen3_5MultiTokenPredictor", FakePredictor)
    monkeypatch.setattr(q, "Qwen3_5MTP", FakeMTP)
    monkeypatch.setattr(q, "_gguf_mtp_placeholder_patched", False, raising=False)
    monkeypatch.setattr(prop.SpecDecodeBaseProposer, "_gguf_mtp_share_patched", False, raising=False)
    monkeypatch.setattr(prop.SpecDecodeBaseProposer, "_maybe_share_embeddings", prop.SpecDecodeBaseProposer._maybe_share_embeddings)
    monkeypatch.setattr(prop.SpecDecodeBaseProposer, "_maybe_share_lm_head", prop.SpecDecodeBaseProposer._maybe_share_lm_head)
    monkeypatch.setattr(eu, "_gguf_mtp_log_patched", False, raising=False)
    monkeypatch.setattr(eu, "maybe_share_target_embed", eu.maybe_share_target_embed)
    monkeypatch.setattr(spec.SpeculativeConfig, "_gguf_mtp_patched", False, raising=False)
    monkeypatch.setattr(spec.SpeculativeConfig, "hf_config_override", spec.SpeculativeConfig.hf_config_override)
    monkeypatch.setattr(spec.SpeculativeConfig, "__post_init__", spec.SpeculativeConfig.__post_init__)
    from vllm_gguf_plugin.weights_adapter import default as wa

    monkeypatch.setattr(wa.GGUFWeightsAdapter, "_gguf_mtp_namemap_patched", False, raising=False)
    monkeypatch.setattr(wa.GGUFWeightsAdapter, "build_name_map", wa.GGUFWeightsAdapter.build_name_map)
    mtp_enable.install()
    return types.SimpleNamespace(q=q, eu=eu, prop=prop, allocs=allocs, FakeMTP=FakeMTP)


def _target():
    t = nn.Module()
    t.model = nn.Module()
    t.model.embed_tokens = _QuantStyle()
    t.lm_head = _QuantStyle()
    return t


def test_draft_never_allocates_own_embed_or_lm_head(installed):
    draft = installed.q.Qwen3_5MTP(vllm_config=_gguf_cfg())
    assert installed.allocs == []  # constructor spy: no embed/lm_head built
    assert sum(p.numel() for p in draft.parameters()) == 0


def test_non_gguf_config_keeps_upstream_allocation(installed):
    cfg = types.SimpleNamespace(quant_config=None, load_config=None)
    installed.q.Qwen3_5MTP(vllm_config=cfg)
    assert installed.allocs == ["embed", "lm_head"]
    # globals restored after a GGUF construction too
    installed.q.Qwen3_5MTP(vllm_config=_gguf_cfg())
    installed.q.Qwen3_5MTP(vllm_config=cfg)
    assert installed.allocs.count("embed") == 2


def test_mrv2_share_path_with_quantized_target(installed, monkeypatch):
    eu = installed.eu
    draft = installed.q.Qwen3_5MTP(vllm_config=_gguf_cfg())
    target = _target()

    spec_cfg = types.SimpleNamespace(
        draft_model_config=object(),
        draft_lm_head_quantization=None,
        apply_draft_overrides=lambda c: c,
    )
    vc = types.SimpleNamespace(speculative_config=spec_cfg, load_config=object())
    monkeypatch.setattr(eu, "get_model", lambda **k: draft)
    monkeypatch.setattr(eu, "get_pp_safe_draft_load_config", lambda c: c)
    monkeypatch.setattr(eu, "get_draft_load_config", lambda c: vc.load_config)
    monkeypatch.setattr(eu, "get_pp_group", lambda: types.SimpleNamespace(world_size=1))
    monkeypatch.setattr(eu, "replace", lambda c, **k: c)

    out = eu.load_eagle_model(target, vc)
    assert out.model.embed_tokens is target.model.embed_tokens
    assert out.lm_head is target.lm_head
    assert sum(p.numel() for p in out.parameters()) == sum(
        p.numel() for p in target.parameters()
    )  # only the target's own modules, nothing extra
    assert installed.allocs == []


def test_mrv1_wrapper_shares_modules(installed):
    P = installed.prop.SpecDecodeBaseProposer
    draft = installed.q.Qwen3_5MTP(vllm_config=_gguf_cfg())
    target = _target()
    proposer = types.SimpleNamespace(model=draft)
    P._maybe_share_embeddings(proposer, target)
    P._maybe_share_lm_head(proposer, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    assert installed.allocs == []


def test_mrv1_wrapper_defers_to_upstream_for_other_drafts(installed):
    P = installed.prop.SpecDecodeBaseProposer
    other = types.SimpleNamespace(model=nn.Module())
    # a non-placeholder draft must hit the ORIGINAL (needs a pp group -> fails
    # in this bare CPU context rather than silently module-sharing)
    with pytest.raises(Exception):
        P._maybe_share_embeddings(other, _target())
    assert not hasattr(other.model, "model")


def test_weight_map_has_no_embed_or_lm_head_for_draft(installed):
    from vllm_gguf_plugin.weights_adapter import default as wa

    text_cfg = types.SimpleNamespace(num_hidden_layers=64)
    hf = types.SimpleNamespace(
        model_type="qwen3_5_mtp", get_text_config=lambda: text_cfg
    )
    m = wa.GGUFWeightsAdapter.build_name_map(
        object.__new__(wa.GGUFWeightsAdapter), types.SimpleNamespace(hf_config=hf)
    )
    assert "token_embd.weight" not in m
    assert "output.weight" not in m
    assert "blk.64.nextn.eh_proj.weight" in m
