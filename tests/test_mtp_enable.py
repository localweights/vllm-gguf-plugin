# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the ``mtp_enable`` SpeculativeConfig.__post_init__ wrapper.

Gate A (vLLM 0.33.0 boot, MTP N=1) failed inside
``arg_utils.create_speculative_config`` -> ``SpeculativeConfig.__post_init__``
-> ``ModelConfig(model=self.model, ...)`` with

    HFValidationError: Repo id must be in the form ...:
    '/models/qwen38-27b-iq4xs/Qwen3.8-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf'

because in 0.33 ``SpeculativeConfig.model`` for method "mtp" is whatever the
CLI passed (the raw .gguf), while the *config source* the draft ModelConfig
needs is the hf-config DIRECTORY (``target_model_config.model``; the .gguf
itself lives in ``target_model_config.model_weights``).

The wrapper must therefore rewrite ``self.model`` to
``target_model_config.model`` BEFORE the original ``__post_init__`` runs (the
upstream body is what constructs the draft ``ModelConfig``), and keep the
existing post-step that copies the target's ``model_weights`` onto the draft.

These tests install the real plugin wrapper (``mtp_enable.install()``) onto a
stand-in class substituted for ``vllm.config.speculative.SpeculativeConfig``,
so the wrapper code under test is the production code path, not a copy.
"""

import types

import pytest

import vllm_gguf_plugin.mtp_enable as mtp_enable

GGUF = "/models/qwen38-27b-iq4xs/Qwen3.8-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf"
CFG_DIR = "/models/qwen38-27b-iq4xs"


class _FakeSpeculativeConfig:
    """Minimal 0.33-``SpeculativeConfig``-shaped stand-in.

    ``__post_init__`` mirrors the upstream flow closely enough to expose the
    bug: it records the ``model`` it saw (proving the plugin's rewrite happened
    *before* it ran) and then builds a draft config from ``self.model`` the way
    upstream does with ``ModelConfig(model=self.model, runner="draft", ...)``.
    """

    def __init__(self, model, method, target_model_config=None):
        self.model = model
        self.method = method
        self.target_model_config = target_model_config
        self.draft_model_config = None
        self.model_seen_by_orig_post: list = []

    def __post_init__(self):
        self.model_seen_by_orig_post.append(self.model)
        self.draft_model_config = types.SimpleNamespace(
            model=self.model, model_weights=None
        )

    @staticmethod
    def hf_config_override(hf_config, transformers_mtp_overrides=None):
        return hf_config


@pytest.fixture
def patched(monkeypatch):
    """Install the real plugin wrapper onto the stand-in class."""
    from vllm.config import speculative as spec

    monkeypatch.setattr(spec, "SpeculativeConfig", _FakeSpeculativeConfig)
    mtp_enable.install()
    return _FakeSpeculativeConfig


def _target(model=CFG_DIR, model_weights=GGUF):
    return types.SimpleNamespace(model=model, model_weights=model_weights)


def test_gguf_draft_model_is_rewritten_to_config_dir(patched):
    """Bug 2: mtp + a .gguf ``model`` -> the hf-config directory."""
    cfg = patched(GGUF, "mtp", _target())
    cfg.__post_init__()

    assert cfg.model == CFG_DIR
    # Order is load-bearing: the rewrite must be visible to the ORIGINAL
    # __post_init__, because that is what builds the draft ModelConfig.
    assert cfg.model_seen_by_orig_post == [CFG_DIR]
    # Draft gets the config dir as its `model`, and the .gguf as its weights.
    assert cfg.draft_model_config.model == CFG_DIR
    assert cfg.draft_model_config.model_weights == GGUF


@pytest.mark.parametrize("method", ["mtp", "qwen3_5_mtp", "deepseek_mtp"])
def test_all_mtp_method_spellings_rewrite_gguf_model(patched, method):
    """0.33 aliases every MTP variant through ``MTPModelTypes`` (``*_mtp``)."""
    cfg = patched(GGUF, method, _target())
    cfg.__post_init__()

    assert cfg.model == CFG_DIR
    assert cfg.draft_model_config.model_weights == GGUF


def test_none_model_for_mtp_uses_target_config_dir(patched):
    """``model=None`` + mtp: upstream would pick ``model_weights`` (the .gguf)
    itself and hit the same HFValidationError, so the wrapper fills in the
    config dir first."""
    cfg = patched(None, "mtp", _target())
    cfg.__post_init__()

    assert cfg.model == CFG_DIR
    assert cfg.model_seen_by_orig_post == [CFG_DIR]
    assert cfg.draft_model_config.model_weights == GGUF


def test_non_gguf_mtp_model_untouched(patched):
    """An already-valid (non-.gguf) draft source must not be rewritten."""
    cfg = patched("Qwen/Qwen3.8-27B-MTP", "mtp", _target())
    cfg.__post_init__()

    assert cfg.model == "Qwen/Qwen3.8-27B-MTP"
    assert cfg.model_seen_by_orig_post == ["Qwen/Qwen3.8-27B-MTP"]
    assert cfg.draft_model_config.model_weights == GGUF


@pytest.mark.parametrize("method", ["draft_model", "eagle", "eagle3", "ngram"])
def test_non_mtp_method_with_gguf_path_untouched(patched, method):
    """Only the MTP lane is in scope: other speculative methods keep their
    ``model`` exactly as configured."""
    cfg = patched(GGUF, method, _target())
    cfg.__post_init__()

    assert cfg.model == GGUF
    assert cfg.model_seen_by_orig_post == [GGUF]
    # The post-step still propagates model_weights (pre-existing behaviour).
    assert cfg.draft_model_config.model_weights == GGUF


def test_mtp_without_target_config_does_not_crash(patched):
    """Defensive: no target config -> leave ``model`` alone (upstream raises
    its own clear error for mtp without a target)."""
    cfg = patched(GGUF, "mtp", None)
    cfg.__post_init__()

    assert cfg.model == GGUF
