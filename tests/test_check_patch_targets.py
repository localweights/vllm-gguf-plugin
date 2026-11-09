# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for ``tools/check_patch_targets.py``.

The tool is the guard against silent upstream renames of the attributes the
plugin monkeypatches, so it is tested on both halves: the static collection
(aliased imports, ``setattr``, reads, constant-value exclusion) and the
resolution/exit-status behaviour.
"""

import importlib.util
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "check_patch_targets", TOOLS_DIR / "check_patch_targets.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_patch_targets", module)
    spec.loader.exec_module(module)
    return module


cpt = _load_tool()


def _write(tmp_path: Path, name: str, source: str) -> Path:
    package = tmp_path / name
    package.mkdir()
    (package / "mod.py").write_text(source)
    return package


def test_collect_finds_aliased_assignment_and_read(tmp_path):
    package = _write(
        tmp_path,
        "pkg_alias",
        "import vllm.config as vc\n"
        "from vllm.engine.arg_utils import EngineArgs as EA\n"
        "\n"
        "def install():\n"
        "    orig = EA.create_model_config\n"
        "    vc.VllmConfig.__post_init__ = orig\n",
    )
    paths = {t.path for t in cpt.collect(str(package))}
    assert "vllm.engine.arg_utils.EngineArgs.create_model_config" in paths
    assert "vllm.config.VllmConfig.__post_init__" in paths


def test_collect_follows_class_alias_inside_function(tmp_path):
    package = _write(
        tmp_path,
        "pkg_class_alias",
        "def install():\n"
        "    from vllm.v1.spec_decode import llm_base_proposer as prop\n"
        "    P = prop.SpecDecodeBaseProposer\n"
        "    orig = P.propose\n"
        "    P.propose = orig\n",
    )
    paths = {t.path for t in cpt.collect(str(package))}
    propose = "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.propose"
    assert propose in paths
    assert paths == {
        "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer",
        "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.propose",
    }


def test_collect_handles_setattr_and_try_scoped_import(tmp_path):
    package = _write(
        tmp_path,
        "pkg_setattr",
        "def install():\n"
        "    try:\n"
        "        from vllm.v1.kv_offload.cpu.gpu_worker import Handler\n"
        "    except ImportError:\n"
        "        return\n"
        "    setattr(Handler, '__init__', lambda self: None)\n",
    )
    paths = {t.path for t in cpt.collect(str(package))}
    assert paths == {"vllm.v1.kv_offload.cpu.gpu_worker.Handler.__init__"}


def test_collect_ignores_constant_marker_attributes(tmp_path):
    package = _write(
        tmp_path,
        "pkg_marker",
        "import vllm.config as vc\n"
        "vc.VllmConfig._plugin_patched = True\n"
        "vc.VllmConfig._plugin_name = 'gguf'\n",
    )
    assert cpt.collect(str(package)) == []


def test_collect_ignores_plugin_owned_targets(tmp_path):
    package = _write(
        tmp_path,
        "pkg_own",
        "from . import weights_adapter as wa\n"
        "wa.Adapter.build_name_map = lambda self, cfg: {}\n",
    )
    assert cpt.collect(str(package)) == []


def test_resolve_existing_and_missing():
    assert cpt.resolve("vllm.config.VllmConfig")[0] == cpt.EXISTS_YES
    exists, _sig, detail = cpt.resolve("vllm.config.VllmConfig.__no_such_attr__")
    assert exists == cpt.EXISTS_NO
    assert "missing attribute" in detail


def test_signature_is_rendered():
    exists, signature, _detail = cpt.resolve(
        "vllm.engine.arg_utils.EngineArgs.create_model_config"
    )
    assert exists == cpt.EXISTS_YES
    assert "self" in signature


def test_main_exit_code_tracks_missing_targets(tmp_path):
    ok = _write(
        tmp_path,
        "pkg_ok",
        "import vllm.config as vc\n"
        "vc.VllmConfig.__init__ = vc.VllmConfig.__init__\n",
    )
    assert cpt.main([str(ok)]) == 0

    bad = _write(
        tmp_path,
        "pkg_bad",
        "import vllm.config as vc\n"
        "vc.VllmConfig.__no_such_attr__ = object()\n",
    )
    assert cpt.main([str(bad)]) == 1


def test_main_rejects_bad_usage(tmp_path):
    assert cpt.main([]) == 2
    assert cpt.main([str(tmp_path / "does-not-exist")]) == 2


def test_collect_ignores_import_error_fallbacks(tmp_path):
    """A moved symbol imported from its OLD location inside `except ImportError`
    is a version fallback — the old location is expected to lack it on the new
    vLLM, so it must not be reported as a missing patch target."""
    package = _write(
        tmp_path,
        "pkg_fallback",
        "try:\n"
        "    from vllm.v1.kv_cache_interface import is_quantized_kv_cache\n"
        "except ImportError:\n"
        "    from vllm.utils import torch_utils as _tu\n"
        "    is_quantized_kv_cache = _tu.is_quantized_kv_cache\n",
    )
    assert cpt.collect(str(package)) == []


def test_declared_inert_target_does_not_fail(tmp_path):
    """A patch site annotated `# patch-target: inert` is reported as inert and
    does not drive the exit status (removed upstream on purpose)."""
    package = _write(
        tmp_path,
        "pkg_inert",
        "import vllm.config as vc\n"
        "# patch-target: inert\n"
        "vc.VllmConfig.__no_such_attr__ = object()\n",
    )
    rows, failures = cpt.check(str(package))
    assert failures == 0
    assert [row[1] for row in rows] == [cpt.EXISTS_INERT]
    assert cpt.main([str(package)]) == 0


def test_undeclared_missing_target_still_fails(tmp_path):
    """The inert escape hatch must not weaken the gate for real regressions."""
    package = _write(
        tmp_path,
        "pkg_regression",
        "import vllm.config as vc\n"
        "vc.VllmConfig.__no_such_attr__ = object()\n",
    )
    _rows, failures = cpt.check(str(package))
    assert failures == 1
    assert cpt.main([str(package)]) == 1


def test_version_guarded_target_does_not_fail(tmp_path):
    """A symbol that only exists on some supported vLLM versions is declared
    `# patch-target: version-guarded` at its (multi-line comment) patch site and
    must not fail the check when this interpreter lacks it."""
    package = _write(
        tmp_path,
        "pkg_guarded",
        "import vllm.config as vc\n"
        "# patch-target: version-guarded — new in vLLM 0.29\n"
        "# (the import above is guarded)\n"
        "vc.VllmConfig.__no_such_attr__ = object()\n",
    )
    rows, failures = cpt.check(str(package))
    assert failures == 0
    assert [row[1] for row in rows] == [cpt.EXISTS_GUARDED]
    assert cpt.main([str(package)]) == 0
