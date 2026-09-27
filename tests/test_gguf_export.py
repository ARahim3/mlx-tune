"""
Unit tests for llama.cpp-based GGUF export (GitHub issue #3).

These tests mock the model merge and llama.cpp subprocess calls, so they need
neither llama.cpp nor model downloads. The real pipeline is exercised by the
slow/integration tests when llama.cpp is available (see LLAMA_CPP_PATH).
"""

import stat
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from mlx_tune import trainer
from mlx_tune.trainer import (
    LlamaCppNotFoundError,
    _find_llama_cpp,
    _normalize_gguf_quantization,
    export_to_gguf,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_llama_cpp(root: Path, with_quantize: bool = True) -> Path:
    """Create a minimal fake llama.cpp checkout layout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_hf_to_gguf.py").write_text("# fake converter\n")
    if with_quantize:
        q = root / "build" / "bin" / "llama-quantize"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text("#!/bin/sh\n")
        q.chmod(q.stat().st_mode | stat.S_IXUSR)
    return root


def _fake_run_factory(calls):
    """subprocess.run stand-in that records calls and creates the output file."""
    def _run(cmd, capture_output=True, text=True, **kw):
        calls.append(list(cmd))
        if "--outfile" in cmd:
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF-f16")
        else:  # llama-quantize <in> <out> <TYPE>
            Path(cmd[2]).write_bytes(b"GGUF-quant")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return _run


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """No llama.cpp anywhere unless a test adds it."""
    monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)
    monkeypatch.delenv("LLAMA_CPP_PYTHON", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr("shutil.which", lambda name: None)  # hide e.g. Homebrew llama-quantize
    return tmp_path


@pytest.fixture
def fake_merge(monkeypatch):
    """Skip the real mlx-lm load/fuse/dequantize step."""
    seen = {}

    def _merge(model_path, adapter_path, out_dir):
        seen["model_path"], seen["adapter_path"] = model_path, adapter_path
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text("{}")
        return out_dir

    monkeypatch.setattr(trainer, "_merge_for_gguf", _merge)
    return seen


# ---------------------------------------------------------------------------
# quantization name handling
# ---------------------------------------------------------------------------

class TestQuantizationNormalization:
    @pytest.mark.parametrize("given,expected", [
        ("q4_k_m", "q4_k_m"), ("Q4_0", "q4_0"), ("q8_0", "q8_0"), ("f16", "f16"),
        ("quantized", "q4_k_m"), ("fast_quantized", "q8_0"), ("not_quantized", "f16"),
        (None, "q4_k_m"),
    ])
    def test_names_and_unsloth_aliases(self, given, expected):
        assert _normalize_gguf_quantization(given) == expected

    def test_qat_forces_q4_0(self):
        assert _normalize_gguf_quantization("q8_0", qat=True) == "q4_0"
        assert _normalize_gguf_quantization(None, qat=True) == "q4_0"

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Unsupported GGUF quantization"):
            _normalize_gguf_quantization("q9_z")


# ---------------------------------------------------------------------------
# toolchain discovery
# ---------------------------------------------------------------------------

class TestFindLlamaCpp:
    def test_nothing_found(self, isolated_env):
        assert _find_llama_cpp() == {"convert_script": None, "quantize_bin": None}

    def test_explicit_path(self, isolated_env):
        root = _fake_llama_cpp(isolated_env / "lc")
        tools = _find_llama_cpp(str(root))
        assert tools["convert_script"] == str(root / "convert_hf_to_gguf.py")
        assert tools["quantize_bin"] == str(root / "build/bin/llama-quantize")

    def test_env_var(self, isolated_env, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc_env")
        monkeypatch.setenv("LLAMA_CPP_PATH", str(root))
        assert _find_llama_cpp()["convert_script"] == str(root / "convert_hf_to_gguf.py")

    def test_cwd_checkout(self, isolated_env):
        _fake_llama_cpp(isolated_env / "llama.cpp")
        assert _find_llama_cpp()["convert_script"].endswith("llama.cpp/convert_hf_to_gguf.py")

    def test_quantize_from_path(self, isolated_env, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc", with_quantize=False)
        monkeypatch.setattr("shutil.which", lambda name: "/opt/homebrew/bin/llama-quantize")
        tools = _find_llama_cpp(str(root))
        assert tools["quantize_bin"] == "/opt/homebrew/bin/llama-quantize"


# ---------------------------------------------------------------------------
# export_to_gguf orchestration
# ---------------------------------------------------------------------------

class TestExportToGGUF:
    def test_missing_llama_cpp_raises_helpful_error(self, isolated_env, fake_merge):
        with pytest.raises(LlamaCppNotFoundError) as exc:
            export_to_gguf("some/model", output_path=str(isolated_env / "m.gguf"))
        msg = str(exc.value)
        assert "convert_hf_to_gguf.py" in msg and "llama-quantize" in msg
        assert "LLAMA_CPP_PATH" in msg and "git clone" in msg
        assert fake_merge == {}, "must fail fast, before any expensive merge"

    def test_f16_does_not_need_quantize_binary(self, isolated_env, fake_merge, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc", with_quantize=False)
        calls = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
        out = export_to_gguf("m", output_path=str(isolated_env / "m.gguf"),
                             quantization="f16", llama_cpp_path=str(root))
        assert Path(out).read_bytes() == b"GGUF-f16"
        assert len(calls) == 1 and calls[0][calls[0].index("--outtype") + 1] == "f16"

    @pytest.mark.parametrize("quant,qat,expected_type", [
        ("q4_k_m", False, "Q4_K_M"),
        ("q8_0", False, "Q8_0"),
        ("q8_0", True, "Q4_0"),   # qat overrides
    ])
    def test_convert_then_quantize(self, isolated_env, fake_merge, monkeypatch, quant, qat, expected_type):
        root = _fake_llama_cpp(isolated_env / "lc")
        calls = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
        out_path = isolated_env / "out" / "model.gguf"
        out = export_to_gguf("base/model", output_path=str(out_path), quantization=quant,
                             adapter_path="./adapters", qat=qat, llama_cpp_path=str(root),
                             llama_cpp_python="/py/with/deps")

        assert fake_merge == {"model_path": "base/model", "adapter_path": "./adapters"}
        convert, quantize = calls
        assert convert[:2] == ["/py/with/deps", str(root / "convert_hf_to_gguf.py")]
        assert convert[convert.index("--outtype") + 1] == "f16"
        assert quantize[0] == str(root / "build/bin/llama-quantize")
        assert quantize[2] == str(out_path) and quantize[3] == expected_type
        assert Path(out).read_bytes() == b"GGUF-quant"
        # intermediates cleaned up: only the final file remains
        assert sorted(p.name for p in out_path.parent.iterdir()) == ["model.gguf"]

    def test_converter_failure_surfaces_missing_deps_hint(self, isolated_env, fake_merge, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc")
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="ModuleNotFoundError: No module named 'torch'"))
        with pytest.raises(RuntimeError, match="requirements-convert_hf_to_gguf.txt"):
            export_to_gguf("m", output_path=str(isolated_env / "m.gguf"), llama_cpp_path=str(root))
        assert not any(p.name.startswith("mlx_tune_gguf_") for p in isolated_env.iterdir())

    def test_keep_intermediates(self, isolated_env, fake_merge, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc")
        monkeypatch.setattr(subprocess, "run", _fake_run_factory([]))
        export_to_gguf("m", output_path=str(isolated_env / "m.gguf"), llama_cpp_path=str(root),
                       keep_intermediates=True)
        assert any(p.name.startswith("mlx_tune_gguf_") for p in isolated_env.iterdir())


# ---------------------------------------------------------------------------
# save_pretrained_gguf wiring
# ---------------------------------------------------------------------------

class TestSavePretrainedGGUFWiring:
    def test_passes_in_memory_model_quantization_and_kwargs(self, tmp_path):
        from mlx_tune.model import MLXModelWrapper

        tok = object()
        wrapper = MLXModelWrapper(model=object(), tokenizer=tok, max_seq_length=256,
                                  model_name="mlx-community/gemma-2-2b-it-4bit",
                                  config={"model_type": "gemma2", "quantization": {"bits": 4}})
        wrapper._lora_applied = True

        with mock.patch("mlx_tune.trainer.export_to_gguf", return_value="x.gguf") as m:
            ret = wrapper.save_pretrained_gguf(str(tmp_path / "out"), None,
                                               quantization_method="q4_0", qat=True)
        assert ret == "x.gguf"
        args, kwargs = m.call_args
        assert args[0] == "mlx-community/gemma-2-2b-it-4bit"
        assert kwargs["output_path"] == str(tmp_path / "out" / "model.gguf")
        assert kwargs["quantization"] == "q4_0"
        assert kwargs["qat"] is True
        # merges the in-memory model instead of reloading base + adapters from disk
        assert kwargs["model"] is wrapper
        assert kwargs["tokenizer"] is tok  # falls back to the wrapper's tokenizer
        assert "adapter_path" not in kwargs


class TestExportInMemoryModel:
    def test_in_memory_model_uses_save_model_hf_format(self, isolated_env, monkeypatch):
        root = _fake_llama_cpp(isolated_env / "lc")
        monkeypatch.setattr(subprocess, "run", _fake_run_factory([]))
        seen = {}

        def fake_hf_save(model, tokenizer, output_dir, save_method="merged_16bit", **kw):
            seen.update(model=model, tokenizer=tokenizer, save_method=save_method)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "config.json").write_text("{}")

        monkeypatch.setattr(trainer, "save_model_hf_format", fake_hf_save)
        monkeypatch.setattr(trainer, "_copy_tokenizer_assets", lambda *a, **k: None)
        monkeypatch.setattr(trainer, "_merge_for_gguf",
                            lambda *a, **k: pytest.fail("must not reload from disk"))

        model, tok = object(), object()
        export_to_gguf("base/model", output_path=str(isolated_env / "m.gguf"),
                       llama_cpp_path=str(root), model=model, tokenizer=tok)
        assert seen == {"model": model, "tokenizer": tok, "save_method": "merged_16bit"}
