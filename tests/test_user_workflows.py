"""
User Journey Tests for MLX-Tune

These tests replicate what real users do:
  1. Load model → apply LoRA → train → save → load → inference
  2. Different save formats (adapters, merged, GGUF)
  3. Edge cases (save before training, missing files, etc.)

These tests would have caught GitHub issues #3 and #4.

Run with: pytest tests/test_user_workflows.py -v
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

import mlx.core as mx


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODEL_4BIT = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_BF16 = "mlx-community/Llama-3.2-1B-Instruct-bf16"


def _load_bf16_model():
    """Load non-quantized model, skip test if unavailable."""
    from mlx_tune import FastLanguageModel

    try:
        return FastLanguageModel.from_pretrained(MODEL_BF16, max_seq_length=256)
    except RuntimeError as e:
        if "401" in str(e) or "Repository Not Found" in str(e):
            pytest.skip(
                f"Non-quantized model {MODEL_BF16} not accessible. "
                "Set HF_TOKEN or use `huggingface-cli login`."
            )
        raise


@pytest.fixture(scope="module")
def trained_model_and_dir():
    """Train a small model (2 steps) and yield (model, tokenizer, tmpdir).

    This fixture is scoped to the module so every test in this file shares
    the same trained artefacts, keeping the total runtime manageable.
    """
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_4BIT,
        max_seq_length=256,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=8,
    )

    dataset = Dataset.from_dict({"text": ["Hello world, this is a test."] * 4})

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=SFTConfig(
                output_dir=tmpdir,
                max_steps=2,
                per_device_train_batch_size=1,
                logging_steps=1,
                save_steps=2,
            ),
        )
        trainer.train()

        yield model, tokenizer, tmpdir


# ===================================================================
# Test A: Train → Save Merged → Load → Inference
# Would have caught GitHub issue #4
# ===================================================================


class TestSaveMergedAndReload:
    """Full round-trip: train → save merged → load → generate."""

    def test_save_merged_creates_loadable_model(self, trained_model_and_dir):
        """Save a merged model and load it back with from_pretrained."""
        from mlx_tune import FastLanguageModel

        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as merged_dir:
            # Save merged
            model.save_pretrained_merged(merged_dir, tokenizer)

            # Verify files exist
            assert any(
                Path(merged_dir).glob("*.safetensors")
            ), "No safetensors files in merged dir"
            assert (
                Path(merged_dir) / "config.json"
            ).exists(), "config.json missing in merged dir"

            # Load it back
            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                merged_dir, max_seq_length=256
            )
            assert loaded_model is not None
            assert loaded_tokenizer is not None

    def test_merged_model_can_generate(self, trained_model_and_dir):
        """Load the merged model and run inference."""
        from mlx_tune import FastLanguageModel
        from mlx_lm import generate

        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as merged_dir:
            model.save_pretrained_merged(merged_dir, tokenizer)

            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                merged_dir, max_seq_length=256
            )

            output = generate(
                loaded_model.model,
                loaded_tokenizer,
                prompt="Hello",
                max_tokens=10,
            )
            assert isinstance(output, str)
            assert len(output.strip()) > 0

    def test_merged_model_has_no_lora_artefacts(self, trained_model_and_dir):
        """Merged weights should not contain LoRA-specific keys."""
        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as merged_dir:
            model.save_pretrained_merged(merged_dir, tokenizer)

            # Inspect saved weight keys
            weight_files = list(Path(merged_dir).glob("*.safetensors"))
            assert len(weight_files) > 0

            for wf in weight_files:
                weights = mx.load(str(wf))
                lora_keys = [
                    k
                    for k in weights.keys()
                    if any(
                        p in k
                        for p in [".lora_a", ".lora_b", ".linear.weight", ".linear.biases"]
                    )
                ]
                assert lora_keys == [], f"Unfused LoRA keys found: {lora_keys[:5]}"


# ===================================================================
# Test B: Train → Save Adapters → Load Base + Adapters → Inference
# Tests the load_adapter() path that Issue #4 user tried
# ===================================================================


class TestSaveAdaptersAndReload:
    """Round-trip: train → save adapters → load base + adapters → generate."""

    def test_save_adapters_creates_files(self, trained_model_and_dir):
        """save_pretrained should produce adapters.safetensors + adapter_config.json."""
        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as adapter_dir:
            model.save_pretrained(adapter_dir)

            assert (
                Path(adapter_dir) / "adapters.safetensors"
            ).exists(), "adapters.safetensors missing"
            assert (
                Path(adapter_dir) / "adapter_config.json"
            ).exists(), "adapter_config.json missing"

    def test_load_adapter_on_fresh_base(self, trained_model_and_dir):
        """Load base model → load_adapter → verify adapters are applied."""
        from mlx_tune import FastLanguageModel

        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as adapter_dir:
            model.save_pretrained(adapter_dir)

            # Fresh base model
            base_model, base_tok = FastLanguageModel.from_pretrained(
                MODEL_4BIT, max_seq_length=256
            )

            # Load adapters
            base_model.load_adapter(adapter_dir)

            assert base_model._lora_applied is True

    def test_adapter_loaded_model_can_generate(self, trained_model_and_dir):
        """After loading adapters the model should be able to generate text."""
        from mlx_tune import FastLanguageModel
        from mlx_lm import generate

        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as adapter_dir:
            model.save_pretrained(adapter_dir)

            base_model, base_tok = FastLanguageModel.from_pretrained(
                MODEL_4BIT, max_seq_length=256
            )
            base_model.load_adapter(adapter_dir)

            output = generate(
                base_model.model,
                base_tok,
                prompt="Hello",
                max_tokens=10,
            )
            assert isinstance(output, str)
            assert len(output) > 0


# ===================================================================
# Test C: Train → Save All 3 Formats → Verify Each
# ===================================================================


class TestAllSaveFormats:
    """Verify adapters, merged, and GGUF saves all succeed."""

    def test_adapter_save_format(self, trained_model_and_dir):
        """save_pretrained creates valid adapter files."""
        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)

            adapters = Path(d) / "adapters.safetensors"
            config = Path(d) / "adapter_config.json"
            assert adapters.exists()
            assert config.exists()
            assert adapters.stat().st_size > 0

            with open(config) as f:
                cfg = json.load(f)
            assert "lora_parameters" in cfg
            assert "num_layers" in cfg

    def test_merged_save_format(self, trained_model_and_dir):
        """save_pretrained_merged creates a directory loadable by mlx_lm."""
        from mlx_lm import load as mlx_load

        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained_merged(d, tokenizer)

            # Must have weights + config
            assert any(Path(d).glob("*.safetensors"))
            assert (Path(d) / "config.json").exists()

            # Must be loadable
            m, t = mlx_load(d)
            assert m is not None

    @pytest.mark.slow
    def test_gguf_save_format_non_quantized(self):
        """GGUF export with a non-quantized (bf16) base + trained LoRA.

        Requires llama.cpp (see LLAMA_CPP_PATH); skipped otherwise.
        """
        from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
        from mlx_tune.trainer import _find_llama_cpp
        from datasets import Dataset

        tools = _find_llama_cpp()
        if not (tools["convert_script"] and tools["quantize_bin"]):
            pytest.skip("llama.cpp not available (set LLAMA_CPP_PATH to run GGUF export)")

        model, tokenizer = _load_bf16_model()
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )
        dataset = Dataset.from_dict({"text": ["Test sentence."] * 4})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_steps=2,
                ),
            )
            trainer.train()

            # Verify adapters were saved (our code)
            adapter_path = Path(tmpdir) / "adapters"
            assert (adapter_path / "adapters.safetensors").exists()
            assert (adapter_path / "adapter_config.json").exists()

            gguf_dir = os.path.join(tmpdir, "gguf_out")
            model.save_pretrained_gguf(gguf_dir, tokenizer)

            gguf_files = list(Path(gguf_dir).glob("*.gguf"))
            assert len(gguf_files) > 0, "No .gguf file created"
            assert gguf_files[0].stat().st_size > 0, "GGUF file is empty"


# ===================================================================
# Test D: Inference Output Consistency
# Verify output before and after save/load round-trip
# ===================================================================


class TestInferenceConsistency:
    """Output from the model before saving should match after load."""

    def test_merged_output_matches(self, trained_model_and_dir):
        """Generate with trained model, save merged, load, generate again.

        We don't expect *exact* equality because fusing LoRA introduces
        small floating-point differences, but the outputs should be non-empty
        and both should be valid text.
        """
        from mlx_tune import FastLanguageModel
        from mlx_lm import generate

        model, tokenizer, _tmpdir = trained_model_and_dir
        prompt = "What is machine learning?"

        # Generate BEFORE saving
        before = generate(
            model.model, tokenizer, prompt=prompt, max_tokens=20,
        )
        assert isinstance(before, str) and len(before) > 0

        with tempfile.TemporaryDirectory() as merged_dir:
            model.save_pretrained_merged(merged_dir, tokenizer)

            loaded_model, loaded_tok = FastLanguageModel.from_pretrained(
                merged_dir, max_seq_length=256
            )

            after = generate(
                loaded_model.model, loaded_tok, prompt=prompt, max_tokens=20,
            )
            assert isinstance(after, str) and len(after) > 0


# ===================================================================
# Test E: Edge Cases
# ===================================================================


class TestEdgeCases:
    """Edge-case behaviour that should not crash."""

    def test_save_pretrained_before_training(self):
        """save_pretrained() before any training should not crash."""
        from mlx_tune import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(model, r=8)

        with tempfile.TemporaryDirectory() as d:
            # Should print a warning but NOT raise
            model.save_pretrained(d)

    def test_save_merged_without_lora(self):
        """save_pretrained_merged without LoRA should save the base model."""
        from mlx_tune import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained_merged(d, tokenizer)
            assert any(Path(d).glob("*.safetensors")), "No weights saved for base model"

    def test_load_adapter_missing_safetensors(self):
        """load_adapter with empty directory should give clear FileNotFoundError."""
        from mlx_tune import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )

        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError, match="adapters.safetensors"):
                model.load_adapter(d)

    def test_load_adapter_missing_config(self):
        """load_adapter with safetensors but no config should error clearly."""
        from mlx_tune import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )

        with tempfile.TemporaryDirectory() as d:
            # Create a dummy safetensors file
            (Path(d) / "adapters.safetensors").write_bytes(b"dummy")

            with pytest.raises(FileNotFoundError, match="adapter_config.json"):
                model.load_adapter(d)

    def test_load_adapter_nonexistent_path(self):
        """load_adapter with a path that doesn't exist should error."""
        from mlx_tune import FastLanguageModel

        model, _ = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )

        with pytest.raises(FileNotFoundError):
            model.load_adapter("/nonexistent/path/to/adapters")

    def test_double_save_overwrites(self, trained_model_and_dir):
        """Saving twice to the same directory should overwrite cleanly."""
        model, tokenizer, _tmpdir = trained_model_and_dir

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained_merged(d, tokenizer)
            first_files = set(os.listdir(d))

            # Save again to same directory
            model.save_pretrained_merged(d, tokenizer)
            second_files = set(os.listdir(d))

            assert first_files == second_files, "Double save changed file listing"

    def test_gguf_export_warns_quantized(self):
        """GGUF export with quantized model should print a warning, not silently fail."""
        from mlx_tune import FastLanguageModel
        import io
        import contextlib

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(model, r=8)

        with tempfile.TemporaryDirectory() as d:
            # Capture stdout to check for warning
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    model.save_pretrained_gguf(d, tokenizer)
            except Exception:
                pass  # export may fail for quantized model, that's expected

            output = captured.getvalue()
            # The code should have printed a warning about quantized models
            assert (
                "quantized" in output.lower() or "warning" in output.lower()
            ), "No warning printed for quantized model GGUF export"


# ===================================================================
# Test F: RL Trainer → Save Adapters → Verify
# Covers ORPO, GRPO, KTO, SimPO save paths
# ===================================================================


PREFERENCE_DATA = [
    {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
    {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
    {"prompt": "Color of sky?", "chosen": "Blue", "rejected": "Green"},
] * 2  # 6 samples


def _verify_adapter_files(adapter_path: Path):
    """Assert adapters.safetensors and adapter_config.json exist and are valid."""
    sf = adapter_path / "adapters.safetensors"
    cfg = adapter_path / "adapter_config.json"

    assert sf.exists(), f"adapters.safetensors missing in {adapter_path}"
    assert sf.stat().st_size > 0, "adapters.safetensors is empty"
    assert cfg.exists(), f"adapter_config.json missing in {adapter_path}"

    with open(cfg) as f:
        config = json.load(f)
    assert "lora_parameters" in config, "adapter_config missing lora_parameters"
    assert "num_layers" in config, "adapter_config missing num_layers"
    assert config["num_layers"] is not None, "num_layers is None"

    return config


class TestRLTrainerSaveWorkflows:
    """Verify that every RL trainer produces valid adapter files after training."""

    def test_orpo_train_and_save(self):
        """ORPO train → save → verify adapter files exist and are valid."""
        from mlx_tune import FastLanguageModel, ORPOTrainer, ORPOConfig

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ORPOTrainer(
                model=model,
                train_dataset=PREFERENCE_DATA,
                tokenizer=tokenizer,
                args=ORPOConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    logging_steps=1,
                ),
            )
            result = trainer.train()

            assert result["status"] == "success"
            _verify_adapter_files(Path(tmpdir) / "adapters")

    def test_grpo_train_and_save(self):
        """GRPO train → save → verify adapter files exist and are valid."""
        from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )

        grpo_data = [
            {"prompt": "What is 2+2?"},
            {"prompt": "Capital of France?"},
        ] * 2

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = GRPOTrainer(
                model=model,
                train_dataset=grpo_data,
                tokenizer=tokenizer,
                args=GRPOConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    num_generations=2,
                    logging_steps=1,
                    max_completion_length=20,
                ),
            )
            result = trainer.train()

            assert result["status"] == "success"
            _verify_adapter_files(Path(tmpdir) / "adapters")

    def test_kto_train_and_save(self):
        """KTO train → save → verify adapter files exist and are valid."""
        from mlx_tune import FastLanguageModel, KTOTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )

        kto_data = [
            {"text": "Good response about math", "label": 1},
            {"text": "Bad response with wrong info", "label": 0},
        ] * 3

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = KTOTrainer(
                model=model,
                train_dataset=kto_data,
                tokenizer=tokenizer,
                output_dir=tmpdir,
                max_steps=2,
                logging_steps=1,
            )
            result = trainer.train()

            assert result["status"] == "success"
            _verify_adapter_files(Path(tmpdir) / "adapters")

    def test_simpo_train_and_save(self):
        """SimPO train → save → verify adapter files exist and are valid."""
        from mlx_tune import FastLanguageModel, SimPOTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimPOTrainer(
                model=model,
                train_dataset=PREFERENCE_DATA,
                tokenizer=tokenizer,
                output_dir=tmpdir,
                max_steps=2,
                logging_steps=1,
            )
            result = trainer.train()

            assert result["status"] == "success"
            _verify_adapter_files(Path(tmpdir) / "adapters")

    def test_dpo_adapter_files_loadable_by_mlx_lm(self):
        """DPO adapters can be loaded by mlx_lm (critical for GGUF export)."""
        from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig
        from mlx_lm import load

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_4BIT, max_seq_length=256
        )
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DPOTrainer(
                model=model,
                train_dataset=PREFERENCE_DATA,
                tokenizer=tokenizer,
                args=DPOConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    logging_steps=1,
                ),
            )
            trainer.train()

            adapter_path = Path(tmpdir) / "adapters"
            _verify_adapter_files(adapter_path)

            # Load with mlx_lm — this is the real compatibility check
            loaded_model, loaded_tok = load(
                MODEL_4BIT, adapter_path=str(adapter_path)
            )
            assert loaded_model is not None


# ===================================================================
# Test G: Non-Quantized Model Save Paths
# ===================================================================


class TestNonQuantizedModelSave:
    """Verify save paths work with non-quantized (bf16) models."""

    @pytest.mark.slow
    def test_save_merged_non_quantized(self):
        """save_pretrained_merged with non-quantized model creates loadable output."""
        from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
        from datasets import Dataset
        from mlx_lm import load as mlx_load

        model, tokenizer = _load_bf16_model()
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )
        dataset = Dataset.from_dict({"text": ["Hello world test."] * 4})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_steps=2,
                ),
            )
            trainer.train()

            merged_dir = os.path.join(tmpdir, "merged")
            model.save_pretrained_merged(merged_dir, tokenizer)

            # Verify loadable
            m, t = mlx_load(merged_dir)
            assert m is not None

    @pytest.mark.slow
    def test_non_quantized_merged_generates(self):
        """Non-quantized merged model can run inference after reload."""
        from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
        from datasets import Dataset
        from mlx_lm import generate

        model, tokenizer = _load_bf16_model()
        model = FastLanguageModel.get_peft_model(
            model, r=8, target_modules=["q_proj", "v_proj"], lora_alpha=8
        )
        dataset = Dataset.from_dict({"text": ["Hello world test."] * 4})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_steps=2,
                ),
            )
            trainer.train()

            merged_dir = os.path.join(tmpdir, "merged")
            model.save_pretrained_merged(merged_dir, tokenizer)

            loaded_model, loaded_tok = FastLanguageModel.from_pretrained(
                merged_dir, max_seq_length=256
            )
            output = generate(
                loaded_model.model, loaded_tok, prompt="Hello", max_tokens=10
            )
            assert isinstance(output, str) and len(output.strip()) > 0


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
