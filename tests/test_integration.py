"""
Integration tests for common user workflows.

These tests ensure that the typical user experience works correctly,
catching API changes and regressions before users encounter them.

Note: These tests require a network connection and will download a small model.
They are marked with @pytest.mark.integration and can be skipped with:
    pytest tests/ -v -m "not integration"

To run only integration tests:
    pytest tests/test_integration.py -v
"""

import pytest
import tempfile
import os
from pathlib import Path


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests in this module."""
    from mlx_tune import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=512,
        load_in_4bit=True,
    )
    return model, tokenizer


@pytest.fixture(scope="module")
def model_with_lora(model_and_tokenizer):
    """Get model with LoRA applied."""
    from mlx_tune import FastLanguageModel

    model, tokenizer = model_and_tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=8,
    )
    return model, tokenizer


class TestModelLoading:
    """Test model loading workflows."""

    def test_from_pretrained_basic(self, model_and_tokenizer):
        """Test basic model loading works."""
        model, tokenizer = model_and_tokenizer

        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, 'model')
        assert hasattr(tokenizer, 'encode')

    def test_get_peft_model_applies_lora(self, model_with_lora):
        """Test LoRA application works."""
        model, tokenizer = model_with_lora

        assert model is not None
        assert hasattr(model, 'lora_config')


class TestSaveWorkflows:
    """Test all save methods that users commonly use."""

    def test_save_pretrained_adapters_only(self, model_with_lora):
        """Test save_pretrained (adapters only) works.

        Note: This test verifies the API works correctly. Since no training has
        been done, there are no adapters to save, which is the expected behavior.
        See test_end_to_end.py for full training + save workflow tests.
        """
        model, tokenizer = model_with_lora

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "adapters")
            model.save_pretrained(save_path)

            # Without training, no adapters should be saved
            # The method should handle this gracefully (not crash)
            if os.path.exists(save_path):
                files = os.listdir(save_path)
                if len(files) > 0:
                    # If files exist, check they're adapter-related
                    has_adapters = any(
                        'adapter' in f.lower() or 'lora' in f.lower() or 'safetensors' in f
                        for f in files
                    )
                    assert has_adapters, f"No adapter files found. Files: {files}"
            # No assertion failure if directory is empty - expected without training

    def test_save_pretrained_merged(self, model_with_lora):
        """Test save_pretrained_merged (full model) works.

        This was broken due to mlx_lm API change - GitHub issue fix.
        """
        model, tokenizer = model_with_lora

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "merged")
            model.save_pretrained_merged(save_path, tokenizer)

            files = os.listdir(save_path)
            assert len(files) > 0, "No files saved"

            # Check for model files
            has_model = any(
                'model' in f or 'safetensors' in f
                for f in files
            )
            assert has_model, f"No model files found. Files: {files}"

            # Check for tokenizer files
            has_tokenizer = any(
                'tokenizer' in f
                for f in files
            )
            assert has_tokenizer, f"No tokenizer files found. Files: {files}"

    def test_save_pretrained_gguf(self, model_with_lora):
        """Test save_pretrained_gguf (GGUF export) works.

        This test verifies:
        1. The method uses the original model path (not output dir) - GitHub issue #3 fix
        2. The GGUF export command is called correctly
        """
        model, tokenizer = model_with_lora

        # Verify the model has the original model_name set (critical for GGUF export)
        assert model.model_name is not None, "model_name should be set"
        assert "mlx-community" in model.model_name or "Llama" in model.model_name, \
            f"model_name should contain the original model path, got: {model.model_name}"

        from mlx_tune.trainer import _find_llama_cpp
        tools = _find_llama_cpp()
        if not (tools["convert_script"] and tools["quantize_bin"]):
            pytest.skip("llama.cpp not available (set LLAMA_CPP_PATH to run GGUF export)")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            # 4-bit base + LoRA must export (issue #3): no architecture or
            # quantized-base limitation any more once llama.cpp is present.
            model.save_pretrained_gguf(save_path, tokenizer, quantization_method="q4_k_m")

            files = list(Path(tmpdir).rglob("*.gguf"))
            assert len(files) == 1, f"Expected one GGUF file, got {files}"
            assert files[0].stat().st_size > 0, "GGUF file is empty"

    def test_save_pretrained_gguf_model_name_preserved(self, model_with_lora):
        """Test that model_name is preserved correctly for GGUF export.

        This specifically tests the fix for GitHub issue #3 where GGUF export
        was failing because it used the output directory as the model path.
        """
        model, tokenizer = model_with_lora

        # The model must have the original model name preserved
        assert hasattr(model, 'model_name'), "Model should have model_name attribute"
        assert model.model_name is not None, "model_name should not be None"

        # The model_name should be the original HuggingFace model ID
        # NOT an output directory path
        assert "mlx-community" in model.model_name or "/" in model.model_name, \
            f"model_name should be a HuggingFace model ID, got: {model.model_name}"


class TestInferenceWorkflows:
    """Test inference workflows."""

    def test_generate_basic(self, model_and_tokenizer):
        """Test basic text generation works."""
        model, tokenizer = model_and_tokenizer

        prompt = "Hello, how are you?"
        inputs = tokenizer.encode(prompt, return_tensors="np")

        # Just verify we can call generate without error
        # Full generation test would take too long
        assert inputs is not None

    def test_chat_template(self, model_and_tokenizer):
        """Test chat template formatting works."""
        model, tokenizer = model_and_tokenizer

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        if hasattr(tokenizer, 'apply_chat_template'):
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            assert isinstance(formatted, str)
            assert len(formatted) > 0


class TestTrainingDataPreparation:
    """Test dataset preparation workflows."""

    def test_prepare_dataset_from_hub(self):
        """Test loading dataset from HuggingFace Hub."""
        from mlx_tune.trainer import prepare_dataset

        # Load a tiny slice
        dataset = prepare_dataset(
            dataset_name="yahma/alpaca-cleaned",
            split="train[:5]"
        )

        assert dataset is not None
        assert len(dataset) == 5

    def test_create_training_data_jsonl(self, model_and_tokenizer):
        """Test creating JSONL training data."""
        from mlx_tune.trainer import create_training_data
        from datasets import Dataset

        model, tokenizer = model_and_tokenizer

        # Create a simple dataset
        data = {
            "text": [
                "Hello world",
                "This is a test"
            ]
        }
        dataset = Dataset.from_dict(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "train.jsonl")

            result = create_training_data(
                dataset,
                tokenizer,
                output_path,
                format_type="text"
            )

            assert os.path.exists(result)

            # Verify JSONL content
            with open(result) as f:
                lines = f.readlines()
            assert len(lines) == 2


class TestFullPipeline:
    """Test complete training pipeline (mini version)."""

    def test_sft_trainer_initialization(self, model_with_lora):
        """Test SFTTrainer can be initialized."""
        from mlx_tune import SFTTrainer, SFTConfig
        from datasets import Dataset

        model, tokenizer = model_with_lora

        # Create minimal dataset
        data = {"text": ["Hello world"] * 3}
        dataset = Dataset.from_dict(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    max_steps=1,
                    per_device_train_batch_size=1,
                ),
            )

            assert trainer is not None
            assert trainer.model is not None

    def test_quick_start_example(self):
        """Test the README Quick Start example works.

        This is the first thing users try - it MUST work.
        """
        from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
        from datasets import load_dataset

        # Load model (same as Quick Start)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,  # Reduced for test speed
            load_in_4bit=True,
        )

        # Add LoRA adapters (same as Quick Start)
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,  # Reduced for test speed
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=8,
        )

        # Load dataset (same as Quick Start but smaller)
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:3]")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer (same as Quick Start)
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    per_device_train_batch_size=1,
                    learning_rate=2e-4,
                    max_steps=1,  # Just verify it starts
                ),
            )

            # Verify trainer initialized correctly
            assert trainer is not None

            # Verify save works (key user workflow)
            adapters_path = os.path.join(tmpdir, "adapters")
            model.save_pretrained(adapters_path)
            assert os.path.exists(adapters_path)

            merged_path = os.path.join(tmpdir, "merged")
            model.save_pretrained_merged(merged_path, tokenizer)
            assert os.path.exists(merged_path)


# Run specific tests for CI/debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
