"""
Unit tests for trainers in mlx_tune
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestSFTConfig:
    """Test SFTConfig class."""

    def test_sftconfig_defaults(self):
        """Test SFTConfig has correct defaults."""
        from mlx_tune import SFTConfig

        config = SFTConfig()

        assert config.output_dir == "./outputs"
        assert config.per_device_train_batch_size == 2
        assert config.learning_rate == 2e-4
        assert config.lr_scheduler_type == "cosine"
        assert config.use_native_training is True
        assert config.grad_checkpoint is False

    def test_sftconfig_custom_values(self):
        """Test SFTConfig with custom values."""
        from mlx_tune import SFTConfig

        config = SFTConfig(
            output_dir="./custom_output",
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            use_native_training=False,
        )

        assert config.output_dir == "./custom_output"
        assert config.learning_rate == 1e-5
        assert config.per_device_train_batch_size == 4
        assert config.use_native_training is False

    def test_sftconfig_to_dict(self):
        """Test SFTConfig to_dict method."""
        from mlx_tune import SFTConfig

        config = SFTConfig(learning_rate=1e-4)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "learning_rate" in config_dict
        assert config_dict["learning_rate"] == 1e-4


class TestDPOConfig:
    """Test DPOConfig class."""

    def test_dpoconfig_defaults(self):
        """Test DPOConfig has correct defaults."""
        from mlx_tune import DPOConfig

        config = DPOConfig()

        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"
        assert config.learning_rate == 5e-7

    def test_dpoconfig_custom_beta(self):
        """Test DPOConfig with custom beta."""
        from mlx_tune import DPOConfig

        config = DPOConfig(beta=0.5)

        assert config.beta == 0.5


class TestGRPOConfig:
    """Test GRPOConfig class."""

    def test_grpoconfig_defaults(self):
        """Test GRPOConfig has correct defaults."""
        from mlx_tune import GRPOConfig

        config = GRPOConfig()

        assert config.loss_type == "grpo"
        assert config.num_generations == 4
        assert config.temperature == 0.7
        assert config.beta == 0.04

    def test_grpoconfig_with_reward_fn(self):
        """Test GRPOConfig with custom reward function."""
        from mlx_tune import GRPOConfig

        def custom_reward(response, prompt):
            return 1.0

        config = GRPOConfig(reward_fn=custom_reward, num_generations=8)

        assert config.reward_fn is not None
        assert config.num_generations == 8


class TestTrainerInitialization:
    """Test trainer initialization (without actual model loading)."""

    def test_imports_work(self):
        """Test all trainers can be imported."""
        from mlx_tune import (
            SFTTrainer,
            SFTConfig,
            DPOTrainer,
            DPOConfig,
            ORPOTrainer,
            ORPOConfig,
            GRPOTrainer,
            GRPOConfig,
            KTOTrainer,
            SimPOTrainer,
        )

        # Just verify imports work
        assert SFTTrainer is not None
        assert DPOTrainer is not None
        assert ORPOTrainer is not None
        assert GRPOTrainer is not None
        assert KTOTrainer is not None
        assert SimPOTrainer is not None

    def test_sft_native_training_forwards_gradient_accumulation(self, monkeypatch, tmp_path):
        """Test SFTTrainer forwards gradient accumulation to MLX-LM TrainingArgs."""
        from mlx_tune import SFTConfig, SFTTrainer
        import mlx_tune.sft_trainer as sft_trainer

        if not sft_trainer.HAS_NATIVE_TRAINING:
            pytest.skip("Native MLX-LM training components are not available")

        captured = {}

        def fake_train(**kwargs):
            captured["args"] = kwargs["args"]

        trainer = SFTTrainer(
            model=object(),
            train_dataset=[{"text": "hello"}],
            args=SFTConfig(
                output_dir=str(tmp_path / "output"),
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=7,
            ),
            adapter_path=str(tmp_path / "adapter"),
        )

        monkeypatch.setattr(trainer, "_prepare_training_data", lambda: str(tmp_path))
        monkeypatch.setattr(trainer, "_get_lr_schedule", lambda: 1e-4)
        monkeypatch.setattr(trainer, "_save_adapter_config", lambda: None)
        monkeypatch.setattr(sft_trainer, "mlx_load_dataset", lambda **kwargs: ([1], [1], None))
        monkeypatch.setattr(sft_trainer, "CacheDataset", lambda dataset: dataset)
        monkeypatch.setattr(sft_trainer, "mlx_train", fake_train)

        result = trainer._train_native()

        assert result["status"] == "success"
        assert captured["args"].grad_accumulation_steps == 7


class TestLossFunctionImports:
    """Test loss function imports."""

    def test_loss_imports(self):
        """Test all loss functions can be imported."""
        from mlx_tune import (
            compute_log_probs,
            compute_log_probs_with_lengths,
            dpo_loss,
            orpo_loss,
            kto_loss,
            simpo_loss,
            sft_loss,
            grpo_loss,
            grpo_batch_loss,
            compute_reference_logprobs,
        )

        # Verify imports
        assert dpo_loss is not None
        assert orpo_loss is not None
        assert grpo_loss is not None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_prepare_dataset_import(self):
        """Test prepare_dataset can be imported."""
        from mlx_tune import prepare_dataset
        assert prepare_dataset is not None

    def test_prepare_preference_dataset_import(self):
        """Test prepare_preference_dataset can be imported."""
        from mlx_tune import prepare_preference_dataset
        assert prepare_preference_dataset is not None

    def test_create_reward_function_simple(self):
        """Test create_reward_function with simple type."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("simple")

        # Test the reward function
        result = reward_fn("The answer is 42", "42")
        assert result == 1.0

        result = reward_fn("I don't know", "42")
        assert result == 0.0

    def test_create_reward_function_math(self):
        """Test create_reward_function with math type."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("math")

        # Test the reward function
        result = reward_fn("The answer is 42", "42")
        assert result == 1.0

        result = reward_fn("The answer is 10", "42")
        assert result == 0.0

    def test_create_reward_function_length(self):
        """Test create_reward_function with length type."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("length")

        # Short response
        short_result = reward_fn("Hi", "")
        assert short_result == 0.2

        # Medium response
        medium_result = reward_fn(" ".join(["word"] * 30), "")
        assert medium_result == 0.5


class TestExportFunctions:
    """Test export utility functions."""

    def test_get_training_config(self):
        """Test get_training_config returns correct structure."""
        from mlx_tune import get_training_config

        config = get_training_config(
            output_dir="./test_output",
            num_train_epochs=5,
            learning_rate=1e-4,
        )

        assert isinstance(config, dict)
        assert config["output_dir"] == "./test_output"
        assert config["num_train_epochs"] == 5
        assert config["learning_rate"] == 1e-4
        assert "lora_r" in config
        assert "lora_alpha" in config
