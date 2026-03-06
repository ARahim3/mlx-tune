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


class TestRewardConfig:
    def test_rewardconfig_defaults(self):
        from mlx_tune import RewardConfig

        config = RewardConfig()

        assert config.learning_rate == 5e-6
        assert config.regression_loss_type == "mse"
        assert config.dataset_mode is None


class TestGRPOConfig:
    """Test GRPOConfig class."""

    def test_grpoconfig_defaults(self):
        """Test GRPOConfig has correct defaults."""
        from mlx_tune import GRPOConfig

        config = GRPOConfig()

        assert config.loss_type == "grpo"
        assert config.advantage_mode == "group_zscore"
        assert config.advantage_estimator == "group_zscore"
        assert config.num_generations == 4
        assert config.temperature == 0.7
        assert config.beta == 0.04
        assert config.kl_beta == 0.04
        assert config.reward_source == "auto"

    def test_grpoconfig_with_reward_fn(self):
        """Test GRPOConfig with custom reward function."""
        from mlx_tune import GRPOConfig

        def custom_reward(response, prompt):
            return 1.0

        config = GRPOConfig(reward_fn=custom_reward, num_generations=8)

        assert config.reward_fn is not None
        assert config.num_generations == 8

    def test_grpoconfig_aliases_and_to_dict(self):
        from mlx_tune import GRPOConfig

        config = GRPOConfig(
            generations_per_prompt=6,
            advantage_estimator="rloo",
            reward_fn=lambda *_: 1.0,
        )

        assert config.num_generations == 6
        assert config.advantage_mode == "rloo"
        assert config.to_dict()["num_generations"] == 6
        assert "reward_fn" not in config.to_dict()

    def test_grpoconfig_variant_defaults_match_loss_family(self):
        from mlx_tune import GRPOConfig

        dapo = GRPOConfig(loss_type="dapo")
        dr_grpo = GRPOConfig(loss_type="dr_grpo")

        assert dapo.mask_truncated_completions is True
        assert dapo.epsilon_high == 0.28
        assert dr_grpo.scale_rewards is False
        assert dr_grpo.epsilon_low == dr_grpo.clip_epsilon
        assert dr_grpo.epsilon_high == dr_grpo.clip_epsilon


class TestPPOAndOnlineDPOConfig:
    def test_ppoconfig_defaults(self):
        from mlx_tune import PPOConfig

        config = PPOConfig()

        assert config.ppo_epochs == 2
        assert config.minibatch_reuse_steps == 2
        assert config.value_learning_rate == config.learning_rate

    def test_online_dpoconfig_defaults(self):
        from mlx_tune import OnlineDPOConfig

        config = OnlineDPOConfig()

        assert config.num_generations == 4
        assert config.beta == 0.1

    def test_new_offline_config_exports(self):
        from mlx_tune import KTOConfig, SimPOConfig

        kto = KTOConfig()
        simpo = SimPOConfig()

        assert kto.beta == 0.1
        assert simpo.gamma == 0.5


class TestTrainerInitialization:
    """Test trainer initialization (without actual model loading)."""

    def test_imports_work(self):
        """Test all trainers can be imported."""
        from mlx_tune import (
            SFTTrainer,
            SFTConfig,
            RewardTrainer,
            RewardConfig,
            DPOTrainer,
            DPOConfig,
            ORPOTrainer,
            ORPOConfig,
            GRPOTrainer,
            GRPOConfig,
            PPOTrainer,
            PPOConfig,
            OnlineDPOTrainer,
            OnlineDPOConfig,
            KTOTrainer,
            SimPOTrainer,
        )

        # Just verify imports work
        assert SFTTrainer is not None
        assert RewardTrainer is not None
        assert DPOTrainer is not None
        assert ORPOTrainer is not None
        assert GRPOTrainer is not None
        assert PPOTrainer is not None
        assert OnlineDPOTrainer is not None
        assert KTOTrainer is not None
        assert SimPOTrainer is not None


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

    def test_prepare_rl_dataset_import(self):
        from mlx_tune import prepare_rl_dataset

        assert prepare_rl_dataset is not None

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

    def test_create_reward_function_composition(self):
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function(
            rewards=[
                {"name": "simple", "source": "simple", "weight": 0.25},
                {"name": "length", "source": "length", "weight": 0.75},
            ]
        )

        result = reward_fn.evaluate({"completion_text": "The answer is 42", "reward_context": "42"})

        assert result["reward"] > 0.0
        assert set(result["components"]) >= {"simple", "length"}


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
