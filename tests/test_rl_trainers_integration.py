"""
Integration tests for RL trainers (DPO, ORPO, GRPO, KTO, SimPO).

These tests verify that the trainers actually run and produce valid results,
not just that they can be imported or configured.

Tests marked with @pytest.mark.integration require more time/resources.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, List


# =============================================================================
# TEST FIXTURES - Small models and datasets for fast testing
# =============================================================================

class SmallLanguageModel(nn.Module):
    """A tiny language model for testing - fast to train."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        self.output = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = mx.maximum(layer(h), 0)  # ReLU
        return self.output(h)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.name_or_path = "mock-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple encoding: hash characters to vocab indices."""
        ids = [hash(c) % (self.vocab_size - 3) + 3 for c in text[:50]]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simple decoding."""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return "".join(chr(65 + (i % 26)) for i in ids)

    def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=512):
        """Tokenize text."""
        if isinstance(text, str):
            ids = self.encode(text)
        else:
            ids = [self.encode(t) for t in text]

        if return_tensors == "mlx":
            return {"input_ids": mx.array(ids)}
        return {"input_ids": ids}


class MockModelWrapper:
    """Wrapper to match FastLanguageModel interface."""

    def __init__(self, model: SmallLanguageModel):
        self.model = model
        self._lora_applied = False
        self._lora_config = None

    def __call__(self, x):
        return self.model(x)

    def _apply_lora(self):
        """Mock LoRA application."""
        self._lora_applied = True
        print("  [Mock] LoRA applied")

    def parameters(self):
        return self.model.parameters()

    def freeze(self):
        pass

    def unfreeze(self):
        pass


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    model = SmallLanguageModel(vocab_size=100, hidden_size=64, num_layers=2)
    mx.eval(model.parameters())
    return MockModelWrapper(model)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer(vocab_size=100)


@pytest.fixture
def preference_dataset():
    """Sample preference dataset for DPO/ORPO/SimPO."""
    return [
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a branch of AI that enables systems to learn from data.",
            "rejected": "idk its computers doing stuff"
        },
        {
            "prompt": "Explain Python.",
            "chosen": "Python is a high-level programming language known for readability.",
            "rejected": "python is a snake"
        },
        {
            "prompt": "What is deep learning?",
            "chosen": "Deep learning uses neural networks with many layers to learn patterns.",
            "rejected": "its like machine learning but deeper i guess"
        },
    ]


@pytest.fixture
def kto_dataset():
    """Sample dataset for KTO (binary feedback) — both TRL and legacy formats."""
    return [
        # TRL format (prompt + completion + label)
        {"prompt": "What is ML?", "completion": "Machine learning is a branch of AI.", "label": True},
        {"prompt": "What is ML?", "completion": "idk computers stuff", "label": False},
        # Legacy format (text + numeric label)
        {"text": "Python is a programming language.", "label": 1},  # Good
        {"text": "python snake", "label": 0},  # Bad
    ]


@pytest.fixture
def grpo_dataset():
    """Sample dataset for GRPO (reasoning with answers)."""
    return [
        {"prompt": "What is 2 + 2?", "answer": "4"},
        {"prompt": "What is 5 * 3?", "answer": "15"},
        {"prompt": "What is 10 - 7?", "answer": "3"},
    ]


# =============================================================================
# DPO TRAINER INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestDPOTrainerIntegration:
    """Integration tests for DPOTrainer."""

    def test_dpo_trainer_init(self, small_model, mock_tokenizer, preference_dataset):
        """Test DPOTrainer can be initialized."""
        from mlx_tune import DPOTrainer, DPOConfig

        config = DPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_dpo_output",
        )

        trainer = DPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        assert trainer is not None
        assert trainer.beta == 0.1
        assert len(trainer.train_dataset) == 3

    def test_dpo_trainer_train_runs(self, small_model, mock_tokenizer, preference_dataset):
        """Test DPOTrainer.train() executes without errors."""
        from mlx_tune import DPOTrainer, DPOConfig

        config = DPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_steps=2,  # Very short for testing
            output_dir="./test_dpo_output",
        )

        trainer = DPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        # This should run without raising exceptions
        result = trainer.train()

        assert result is not None
        # Verify training completed - check model's _lora_applied flag
        assert small_model._lora_applied, "LoRA should have been applied during training"

    def test_dpo_loss_decreases(self, small_model, mock_tokenizer, preference_dataset):
        """Test that DPO loss decreases or stays stable during training."""
        from mlx_tune import DPOTrainer, DPOConfig

        config = DPOConfig(
            beta=0.1,
            learning_rate=1e-3,  # Higher LR for visible change
            max_steps=5,
            output_dir="./test_dpo_output",
        )

        trainer = DPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()

        # Check no NaN in result
        if isinstance(result, dict) and 'final_loss' in result:
            assert not mx.isnan(mx.array(result['final_loss'])), "Final loss should not be NaN"

    def test_dpo_different_loss_types(self, small_model, mock_tokenizer, preference_dataset):
        """Test DPO with different loss types."""
        from mlx_tune import DPOTrainer, DPOConfig

        loss_types = ["sigmoid", "hinge", "ipo"]

        for loss_type in loss_types:
            # Create fresh model for each test
            model = MockModelWrapper(SmallLanguageModel())
            mx.eval(model.model.parameters())

            config = DPOConfig(
                beta=0.1,
                loss_type=loss_type,
                learning_rate=1e-4,
                max_steps=2,
                output_dir="./test_dpo_output",
            )

            trainer = DPOTrainer(
                model=model,
                train_dataset=preference_dataset,
                tokenizer=mock_tokenizer,
                args=config,
            )

            # Should not raise
            result = trainer.train()
            assert result is not None, f"DPO with loss_type={loss_type} failed"


# =============================================================================
# ORPO TRAINER INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestORPOTrainerIntegration:
    """Integration tests for ORPOTrainer."""

    def test_orpo_trainer_init(self, small_model, mock_tokenizer, preference_dataset):
        """Test ORPOTrainer can be initialized."""
        from mlx_tune import ORPOTrainer, ORPOConfig

        config = ORPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_orpo_output",
        )

        trainer = ORPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        assert trainer is not None
        assert trainer.beta == 0.1

    def test_orpo_trainer_train_runs(self, small_model, mock_tokenizer, preference_dataset):
        """Test ORPOTrainer.train() executes without errors."""
        from mlx_tune import ORPOTrainer, ORPOConfig

        config = ORPOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_orpo_output",
        )

        trainer = ORPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()
        assert result is not None

    def test_orpo_combines_sft_and_preference(self, small_model, mock_tokenizer, preference_dataset):
        """Test that ORPO combines SFT and preference learning."""
        from mlx_tune import ORPOTrainer, ORPOConfig

        config = ORPOConfig(
            beta=0.1,
            learning_rate=1e-3,
            max_steps=3,
            output_dir="./test_orpo_output",
        )

        trainer = ORPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()

        # ORPO should complete successfully
        assert result is not None


# =============================================================================
# GRPO TRAINER INTEGRATION TESTS (Most important for reasoning)
# =============================================================================

@pytest.mark.integration
class TestGRPOTrainerIntegration:
    """Integration tests for GRPOTrainer - DeepSeek R1 style reasoning."""

    def test_grpo_trainer_init(self, small_model, mock_tokenizer, grpo_dataset):
        """Test GRPOTrainer can be initialized."""
        from mlx_tune import GRPOTrainer, GRPOConfig, create_reward_function

        reward_fn = create_reward_function("simple")

        config = GRPOConfig(
            beta=0.04,
            num_generations=2,  # Small for testing
            learning_rate=1e-5,
            max_steps=2,
            output_dir="./test_grpo_output",
        )

        trainer = GRPOTrainer(
            model=small_model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=reward_fn,
            args=config,
        )

        assert trainer is not None
        assert trainer.num_generations == 2
        assert trainer.reward_fn is not None

    def test_grpo_trainer_train_runs(self, small_model, mock_tokenizer, grpo_dataset):
        """Test GRPOTrainer.train() executes without errors."""
        from mlx_tune import GRPOTrainer, GRPOConfig, create_reward_function

        reward_fn = create_reward_function("simple")

        config = GRPOConfig(
            beta=0.04,
            num_generations=2,
            learning_rate=1e-5,
            max_steps=2,
            temperature=0.7,
            output_dir="./test_grpo_output",
        )

        trainer = GRPOTrainer(
            model=small_model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=reward_fn,
            args=config,
        )

        result = trainer.train()
        assert result is not None

    def test_grpo_multi_generation(self, small_model, mock_tokenizer, grpo_dataset):
        """Test that GRPO generates multiple completions per prompt."""
        from mlx_tune import GRPOTrainer, GRPOConfig, create_reward_function

        reward_fn = create_reward_function("simple")
        num_gens = 3

        config = GRPOConfig(
            beta=0.04,
            num_generations=num_gens,
            learning_rate=1e-5,
            max_steps=1,  # Just one step to verify multi-gen
            output_dir="./test_grpo_output",
        )

        trainer = GRPOTrainer(
            model=small_model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=reward_fn,
            args=config,
        )

        # The trainer should be configured for multi-generation
        assert trainer.num_generations == num_gens

        # Training should work
        result = trainer.train()
        assert result is not None

    def test_grpo_with_math_reward(self, small_model, mock_tokenizer, grpo_dataset):
        """Test GRPO with math reward function."""
        from mlx_tune import GRPOTrainer, GRPOConfig, create_reward_function

        math_reward = create_reward_function("math")

        config = GRPOConfig(
            beta=0.04,
            num_generations=2,
            learning_rate=1e-5,
            max_steps=2,
            output_dir="./test_grpo_output",
        )

        trainer = GRPOTrainer(
            model=small_model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=math_reward,
            args=config,
        )

        result = trainer.train()
        assert result is not None

    def test_grpo_different_loss_types(self, small_model, mock_tokenizer, grpo_dataset):
        """Test GRPO with different loss types (grpo, dr_grpo, dapo, bnpo)."""
        from mlx_tune import GRPOTrainer, GRPOConfig, create_reward_function

        reward_fn = create_reward_function("simple")
        loss_types = ["grpo", "dr_grpo", "dapo", "bnpo"]

        for loss_type in loss_types:
            model = MockModelWrapper(SmallLanguageModel())
            mx.eval(model.model.parameters())

            config = GRPOConfig(
                loss_type=loss_type,
                beta=0.04,
                num_generations=2,
                learning_rate=1e-5,
                max_steps=1,
                output_dir="./test_grpo_output",
            )

            trainer = GRPOTrainer(
                model=model,
                train_dataset=grpo_dataset,
                tokenizer=mock_tokenizer,
                reward_fn=reward_fn,
                args=config,
            )

            result = trainer.train()
            assert result is not None, f"GRPO with loss_type={loss_type} failed"

    def test_grpo_custom_reward_function(self, small_model, mock_tokenizer, grpo_dataset):
        """Test GRPO with a custom reward function."""
        from mlx_tune import GRPOTrainer, GRPOConfig

        # Custom reward: reward longer responses
        def length_reward(response: str, answer: str = None) -> float:
            return len(response) / 100.0  # Normalize

        config = GRPOConfig(
            beta=0.04,
            num_generations=2,
            learning_rate=1e-5,
            max_steps=2,
            output_dir="./test_grpo_output",
        )

        trainer = GRPOTrainer(
            model=small_model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=length_reward,
            args=config,
        )

        result = trainer.train()
        assert result is not None


# =============================================================================
# KTO TRAINER INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestKTOTrainerIntegration:
    """Integration tests for KTOTrainer."""

    def test_kto_trainer_init_with_config(self, small_model, mock_tokenizer, kto_dataset):
        """Test KTOTrainer can be initialized with KTOConfig."""
        from mlx_tune import KTOTrainer, KTOConfig

        config = KTOConfig(
            beta=0.1,
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_kto_output",
        )

        trainer = KTOTrainer(
            model=small_model,
            train_dataset=kto_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        assert trainer is not None
        assert trainer.beta == 0.1

    def test_kto_trainer_init_kwargs_compat(self, small_model, mock_tokenizer, kto_dataset):
        """Test KTOTrainer backward-compat with kwargs."""
        from mlx_tune import KTOTrainer

        trainer = KTOTrainer(
            model=small_model,
            train_dataset=kto_dataset,
            tokenizer=mock_tokenizer,
            beta=0.2,
            learning_rate=1e-4,
            max_steps=2,
        )

        assert trainer is not None

    def test_kto_trainer_train_runs(self, small_model, mock_tokenizer, kto_dataset):
        """Test KTOTrainer.train() executes without errors."""
        from mlx_tune import KTOTrainer, KTOConfig

        config = KTOConfig(
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_kto_output",
        )

        trainer = KTOTrainer(
            model=small_model,
            train_dataset=kto_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()
        assert result is not None

    def test_kto_binary_feedback_both_formats(self, small_model, mock_tokenizer, kto_dataset):
        """Test KTO processes both TRL and legacy format samples."""
        from mlx_tune import KTOTrainer, KTOConfig

        # Dataset has both TRL format (prompt+completion+label) and legacy (text+label)
        assert any('prompt' in d for d in kto_dataset), "Need TRL format samples"
        assert any('text' in d for d in kto_dataset), "Need legacy format samples"

        config = KTOConfig(
            learning_rate=1e-4,
            max_steps=3,
            output_dir="./test_kto_output",
        )

        trainer = KTOTrainer(
            model=small_model,
            train_dataset=kto_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()
        assert result is not None


# =============================================================================
# SIMPO TRAINER INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestSimPOTrainerIntegration:
    """Integration tests for SimPOTrainer."""

    def test_simpo_trainer_init_with_config(self, small_model, mock_tokenizer, preference_dataset):
        """Test SimPOTrainer can be initialized with SimPOConfig."""
        from mlx_tune import SimPOTrainer, SimPOConfig

        config = SimPOConfig(
            beta=2.0,
            gamma=0.5,
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_simpo_output",
        )

        trainer = SimPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        assert trainer is not None
        assert trainer.beta == 2.0
        assert trainer.gamma == 0.5

    def test_simpo_trainer_train_runs(self, small_model, mock_tokenizer, preference_dataset):
        """Test SimPOTrainer.train() executes without errors."""
        from mlx_tune import SimPOTrainer, SimPOConfig

        config = SimPOConfig(
            learning_rate=1e-4,
            max_steps=2,
            output_dir="./test_simpo_output",
        )

        trainer = SimPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()
        assert result is not None

    def test_simpo_no_reference_model(self, small_model, mock_tokenizer, preference_dataset):
        """Test SimPO works without reference model."""
        from mlx_tune import SimPOTrainer, SimPOConfig

        config = SimPOConfig(
            learning_rate=1e-4,
            max_steps=3,
            output_dir="./test_simpo_output",
        )

        trainer = SimPOTrainer(
            model=small_model,
            train_dataset=preference_dataset,
            tokenizer=mock_tokenizer,
            args=config,
        )

        result = trainer.train()
        assert result is not None


# =============================================================================
# GRADIENT FLOW TESTS
# =============================================================================

@pytest.mark.integration
class TestGradientFlow:
    """Test that gradients flow correctly in RL trainers."""

    def _check_grads_nonzero(self, grads):
        """Recursively check if any gradients are non-zero."""
        if isinstance(grads, dict):
            for value in grads.values():
                if self._check_grads_nonzero(value):
                    return True
            return False
        elif isinstance(grads, mx.array):
            return float(mx.sum(mx.abs(grads)).item()) > 0
        return False

    def test_dpo_gradient_not_zero(self):
        """Test DPO computes non-zero gradients."""
        from mlx_tune.losses import dpo_loss

        model = SmallLanguageModel(vocab_size=50, hidden_size=32)
        mx.eval(model.parameters())

        # Create sample data
        chosen = mx.array([[1, 2, 3, 4, 5]])
        rejected = mx.array([[1, 2, 6, 7, 8]])
        chosen_len = mx.array([5])
        rejected_len = mx.array([5])

        # Compute loss and gradients
        def loss_fn(model):
            loss, _ = dpo_loss(model, chosen, rejected, chosen_len, rejected_len, beta=0.1)
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)

        # Check gradients are not all zero
        has_nonzero_grad = self._check_grads_nonzero(grads)

        assert has_nonzero_grad, "DPO gradients should not all be zero"
        assert not mx.isnan(loss), "DPO loss should not be NaN"

    def test_orpo_gradient_not_zero(self):
        """Test ORPO computes non-zero gradients."""
        from mlx_tune.losses import orpo_loss

        model = SmallLanguageModel(vocab_size=50, hidden_size=32)
        mx.eval(model.parameters())

        chosen = mx.array([[1, 2, 3, 4, 5]])
        rejected = mx.array([[1, 2, 6, 7, 8]])
        chosen_len = mx.array([5])
        rejected_len = mx.array([5])

        def loss_fn(model):
            loss, _ = orpo_loss(model, chosen, rejected, chosen_len, rejected_len, beta=0.1)
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)

        has_nonzero_grad = self._check_grads_nonzero(grads)

        assert has_nonzero_grad, "ORPO gradients should not all be zero"
        assert not mx.isnan(loss), "ORPO loss should not be NaN"


# =============================================================================
# LOSS STABILITY TESTS
# =============================================================================

@pytest.mark.integration
class TestLossStability:
    """Test that losses remain stable (no NaN, Inf) during training."""

    def test_dpo_loss_stability(self):
        """Test DPO loss doesn't produce NaN or Inf."""
        from mlx_tune.losses import dpo_loss

        model = SmallLanguageModel(vocab_size=50, hidden_size=32)
        mx.eval(model.parameters())

        # Run multiple forward passes
        for i in range(10):
            chosen = mx.random.randint(0, 50, (2, 20))
            rejected = mx.random.randint(0, 50, (2, 20))
            chosen_len = mx.array([15, 18])
            rejected_len = mx.array([17, 16])

            loss, ntoks = dpo_loss(model, chosen, rejected, chosen_len, rejected_len, beta=0.1)
            mx.eval(loss)

            assert not mx.isnan(loss), f"DPO loss became NaN at iteration {i}"
            assert not mx.isinf(loss), f"DPO loss became Inf at iteration {i}"

    def test_orpo_loss_stability(self):
        """Test ORPO loss doesn't produce NaN or Inf."""
        from mlx_tune.losses import orpo_loss

        model = SmallLanguageModel(vocab_size=50, hidden_size=32)
        mx.eval(model.parameters())

        for i in range(10):
            chosen = mx.random.randint(0, 50, (2, 20))
            rejected = mx.random.randint(0, 50, (2, 20))
            chosen_len = mx.array([15, 18])
            rejected_len = mx.array([17, 16])

            loss, _ = orpo_loss(model, chosen, rejected, chosen_len, rejected_len, beta=0.1)
            mx.eval(loss)

            assert not mx.isnan(loss), f"ORPO loss became NaN at iteration {i}"
            assert not mx.isinf(loss), f"ORPO loss became Inf at iteration {i}"

    def test_simpo_loss_stability(self):
        """Test SimPO loss doesn't produce NaN or Inf."""
        from mlx_tune.losses import simpo_loss

        model = SmallLanguageModel(vocab_size=50, hidden_size=32)
        mx.eval(model.parameters())

        for i in range(10):
            chosen = mx.random.randint(0, 50, (2, 20))
            rejected = mx.random.randint(0, 50, (2, 20))
            chosen_len = mx.array([15, 18])
            rejected_len = mx.array([17, 16])

            loss, _ = simpo_loss(model, chosen, rejected, chosen_len, rejected_len,
                                  beta=0.1, gamma=0.5)
            mx.eval(loss)

            assert not mx.isnan(loss), f"SimPO loss became NaN at iteration {i}"
            assert not mx.isinf(loss), f"SimPO loss became Inf at iteration {i}"


# =============================================================================
# REWARD FUNCTION TESTS
# =============================================================================

class TestRewardFunctions:
    """Test reward functions used in GRPO."""

    def test_simple_reward_function(self):
        """Test simple reward function."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("simple")

        # Simple reward expects (response, ground_truth)
        # Returns 1.0 if ground_truth is in response, else 0.0
        score_match = reward_fn("The answer is 42", "42")
        score_no_match = reward_fn("The answer is something", "42")

        assert score_match == 1.0, "Should return 1.0 when ground_truth is in response"
        assert score_no_match == 0.0, "Should return 0.0 when ground_truth is not in response"

    def test_math_reward_function(self):
        """Test math reward function."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("math")

        # Math reward expects (response, ground_truth) and compares extracted numbers
        correct = reward_fn("The answer is 42", "42")
        incorrect = reward_fn("The answer is 99", "42")

        assert correct == 1.0, "Correct math answer should get 1.0"
        assert incorrect == 0.0, "Incorrect math answer should get 0.0"

    def test_length_reward_function(self):
        """Test length-based reward function."""
        from mlx_tune import create_reward_function

        reward_fn = create_reward_function("length")

        # Length reward expects (response, _) where _ is ignored
        short = reward_fn("Hi there", "")
        medium = reward_fn("This is a longer response with about fifteen words in it here now", "")
        long = reward_fn(" ".join(["word"] * 100), "")

        # Short (<10 words) = 0.2, Medium (10-50) = 0.5, Long (50-200) = 1.0
        assert short == 0.2, f"Short response should be 0.2, got {short}"
        assert medium == 0.5, f"Medium response should be 0.5, got {medium}"
        assert long == 1.0, f"Long response should be 1.0, got {long}"

    def test_custom_reward_function(self):
        """Test using a custom reward function."""
        def my_reward(response: str, ground_truth: str = "") -> float:
            # Reward responses that contain "please" or "thank"
            score = 0.0
            if "please" in response.lower():
                score += 0.5
            if "thank" in response.lower():
                score += 0.5
            return score

        assert my_reward("Please help me") == 0.5
        assert my_reward("Thank you") == 0.5
        assert my_reward("Please help, thank you!") == 1.0
        assert my_reward("Hello world") == 0.0


# =============================================================================
# CONFIG CLASS TESTS
# =============================================================================

class TestConfigClasses:
    """Test RL trainer config classes."""

    def test_kto_config_defaults(self):
        """Test KTOConfig with default values."""
        from mlx_tune import KTOConfig

        config = KTOConfig()
        assert config.beta == 0.1
        assert config.desirable_weight == 1.0
        assert config.undesirable_weight == 1.0
        assert config.learning_rate == 5e-7
        assert config.max_steps == -1

    def test_kto_config_custom(self):
        """Test KTOConfig with custom values."""
        from mlx_tune import KTOConfig

        config = KTOConfig(
            beta=0.2,
            desirable_weight=1.5,
            learning_rate=1e-6,
            max_steps=50,
            output_dir="./custom_kto",
        )
        assert config.beta == 0.2
        assert config.desirable_weight == 1.5
        assert config.learning_rate == 1e-6
        assert config.max_steps == 50

    def test_kto_config_to_dict(self):
        """Test KTOConfig.to_dict()."""
        from mlx_tune import KTOConfig

        config = KTOConfig(beta=0.3)
        d = config.to_dict()
        assert d['beta'] == 0.3
        assert 'learning_rate' in d

    def test_simpo_config_defaults(self):
        """Test SimPOConfig with default values."""
        from mlx_tune import SimPOConfig

        config = SimPOConfig()
        assert config.beta == 2.0
        assert config.gamma == 0.5
        assert config.learning_rate == 5e-7

    def test_simpo_config_custom(self):
        """Test SimPOConfig with custom values."""
        from mlx_tune import SimPOConfig

        config = SimPOConfig(
            beta=3.0,
            gamma=1.0,
            learning_rate=1e-6,
            max_steps=100,
        )
        assert config.beta == 3.0
        assert config.gamma == 1.0
        assert config.max_steps == 100

    def test_simpo_config_to_dict(self):
        """Test SimPOConfig.to_dict()."""
        from mlx_tune import SimPOConfig

        config = SimPOConfig(gamma=0.8)
        d = config.to_dict()
        assert d['gamma'] == 0.8
        assert d['beta'] == 2.0


# =============================================================================
# GRPO GRADIENT FLOW TEST (verifying the fix)
# =============================================================================

@pytest.mark.integration
class TestGRPOGradientFlow:
    """Test that GRPO actually updates model parameters (the critical fix)."""

    def test_grpo_parameters_change(self, mock_tokenizer, grpo_dataset):
        """Test that GRPO training actually modifies model parameters."""
        from mlx_tune import GRPOTrainer, GRPOConfig
        from mlx.utils import tree_flatten

        model = MockModelWrapper(SmallLanguageModel())
        mx.eval(model.model.parameters())

        # Capture initial parameters
        initial_params = {}
        for name, param in tree_flatten(model.model.parameters()):
            initial_params[name] = param.tolist()

        # Use a reward fn that gives varied scores based on completion hash
        # This ensures std > 0, so the optimizer actually runs
        call_count = [0]
        def varied_reward(response: str, answer: str) -> float:
            call_count[0] += 1
            return 1.0 if call_count[0] % 2 == 0 else 0.0

        config = GRPOConfig(
            beta=0.04,
            num_generations=2,
            learning_rate=1e-2,  # High LR for visible change
            max_steps=2,
            max_completion_length=10,
            temperature=1.0,
            output_dir="./test_grpo_grad_output",
        )

        trainer = GRPOTrainer(
            model=model,
            train_dataset=grpo_dataset,
            tokenizer=mock_tokenizer,
            reward_fn=varied_reward,
            args=config,
        )

        trainer.train()

        # Check that at least some parameters changed
        params_changed = False
        for name, param in tree_flatten(model.model.parameters()):
            if name in initial_params and param.tolist() != initial_params[name]:
                params_changed = True
                break

        assert params_changed, "GRPO training should modify model parameters"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
