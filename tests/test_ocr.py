"""
Tests for OCR fine-tuning support.

Tests FastOCRModel API, OCRModelWrapper, OCR metrics (CER/WER/exact match),
dataset helpers, reward functions, configs, trainers, and the model registry.
"""

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Test OCR Metrics
# ============================================================================


class TestComputeCER:
    """Test Character Error Rate computation."""

    def test_identical_strings(self):
        from mlx_tune.ocr import compute_cer
        assert compute_cer("hello", "hello") == 0.0

    def test_completely_different(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("abc", "xyz")
        assert cer == 1.0  # 3 substitutions / 3 chars

    def test_empty_prediction_nonempty_reference(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("", "hello")
        assert cer == 1.0  # 5 deletions / 5 chars

    def test_nonempty_prediction_empty_reference(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("hello", "")
        assert cer == 5.0  # 5 insertions / 0 chars → len(prediction)

    def test_both_empty(self):
        from mlx_tune.ocr import compute_cer
        assert compute_cer("", "") == 0.0

    def test_single_substitution(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("hallo", "hello")
        assert abs(cer - 0.2) < 1e-9  # 1 sub / 5 chars

    def test_insertion(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("helloo", "hello")
        assert abs(cer - 0.2) < 1e-9  # 1 insertion / 5 chars

    def test_deletion(self):
        from mlx_tune.ocr import compute_cer
        cer = compute_cer("hell", "hello")
        assert abs(cer - 0.2) < 1e-9  # 1 deletion / 5 chars

    def test_latex_example(self):
        from mlx_tune.ocr import compute_cer
        pred = r"\frac{1}{2}"
        ref = r"\frac{1}{2}"
        assert compute_cer(pred, ref) == 0.0

    def test_latex_with_error(self):
        from mlx_tune.ocr import compute_cer
        pred = r"\frac{1}{3}"
        ref = r"\frac{1}{2}"
        cer = compute_cer(pred, ref)
        assert cer > 0.0
        assert cer < 1.0


class TestComputeWER:
    """Test Word Error Rate computation."""

    def test_identical_sentences(self):
        from mlx_tune.ocr import compute_wer
        assert compute_wer("hello world", "hello world") == 0.0

    def test_completely_different(self):
        from mlx_tune.ocr import compute_wer
        wer = compute_wer("foo bar", "hello world")
        assert wer == 1.0  # 2 subs / 2 words

    def test_empty_prediction(self):
        from mlx_tune.ocr import compute_wer
        wer = compute_wer("", "hello world")
        assert wer == 1.0  # 2 deletions / 2 words

    def test_both_empty(self):
        from mlx_tune.ocr import compute_wer
        assert compute_wer("", "") == 0.0

    def test_extra_word(self):
        from mlx_tune.ocr import compute_wer
        wer = compute_wer("hello world foo", "hello world")
        assert abs(wer - 0.5) < 1e-9  # 1 insertion / 2 words

    def test_single_word_match(self):
        from mlx_tune.ocr import compute_wer
        assert compute_wer("hello", "hello") == 0.0


class TestComputeExactMatch:
    """Test exact match computation."""

    def test_exact_match(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("hello", "hello") == 1.0

    def test_no_match(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("hello", "world") == 0.0

    def test_normalize_whitespace(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("  hello  ", "hello") == 1.0

    def test_normalize_case(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("HELLO", "hello") == 1.0

    def test_no_normalize(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("HELLO", "hello", normalize=False) == 0.0

    def test_empty_match(self):
        from mlx_tune.ocr import compute_exact_match
        assert compute_exact_match("", "") == 1.0


class TestComputeOCRMetrics:
    """Test batch OCR metrics computation."""

    def test_perfect_batch(self):
        from mlx_tune.ocr import compute_ocr_metrics
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        metrics = compute_ocr_metrics(preds, refs)
        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0
        assert metrics["exact_match"] == 1.0

    def test_mixed_batch(self):
        from mlx_tune.ocr import compute_ocr_metrics
        preds = ["hello", "wrld"]
        refs = ["hello", "world"]
        metrics = compute_ocr_metrics(preds, refs)
        assert metrics["cer"] > 0.0
        assert metrics["exact_match"] == 0.5  # 1 out of 2

    def test_empty_batch(self):
        from mlx_tune.ocr import compute_ocr_metrics
        metrics = compute_ocr_metrics([], [])
        assert metrics["cer"] == 0.0
        assert metrics["wer"] == 0.0
        assert metrics["exact_match"] == 0.0

    def test_length_mismatch_raises(self):
        from mlx_tune.ocr import compute_ocr_metrics
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_ocr_metrics(["a"], ["a", "b"])

    def test_returns_all_keys(self):
        from mlx_tune.ocr import compute_ocr_metrics
        metrics = compute_ocr_metrics(["a"], ["a"])
        assert set(metrics.keys()) == {"cer", "wer", "exact_match"}


# ============================================================================
# Test Reward Functions
# ============================================================================


class TestCERReward:
    """Test CER-based reward function."""

    def test_perfect_transcription(self):
        from mlx_tune.ocr import cer_reward
        assert cer_reward("hello", "hello") == 1.0

    def test_complete_mismatch(self):
        from mlx_tune.ocr import cer_reward
        reward = cer_reward("abc", "xyz")
        assert reward == 0.0

    def test_partial_match(self):
        from mlx_tune.ocr import cer_reward
        reward = cer_reward("hallo", "hello")
        assert 0.0 < reward < 1.0
        assert abs(reward - 0.8) < 1e-9  # 1 - 0.2

    def test_clamps_to_zero(self):
        from mlx_tune.ocr import cer_reward
        # CER > 1.0 (many insertions vs short reference)
        reward = cer_reward("a very long prediction", "x")
        assert reward == 0.0


class TestExactMatchReward:
    """Test exact match reward function."""

    def test_match(self):
        from mlx_tune.ocr import exact_match_reward
        assert exact_match_reward("hello", "hello") == 1.0

    def test_no_match(self):
        from mlx_tune.ocr import exact_match_reward
        assert exact_match_reward("hello", "world") == 0.0

    def test_normalized_match(self):
        from mlx_tune.ocr import exact_match_reward
        assert exact_match_reward("  HELLO  ", "hello") == 1.0


class TestCombinedOCRReward:
    """Test combined OCR reward function."""

    def test_perfect(self):
        from mlx_tune.ocr import combined_ocr_reward
        assert combined_ocr_reward("hello", "hello") == 1.0

    def test_zero(self):
        from mlx_tune.ocr import combined_ocr_reward
        reward = combined_ocr_reward("abc", "xyz")
        # CER reward = 0.0, EM reward = 0.0
        assert reward == 0.0

    def test_partial(self):
        from mlx_tune.ocr import combined_ocr_reward
        reward = combined_ocr_reward("hallo", "hello")
        # CER reward = 0.8, EM reward = 0.0
        expected = 0.7 * 0.8 + 0.3 * 0.0
        assert abs(reward - expected) < 1e-9

    def test_custom_weights(self):
        from mlx_tune.ocr import combined_ocr_reward
        reward = combined_ocr_reward("hello", "hello", cer_weight=0.5, em_weight=0.5)
        assert reward == 1.0


# ============================================================================
# Test Dataset Helpers
# ============================================================================


class TestConvertOCRPairs:
    """Test OCR pair to VLM message format conversion."""

    def test_basic_conversion(self):
        from mlx_tune.ocr import convert_ocr_pairs_to_messages
        result = convert_ocr_pairs_to_messages("test.png", "hello world")
        assert "messages" in result
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_user_content_has_image_and_text(self):
        from mlx_tune.ocr import convert_ocr_pairs_to_messages
        result = convert_ocr_pairs_to_messages("test.png", "hello")
        content = result["messages"][0]["content"]
        types = [c["type"] for c in content]
        assert "text" in types
        assert "image" in types

    def test_default_instruction(self):
        from mlx_tune.ocr import convert_ocr_pairs_to_messages
        result = convert_ocr_pairs_to_messages("test.png", "hello")
        text_content = [c for c in result["messages"][0]["content"] if c["type"] == "text"][0]
        assert "Transcribe" in text_content["text"]

    def test_custom_instruction(self):
        from mlx_tune.ocr import convert_ocr_pairs_to_messages
        result = convert_ocr_pairs_to_messages("test.png", "hello", instruction="OCR this.")
        text_content = [c for c in result["messages"][0]["content"] if c["type"] == "text"][0]
        assert text_content["text"] == "OCR this."

    def test_assistant_has_ground_truth(self):
        from mlx_tune.ocr import convert_ocr_pairs_to_messages
        result = convert_ocr_pairs_to_messages("test.png", "expected text")
        assistant_content = result["messages"][1]["content"]
        assert assistant_content[0]["text"] == "expected text"


# ============================================================================
# Test OCR Models Registry
# ============================================================================


class TestOCRModelsRegistry:
    """Test the OCR models registry dict."""

    def test_registry_not_empty(self):
        from mlx_tune.ocr import OCR_MODELS
        assert len(OCR_MODELS) > 0

    def test_has_dedicated_models(self):
        from mlx_tune.ocr import OCR_MODELS
        dedicated = [k for k, v in OCR_MODELS.items() if v["type"] == "dedicated"]
        assert len(dedicated) >= 3

    def test_has_general_vlm_models(self):
        from mlx_tune.ocr import OCR_MODELS
        general = [k for k, v in OCR_MODELS.items() if v["type"] == "general_vlm"]
        assert len(general) >= 1

    def test_model_entry_has_required_fields(self):
        from mlx_tune.ocr import OCR_MODELS
        for name, info in OCR_MODELS.items():
            assert "type" in info, f"{name} missing 'type'"
            assert "params" in info, f"{name} missing 'params'"
            assert "description" in info, f"{name} missing 'description'"


# ============================================================================
# Test FastOCRModel API
# ============================================================================


class TestFastOCRModelAPI:
    """Test that FastOCRModel has the correct API surface."""

    def test_has_from_pretrained(self):
        from mlx_tune.ocr import FastOCRModel
        assert hasattr(FastOCRModel, "from_pretrained")
        assert callable(FastOCRModel.from_pretrained)

    def test_has_get_peft_model(self):
        from mlx_tune.ocr import FastOCRModel
        assert hasattr(FastOCRModel, "get_peft_model")
        assert callable(FastOCRModel.get_peft_model)

    def test_has_for_training(self):
        from mlx_tune.ocr import FastOCRModel
        assert hasattr(FastOCRModel, "for_training")
        assert callable(FastOCRModel.for_training)

    def test_has_for_inference(self):
        from mlx_tune.ocr import FastOCRModel
        assert hasattr(FastOCRModel, "for_inference")
        assert callable(FastOCRModel.for_inference)

    def test_get_peft_model_defaults_vision_frozen(self):
        """Verify that get_peft_model defaults finetune_vision_layers=False."""
        import inspect
        from mlx_tune.ocr import FastOCRModel
        sig = inspect.signature(FastOCRModel.get_peft_model)
        params = sig.parameters
        assert params["finetune_vision_layers"].default is False
        assert params["finetune_language_layers"].default is True


# ============================================================================
# Test OCRModelWrapper
# ============================================================================


class TestOCRModelWrapper:
    """Test OCRModelWrapper functionality."""

    def _make_wrapper(self):
        from mlx_tune.ocr import OCRModelWrapper
        mock_vlm = MagicMock()
        mock_vlm.generate.return_value = "transcribed text"
        mock_vlm.model = MagicMock()
        mock_vlm.processor = MagicMock()
        mock_vlm.config = MagicMock()
        mock_vlm._lora_applied = True
        mock_vlm.lora_config = {}
        return OCRModelWrapper(mock_vlm, model_name="mlx-community/DeepSeek-OCR-8bit")

    def test_transcribe_calls_generate(self):
        wrapper = self._make_wrapper()
        result = wrapper.transcribe("test.png")
        wrapper._vlm.generate.assert_called_once()
        assert result == "transcribed text"

    def test_transcribe_default_prompt(self):
        wrapper = self._make_wrapper()
        wrapper.transcribe("test.png")
        call_kwargs = wrapper._vlm.generate.call_args
        assert "OCR" in call_kwargs.kwargs.get("prompt", call_kwargs[1].get("prompt", ""))

    def test_batch_transcribe(self):
        wrapper = self._make_wrapper()
        results = wrapper.batch_transcribe(["a.png", "b.png"], verbose=False)
        assert len(results) == 2
        assert wrapper._vlm.generate.call_count == 2

    def test_evaluate_returns_metrics(self):
        wrapper = self._make_wrapper()
        wrapper._vlm.generate.return_value = "hello"
        metrics = wrapper.evaluate(
            images=["a.png", "b.png"],
            references=["hello", "world"],
            verbose=False,
        )
        assert "cer" in metrics
        assert "wer" in metrics
        assert "exact_match" in metrics

    def test_delegates_save_pretrained(self):
        wrapper = self._make_wrapper()
        wrapper.save_pretrained("output")
        wrapper._vlm.save_pretrained.assert_called_once_with("output")

    def test_delegates_load_adapter(self):
        wrapper = self._make_wrapper()
        wrapper.load_adapter("adapter_path")
        wrapper._vlm.load_adapter.assert_called_once_with("adapter_path")

    def test_property_delegation(self):
        wrapper = self._make_wrapper()
        _ = wrapper.model
        _ = wrapper.processor
        _ = wrapper.config


# ============================================================================
# Test OCRSFTConfig
# ============================================================================


class TestOCRSFTConfig:
    """Test OCR SFT training configuration."""

    def test_default_values(self):
        from mlx_tune.ocr import OCRSFTConfig
        config = OCRSFTConfig()
        assert config.learning_rate == 5e-5
        assert config.max_length == 4096
        assert config.train_on_completions is True
        assert config.output_dir == "ocr_outputs"
        assert config.per_device_train_batch_size == 1

    def test_custom_values(self):
        from mlx_tune.ocr import OCRSFTConfig
        config = OCRSFTConfig(learning_rate=1e-4, max_steps=100)
        assert config.learning_rate == 1e-4
        assert config.max_steps == 100

    def test_eval_params(self):
        from mlx_tune.ocr import OCRSFTConfig
        config = OCRSFTConfig(eval_dataset=["data"], eval_steps=10)
        assert config.eval_dataset == ["data"]
        assert config.eval_steps == 10

    def test_inherits_vlm_config_params(self):
        from mlx_tune.ocr import OCRSFTConfig
        config = OCRSFTConfig(warmup_steps=10, seed=42)
        assert config.warmup_steps == 10
        assert config.seed == 42


# ============================================================================
# Test OCRGRPOConfig
# ============================================================================


class TestOCRGRPOConfig:
    """Test OCR GRPO training configuration."""

    def test_default_values(self):
        from mlx_tune.ocr import OCRGRPOConfig
        config = OCRGRPOConfig()
        assert config.beta == 0.04
        assert config.num_generations == 2
        assert config.max_completion_length == 512
        assert config.output_dir == "./ocr_grpo_outputs"

    def test_custom_values(self):
        from mlx_tune.ocr import OCRGRPOConfig
        config = OCRGRPOConfig(num_generations=4, max_steps=50)
        assert config.num_generations == 4
        assert config.max_steps == 50

    def test_inherits_vlm_grpo_params(self):
        from mlx_tune.ocr import OCRGRPOConfig
        config = OCRGRPOConfig(temperature=0.5, learning_rate=2e-6)
        assert config.temperature == 0.5
        assert config.learning_rate == 2e-6


# ============================================================================
# Test OCR Trainers (mock-based)
# ============================================================================


class TestOCRSFTTrainer:
    """Test OCRSFTTrainer wraps VLMSFTTrainer correctly."""

    @patch("mlx_tune.vlm.VLMSFTTrainer")
    @patch("mlx_tune.vlm.UnslothVisionDataCollator")
    def test_creates_vlm_trainer(self, mock_collator_cls, mock_trainer_cls):
        from mlx_tune.ocr import OCRSFTTrainer, OCRSFTConfig, OCRModelWrapper

        mock_vlm = MagicMock()
        mock_ocr = OCRModelWrapper(mock_vlm, "test-model")

        trainer = OCRSFTTrainer(
            model=mock_ocr,
            processor=MagicMock(),
            train_dataset=[],
            args=OCRSFTConfig(max_steps=10),
        )
        mock_trainer_cls.assert_called_once()

    @patch("mlx_tune.vlm.VLMSFTTrainer")
    @patch("mlx_tune.vlm.UnslothVisionDataCollator")
    def test_train_calls_inner_trainer(self, mock_collator_cls, mock_trainer_cls):
        from mlx_tune.ocr import OCRSFTTrainer, OCRSFTConfig, OCRModelWrapper

        mock_vlm = MagicMock()
        mock_ocr = OCRModelWrapper(mock_vlm, "test-model")
        mock_inner = MagicMock()
        mock_inner.train.return_value = MagicMock(metrics={"loss": 0.5})
        mock_trainer_cls.return_value = mock_inner

        trainer = OCRSFTTrainer(
            model=mock_ocr,
            processor=MagicMock(),
            train_dataset=[],
            args=OCRSFTConfig(max_steps=10),
        )
        stats = trainer.train()
        mock_inner.train.assert_called_once()


class TestOCRGRPOTrainer:
    """Test OCRGRPOTrainer wraps VLMGRPOTrainer correctly."""

    @patch("mlx_tune.vlm.VLMGRPOTrainer")
    def test_creates_vlm_grpo_trainer(self, mock_trainer_cls):
        from mlx_tune.ocr import OCRGRPOTrainer, OCRGRPOConfig, OCRModelWrapper

        mock_vlm = MagicMock()
        mock_ocr = OCRModelWrapper(mock_vlm, "test-model")

        trainer = OCRGRPOTrainer(
            model=mock_ocr,
            train_dataset=[],
            processor=MagicMock(),
            args=OCRGRPOConfig(max_steps=5),
        )
        mock_trainer_cls.assert_called_once()

    @patch("mlx_tune.vlm.VLMGRPOTrainer")
    def test_default_reward_fn_is_combined(self, mock_trainer_cls):
        from mlx_tune.ocr import OCRGRPOTrainer, OCRGRPOConfig, OCRModelWrapper, combined_ocr_reward

        mock_vlm = MagicMock()
        mock_ocr = OCRModelWrapper(mock_vlm, "test-model")

        trainer = OCRGRPOTrainer(
            model=mock_ocr,
            train_dataset=[],
            args=OCRGRPOConfig(),
        )
        # The reward_fn passed should be combined_ocr_reward
        call_kwargs = mock_trainer_cls.call_args
        assert call_kwargs.kwargs.get("reward_fn") == combined_ocr_reward or \
               call_kwargs[1].get("reward_fn") == combined_ocr_reward


# ============================================================================
# Test Imports
# ============================================================================


class TestOCRImports:
    """Test that all OCR exports are importable from mlx_tune."""

    def test_import_fast_ocr_model(self):
        from mlx_tune import FastOCRModel
        assert FastOCRModel is not None

    def test_import_ocr_model_wrapper(self):
        from mlx_tune import OCRModelWrapper
        assert OCRModelWrapper is not None

    def test_import_ocr_sft_trainer(self):
        from mlx_tune import OCRSFTTrainer
        assert OCRSFTTrainer is not None

    def test_import_ocr_sft_config(self):
        from mlx_tune import OCRSFTConfig
        assert OCRSFTConfig is not None

    def test_import_ocr_grpo_trainer(self):
        from mlx_tune import OCRGRPOTrainer
        assert OCRGRPOTrainer is not None

    def test_import_ocr_grpo_config(self):
        from mlx_tune import OCRGRPOConfig
        assert OCRGRPOConfig is not None

    def test_import_metrics(self):
        from mlx_tune import compute_cer, compute_wer, compute_exact_match, compute_ocr_metrics
        assert all(callable(f) for f in [compute_cer, compute_wer, compute_exact_match, compute_ocr_metrics])

    def test_import_dataset_helpers(self):
        from mlx_tune import load_ocr_dataset, convert_ocr_pairs_to_messages
        assert all(callable(f) for f in [load_ocr_dataset, convert_ocr_pairs_to_messages])

    def test_import_reward_functions(self):
        from mlx_tune import cer_reward, exact_match_reward, combined_ocr_reward
        assert all(callable(f) for f in [cer_reward, exact_match_reward, combined_ocr_reward])

    def test_import_registry(self):
        from mlx_tune import OCR_MODELS
        assert isinstance(OCR_MODELS, dict)
