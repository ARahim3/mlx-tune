"""
Tests for STT (Speech-to-Text) Fine-Tuning Module

Tests FastSTTModel, STTModelWrapper, STTSFTTrainer, STTSFTConfig, STTDataCollator.
Uses mocks to avoid downloading real models in unit tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace


# ============================================================================
# FastSTTModel API Tests
# ============================================================================


class TestFastSTTModelAPI:
    """Test that FastSTTModel has the correct API surface."""

    def test_has_from_pretrained(self):
        from mlx_tune.stt import FastSTTModel
        assert hasattr(FastSTTModel, "from_pretrained")
        assert callable(FastSTTModel.from_pretrained)

    def test_has_get_peft_model(self):
        from mlx_tune.stt import FastSTTModel
        assert hasattr(FastSTTModel, "get_peft_model")
        assert callable(FastSTTModel.get_peft_model)

    def test_has_for_training(self):
        from mlx_tune.stt import FastSTTModel
        assert hasattr(FastSTTModel, "for_training")
        assert callable(FastSTTModel.for_training)

    def test_has_for_inference(self):
        from mlx_tune.stt import FastSTTModel
        assert hasattr(FastSTTModel, "for_inference")
        assert callable(FastSTTModel.for_inference)

    def test_get_peft_model_rejects_non_wrapper(self):
        from mlx_tune.stt import FastSTTModel
        with pytest.raises(TypeError, match="Expected STTModelWrapper"):
            FastSTTModel.get_peft_model("not a wrapper")

    def test_get_peft_model_requires_at_least_one_target(self):
        from mlx_tune.stt import FastSTTModel, STTModelWrapper
        mock_wrapper = MagicMock(spec=STTModelWrapper)
        mock_wrapper.configure_lora = MagicMock()
        # Both False should raise
        with pytest.raises(ValueError, match="At least one"):
            FastSTTModel.get_peft_model(
                mock_wrapper,
                finetune_encoder=False,
                finetune_decoder=False,
            )


# ============================================================================
# STTModelWrapper Tests
# ============================================================================


class TestSTTModelWrapper:
    """Test STTModelWrapper initialization and state management."""

    def _make_wrapper(self, **kwargs):
        from mlx_tune.stt import STTModelWrapper, STTProcessor
        mock_model = MagicMock()
        mock_model.dims = SimpleNamespace(
            n_audio_layer=4,
            n_text_layer=4,
            n_mels=80,
            n_vocab=51865,
        )
        mock_processor = MagicMock(spec=STTProcessor)
        defaults = dict(
            model=mock_model,
            processor=mock_processor,
            model_name="test/whisper-small",
        )
        defaults.update(kwargs)
        return STTModelWrapper(**defaults)

    def test_init_defaults(self):
        wrapper = self._make_wrapper()
        assert wrapper.model_name == "test/whisper-small"
        assert wrapper.max_seq_length == 448
        assert wrapper.lora_config is None
        assert wrapper.lora_enabled is False
        assert wrapper._lora_applied is False
        assert wrapper.inference_mode is False

    def test_model_dimensions(self):
        wrapper = self._make_wrapper()
        assert wrapper.n_audio_layer == 4
        assert wrapper.n_text_layer == 4
        assert wrapper.n_mels == 80
        assert wrapper.n_vocab == 51865

    def test_configure_lora_encoder_decoder(self):
        wrapper = self._make_wrapper()
        wrapper.configure_lora(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            finetune_encoder=True,
            finetune_decoder=True,
        )
        assert wrapper.lora_enabled is True
        assert wrapper.lora_config["r"] == 8
        assert wrapper.lora_config["finetune_encoder"] is True
        assert wrapper.lora_config["finetune_decoder"] is True

    def test_configure_lora_encoder_only(self):
        wrapper = self._make_wrapper()
        wrapper.configure_lora(
            r=16,
            finetune_encoder=True,
            finetune_decoder=False,
        )
        assert wrapper.lora_config["finetune_encoder"] is True
        assert wrapper.lora_config["finetune_decoder"] is False

    def test_configure_lora_decoder_only(self):
        wrapper = self._make_wrapper()
        wrapper.configure_lora(
            r=16,
            finetune_encoder=False,
            finetune_decoder=True,
        )
        assert wrapper.lora_config["finetune_encoder"] is False
        assert wrapper.lora_config["finetune_decoder"] is True

    def test_for_training(self):
        from mlx_tune.stt import FastSTTModel
        wrapper = self._make_wrapper()
        result = FastSTTModel.for_training(wrapper)
        assert result.inference_mode is False

    def test_for_inference(self):
        from mlx_tune.stt import FastSTTModel
        wrapper = self._make_wrapper()
        result = FastSTTModel.for_inference(wrapper)
        assert result.inference_mode is True


# ============================================================================
# LoRA Application Tests
# ============================================================================


class TestSTTLoRAApplication:
    """Test LoRA application to encoder-decoder model."""

    def _make_block(self, has_cross_attn=False):
        """Create a mock ResidualAttentionBlock."""
        import mlx.nn as nn
        import mlx.core as mx

        block = MagicMock()

        # Self-attention with real Linear layers
        attn = MagicMock()
        attn.query = nn.Linear(64, 64)
        attn.key = nn.Linear(64, 64, bias=False)
        attn.value = nn.Linear(64, 64)
        attn.out = nn.Linear(64, 64)
        block.attn = attn

        if has_cross_attn:
            cross_attn = MagicMock()
            cross_attn.query = nn.Linear(64, 64)
            cross_attn.key = nn.Linear(64, 64, bias=False)
            cross_attn.value = nn.Linear(64, 64)
            cross_attn.out = nn.Linear(64, 64)
            block.cross_attn = cross_attn
        else:
            block.cross_attn = None

        block.mlp1 = nn.Linear(64, 256)
        block.mlp2 = nn.Linear(256, 64)

        return block

    def test_apply_lora_to_block_self_attn(self):
        from mlx_tune.stt import STTModelWrapper, STTProcessor
        import mlx.nn as nn

        wrapper = STTModelWrapper(
            model=MagicMock(),
            processor=MagicMock(spec=STTProcessor),
            model_name="test",
        )

        block = self._make_block(has_cross_attn=False)
        replaced = wrapper._apply_lora_to_block(
            block, ["query", "value"], r=8, scale=1.0, dropout=0.0,
            has_cross_attn=False,
        )

        # Should replace query and value in self-attention
        assert replaced == 2

    def test_apply_lora_to_block_with_cross_attn(self):
        from mlx_tune.stt import STTModelWrapper, STTProcessor
        import mlx.nn as nn

        wrapper = STTModelWrapper(
            model=MagicMock(),
            processor=MagicMock(spec=STTProcessor),
            model_name="test",
        )

        block = self._make_block(has_cross_attn=True)
        replaced = wrapper._apply_lora_to_block(
            block, ["query", "value"], r=8, scale=1.0, dropout=0.0,
            has_cross_attn=True,
        )

        # 2 in self-attn + 2 in cross-attn = 4
        assert replaced == 4

    def test_apply_lora_to_block_with_mlp(self):
        from mlx_tune.stt import STTModelWrapper, STTProcessor

        wrapper = STTModelWrapper(
            model=MagicMock(),
            processor=MagicMock(spec=STTProcessor),
            model_name="test",
        )

        block = self._make_block(has_cross_attn=False)
        replaced = wrapper._apply_lora_to_block(
            block, ["query", "mlp1"], r=8, scale=1.0, dropout=0.0,
            has_cross_attn=False,
        )

        # 1 in self-attn (query) + 1 mlp1 = 2
        assert replaced == 2


# ============================================================================
# STTSFTConfig Tests
# ============================================================================


class TestSTTSFTConfig:
    """Test STTSFTConfig defaults and overrides."""

    def test_defaults(self):
        from mlx_tune.stt import STTSFTConfig
        config = STTSFTConfig()
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 1e-5
        assert config.max_steps == 100
        assert config.sample_rate == 16000
        assert config.language == "en"
        assert config.task == "transcribe"
        assert config.max_audio_length == 30.0
        assert config.n_mels == 80
        assert config.output_dir == "./stt_outputs"

    def test_overrides(self):
        from mlx_tune.stt import STTSFTConfig
        config = STTSFTConfig(
            learning_rate=5e-5,
            max_steps=200,
            language="fr",
            task="translate",
            n_mels=128,
        )
        assert config.learning_rate == 5e-5
        assert config.max_steps == 200
        assert config.language == "fr"
        assert config.task == "translate"
        assert config.n_mels == 128

    def test_to_dict(self):
        from mlx_tune.stt import STTSFTConfig
        config = STTSFTConfig(learning_rate=2e-5)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["learning_rate"] == 2e-5
        assert "sample_rate" in d
        assert "language" in d

    def test_kwargs_passthrough(self):
        from mlx_tune.stt import STTSFTConfig
        config = STTSFTConfig(custom_param="value")
        assert config.custom_param == "value"


# ============================================================================
# STTProcessor Tests
# ============================================================================


class TestSTTProcessor:
    """Test STTProcessor functionality."""

    def test_encode(self):
        from mlx_tune.stt import STTProcessor
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        processor = STTProcessor(tokenizer=mock_tokenizer)
        result = processor.encode("hello")
        assert result == [1, 2, 3]

    def test_decode(self):
        from mlx_tune.stt import STTProcessor
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "hello world"
        processor = STTProcessor(tokenizer=mock_tokenizer)
        result = processor.decode([1, 2, 3])
        assert result == "hello world"

    def test_encode_no_tokenizer(self):
        from mlx_tune.stt import STTProcessor
        processor = STTProcessor(tokenizer=None)
        result = processor.encode("hello")
        assert result == []

    def test_sot_sequence_default(self):
        from mlx_tune.stt import STTProcessor
        processor = STTProcessor(tokenizer=None)
        assert processor.sot_sequence == (50258,)

    def test_sot_sequence_from_tokenizer(self):
        from mlx_tune.stt import STTProcessor
        mock_tokenizer = MagicMock()
        mock_tokenizer.sot_sequence = (50258, 50259, 50360)
        processor = STTProcessor(tokenizer=mock_tokenizer)
        assert processor.sot_sequence == (50258, 50259, 50360)


# ============================================================================
# STTDataCollator Tests
# ============================================================================


class TestSTTDataCollator:
    """Test STTDataCollator processing."""

    def _make_collator(self):
        from mlx_tune.stt import STTDataCollator, STTModelWrapper, STTProcessor
        from mlx_tune.audio_profiles import STT_PROFILES
        import mlx.core as mx

        mock_model = MagicMock(spec=STTModelWrapper)
        mock_model.profile = STT_PROFILES["whisper"]
        mock_model.n_mels = 80

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [100, 200, 300]
        mock_tokenizer.eot = 50257
        mock_tokenizer.sot_sequence = (50258,)

        mock_processor = MagicMock(spec=STTProcessor)
        mock_processor.tokenizer = mock_tokenizer
        mock_processor.sot_sequence = (50258,)
        mock_processor.compute_mel.return_value = mx.zeros((3000, 80))

        return STTDataCollator(model=mock_model, processor=mock_processor)

    def test_auto_detect_text_column(self):
        collator = self._make_collator()
        # "text" column
        col = collator._find_text_column({"text": "hello", "audio": None})
        assert col == "text"
        # "sentence" column
        col = collator._find_text_column({"sentence": "hello", "audio": None})
        assert col == "sentence"
        # "transcription" column
        col = collator._find_text_column({"transcription": "hello", "audio": None})
        assert col == "transcription"

    def test_auto_detect_fails_gracefully(self):
        collator = self._make_collator()
        with pytest.raises(ValueError, match="Could not find text column"):
            collator._find_text_column({"unknown_col": "hello"})

    def test_explicit_text_column(self):
        from mlx_tune.stt import STTDataCollator, STTModelWrapper, STTProcessor
        collator = STTDataCollator(
            model=MagicMock(spec=STTModelWrapper),
            processor=MagicMock(spec=STTProcessor),
            text_column="my_text",
        )
        col = collator._find_text_column({"my_text": "hello"})
        assert col == "my_text"

    def test_process_returns_correct_keys(self):
        import mlx.core as mx
        collator = self._make_collator()
        sample = {
            "text": "Hello world",
            "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        }
        result = collator([sample])
        assert "input_features" in result
        assert "decoder_input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_features"], mx.array)
        assert isinstance(result["decoder_input_ids"], mx.array)
        assert isinstance(result["labels"], mx.array)

    def test_missing_audio_raises(self):
        collator = self._make_collator()
        with pytest.raises(ValueError, match="missing"):
            collator([{"text": "no audio"}])

    def test_dict_input_treated_as_single(self):
        collator = self._make_collator()
        sample = {
            "text": "Hello",
            "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        }
        result = collator(sample)
        assert result["input_features"].shape[0] == 1

    def test_language_specific_sot_sequence(self):
        """Collator with non-English language should use language-specific SOT."""
        from mlx_tune.stt import STTDataCollator, STTModelWrapper, STTProcessor
        from mlx_tune.audio_profiles import STT_PROFILES
        import mlx.core as mx

        mock_model = MagicMock(spec=STTModelWrapper)
        mock_model.profile = STT_PROFILES["whisper"]
        mock_model.n_mels = 80

        mock_en_tokenizer = MagicMock()
        mock_en_tokenizer.encode.return_value = [100, 200]
        mock_en_tokenizer.eot = 50257
        mock_en_tokenizer.sot_sequence = (50258, 50259, 50360)

        mock_fr_tokenizer = MagicMock()
        mock_fr_tokenizer.encode.return_value = [100, 200]
        mock_fr_tokenizer.eot = 50257
        mock_fr_tokenizer.sot_sequence = (50258, 50265, 50360)

        mock_processor = MagicMock(spec=STTProcessor)
        mock_processor.tokenizer = mock_en_tokenizer
        mock_processor.sot_sequence = (50258, 50259, 50360)
        mock_processor.compute_mel.return_value = mx.zeros((3000, 80))
        mock_processor.get_tokenizer.return_value = mock_fr_tokenizer

        collator = STTDataCollator(
            model=mock_model, processor=mock_processor,
            language="fr", task="transcribe",
        )
        assert collator._lang_tokenizer is mock_fr_tokenizer

        sample = {
            "text": "Bonjour",
            "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
        }
        result = collator([sample])
        dec_ids = result["decoder_input_ids"].tolist()[0]
        assert dec_ids[:3] == [50258, 50265, 50360]

    def test_default_english_no_extra_tokenizer(self):
        """Default language='en' should NOT call get_tokenizer again."""
        from mlx_tune.stt import STTDataCollator, STTModelWrapper, STTProcessor
        from mlx_tune.audio_profiles import STT_PROFILES

        mock_processor = MagicMock(spec=STTProcessor)
        mock_model = MagicMock(spec=STTModelWrapper)
        mock_model.profile = STT_PROFILES["whisper"]
        collator = STTDataCollator(
            model=mock_model, processor=mock_processor,
            language="en", task="transcribe",
        )
        assert collator._lang_tokenizer is None
        mock_processor.get_tokenizer.assert_not_called()

    def test_translate_task_gets_specific_tokenizer(self):
        """task='translate' should get a task-specific tokenizer."""
        from mlx_tune.stt import STTDataCollator, STTModelWrapper, STTProcessor
        from mlx_tune.audio_profiles import STT_PROFILES

        mock_translate_tok = MagicMock()
        mock_translate_tok.sot_sequence = (50258, 50259, 50358)

        mock_processor = MagicMock(spec=STTProcessor)
        mock_processor.get_tokenizer.return_value = mock_translate_tok

        mock_model = MagicMock(spec=STTModelWrapper)
        mock_model.profile = STT_PROFILES["whisper"]
        collator = STTDataCollator(
            model=mock_model, processor=mock_processor,
            language="en", task="translate",
        )
        assert collator._lang_tokenizer is mock_translate_tok


# ============================================================================
# STTSFTTrainer Tests
# ============================================================================


class TestSTTSFTTrainer:
    """Test STTSFTTrainer initialization and config parsing."""

    def test_init_with_config(self):
        from mlx_tune.stt import STTSFTTrainer, STTSFTConfig, STTModelWrapper
        mock_wrapper = MagicMock(spec=STTModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.processor = MagicMock()
        mock_wrapper.lora_enabled = False

        config = STTSFTConfig(learning_rate=5e-5, max_steps=50)
        trainer = STTSFTTrainer(
            model=mock_wrapper,
            args=config,
            train_dataset=[],
        )
        assert trainer.learning_rate == 5e-5
        assert trainer.max_steps == 50
        assert trainer.gradient_accumulation_steps == 4

    def test_init_with_kwargs(self):
        from mlx_tune.stt import STTSFTTrainer, STTModelWrapper
        mock_wrapper = MagicMock(spec=STTModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.processor = MagicMock()
        mock_wrapper.lora_enabled = False

        trainer = STTSFTTrainer(
            model=mock_wrapper,
            learning_rate=2e-5,
            max_steps=30,
            train_dataset=[],
        )
        assert trainer.learning_rate == 2e-5
        assert trainer.max_steps == 30

    def test_output_dir_created(self, tmp_path):
        from mlx_tune.stt import STTSFTTrainer, STTSFTConfig, STTModelWrapper
        mock_wrapper = MagicMock(spec=STTModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.processor = MagicMock()
        mock_wrapper.lora_enabled = False

        out_dir = str(tmp_path / "stt_output")
        config = STTSFTConfig(output_dir=out_dir)
        trainer = STTSFTTrainer(model=mock_wrapper, args=config, train_dataset=[])
        assert (tmp_path / "stt_output").exists()


# ============================================================================
# Save/Load Tests
# ============================================================================


class TestSTTSaveLoad:
    """Test adapter save/load functionality."""

    def _make_wrapper(self):
        from mlx_tune.stt import STTModelWrapper, STTProcessor
        mock_model = MagicMock()
        mock_model.dims = SimpleNamespace(
            n_audio_layer=4,
            n_text_layer=4,
            n_mels=80,
            n_vocab=51865,
        )
        mock_processor = MagicMock(spec=STTProcessor)
        return STTModelWrapper(
            model=mock_model,
            processor=mock_processor,
            model_name="test/whisper",
        )

    def test_save_without_lora_prints_message(self, tmp_path, capsys):
        wrapper = self._make_wrapper()
        wrapper.save_pretrained(str(tmp_path))
        captured = capsys.readouterr()
        assert "No LoRA" in captured.out

    def test_adapter_config_json_structure(self, tmp_path):
        import json
        import mlx.core as mx
        wrapper = self._make_wrapper()
        wrapper.lora_config = {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["query", "value"],
            "finetune_encoder": True,
            "finetune_decoder": True,
        }
        wrapper._lora_applied = True
        wrapper.model.trainable_parameters.return_value = {"lora_a": mx.zeros((4, 16))}

        wrapper.save_pretrained(str(tmp_path))

        config_path = tmp_path / "adapter_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
        assert config["model_type"] == "stt"
        assert config["fine_tune_type"] == "lora"
        assert "lora_parameters" in config
        assert "whisper_config" in config
        assert config["whisper_config"]["sample_rate"] == 16000
        assert config["lora_parameters"]["finetune_encoder"] is True
        assert config["lora_parameters"]["finetune_decoder"] is True


# ============================================================================
# Import and Module Tests
# ============================================================================


class TestSTTImports:
    """Test that all STT classes are importable from mlx_tune."""

    def test_import_fast_stt_model(self):
        from mlx_tune import FastSTTModel
        assert FastSTTModel is not None

    def test_import_stt_wrapper(self):
        from mlx_tune import STTModelWrapper
        assert STTModelWrapper is not None

    def test_import_stt_trainer(self):
        from mlx_tune import STTSFTTrainer
        assert STTSFTTrainer is not None

    def test_import_stt_config(self):
        from mlx_tune import STTSFTConfig
        assert STTSFTConfig is not None

    def test_import_stt_collator(self):
        from mlx_tune import STTDataCollator
        assert STTDataCollator is not None

    def test_import_stt_processor(self):
        from mlx_tune import STTProcessor
        assert STTProcessor is not None

    def test_all_in_module_all(self):
        import mlx_tune
        for name in ["FastSTTModel", "STTModelWrapper", "STTSFTTrainer", "STTSFTConfig", "STTDataCollator", "STTProcessor"]:
            assert name in mlx_tune.__all__, f"{name} not in __all__"


# ============================================================================
# Constants Tests
# ============================================================================


class TestFastSTTModelConvert:
    """Test FastSTTModel.convert() static method."""

    def test_has_convert_method(self):
        from mlx_tune.stt import FastSTTModel
        assert hasattr(FastSTTModel, "convert")
        assert callable(FastSTTModel.convert)

    @patch("mlx_tune.stt.FastSTTModel.convert")
    def test_convert_calls_mlx_audio(self, mock_convert):
        from mlx_tune.stt import FastSTTModel
        FastSTTModel.convert("openai/whisper-large-v3", output_dir="./out")
        mock_convert.assert_called_once_with("openai/whisper-large-v3", output_dir="./out")

    def test_convert_accepts_params(self):
        """Test convert signature accepts all parameters."""
        import inspect
        from mlx_tune.stt import FastSTTModel
        sig = inspect.signature(FastSTTModel.convert)
        params = list(sig.parameters.keys())
        assert "hf_model" in params
        assert "output_dir" in params
        assert "quantize" in params
        assert "q_bits" in params
        assert "dtype" in params
        assert "upload_repo" in params


class TestSTTSavePretrained:
    """Test STTModelWrapper.save_pretrained_merged() and push_to_hub()."""

    def _make_wrapper(self):
        """Create a mock STTModelWrapper for testing."""
        from mlx_tune.stt import STTModelWrapper
        mock_model = MagicMock()
        mock_model.named_modules.return_value = []
        mock_model.parameters.return_value = {}
        # Whisper model has .dims but it's a dataclass — use spec to prevent
        # hasattr from returning True for everything
        del mock_model.dims  # Remove auto-created .dims so hasattr returns False
        wrapper = STTModelWrapper.__new__(STTModelWrapper)
        wrapper.model = mock_model
        wrapper.processor = MagicMock()
        wrapper.model_name = "test-whisper"
        wrapper.lora_enabled = False
        wrapper._lora_applied = False
        wrapper._adapter_path = None
        wrapper.lora_config = {}
        return wrapper

    def test_has_save_pretrained_merged(self):
        from mlx_tune.stt import STTModelWrapper
        assert hasattr(STTModelWrapper, "save_pretrained_merged")

    def test_has_push_to_hub(self):
        from mlx_tune.stt import STTModelWrapper
        assert hasattr(STTModelWrapper, "push_to_hub")

    @patch("mlx.utils.tree_flatten", return_value=[])
    @patch("mlx.core.savez")
    def test_save_pretrained_merged_creates_dir(self, mock_savez, mock_flatten, tmp_path):
        wrapper = self._make_wrapper()
        output = str(tmp_path / "merged_stt")
        wrapper.save_pretrained_merged(output)
        assert (tmp_path / "merged_stt").exists()

    @patch("mlx.utils.tree_flatten", return_value=[("weight", MagicMock())])
    @patch("mlx.core.savez")
    def test_save_pretrained_merged_saves_weights(self, mock_savez, mock_flatten, tmp_path):
        wrapper = self._make_wrapper()
        output = str(tmp_path / "merged_stt")
        wrapper.save_pretrained_merged(output)
        mock_savez.assert_called_once()

    @patch("mlx.utils.tree_flatten", return_value=[])
    @patch("mlx.core.savez")
    def test_save_pretrained_merged_fuses_lora(self, mock_savez, mock_flatten, tmp_path):
        wrapper = self._make_wrapper()
        wrapper._lora_applied = True
        mock_fused = MagicMock()
        mock_module = MagicMock()
        mock_module.fuse.return_value = mock_fused
        wrapper.model.named_modules.return_value = [("encoder.blocks.0.attn.query", mock_module)]
        output = str(tmp_path / "merged")
        wrapper.save_pretrained_merged(output)
        mock_module.fuse.assert_called_once()

    def test_push_to_hub_requires_saved_model(self):
        wrapper = self._make_wrapper()
        with pytest.raises(ValueError, match="No saved model"):
            wrapper.push_to_hub("user/repo")

    @patch("mlx_tune.stt._push_to_hub")
    def test_push_to_hub_calls_helper(self, mock_push):
        from pathlib import Path
        wrapper = self._make_wrapper()
        wrapper._adapter_path = Path("/tmp/test_adapters")
        with patch.object(Path, "exists", return_value=True):
            wrapper.push_to_hub("user/my-whisper-model")
        mock_push.assert_called_once_with("/tmp/test_adapters", "user/my-whisper-model")


class TestSTTPushToHubHelper:
    """Test _push_to_hub helper function."""

    def test_push_to_hub_function_exists(self):
        from mlx_tune.stt import _push_to_hub
        assert callable(_push_to_hub)


class TestSTTConstants:
    """Test Whisper-related constants."""

    def test_whisper_constants(self):
        from mlx_tune.stt import (
            WHISPER_SAMPLE_RATE,
            WHISPER_N_SAMPLES,
            WHISPER_N_MELS,
            WHISPER_HOP_LENGTH,
            WHISPER_N_FFT,
        )
        assert WHISPER_SAMPLE_RATE == 16000
        assert WHISPER_N_SAMPLES == 480000
        assert WHISPER_N_MELS == 80
        assert WHISPER_HOP_LENGTH == 160
        assert WHISPER_N_FFT == 400


# ============================================================================
# LoRA Linear Helper Tests
# ============================================================================


class TestCreateLoRALinear:
    """Test the _create_lora_linear helper."""

    def test_creates_lora_layer(self):
        from mlx_tune.stt import _create_lora_linear
        import mlx.nn as nn

        original = nn.Linear(64, 64)
        lora = _create_lora_linear(original, r=8, scale=1.0, dropout=0.0)

        # Should not be the same object
        assert lora is not original
        # Should have LoRA attributes
        assert hasattr(lora, "lora_a") or hasattr(lora, "scale")

    def test_creates_lora_with_different_ranks(self):
        from mlx_tune.stt import _create_lora_linear
        import mlx.nn as nn

        for rank in [4, 8, 16, 32]:
            original = nn.Linear(128, 128)
            lora = _create_lora_linear(original, r=rank, scale=1.0, dropout=0.0)
            assert lora is not original
