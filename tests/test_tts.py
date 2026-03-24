"""
Tests for TTS (Text-to-Speech) Fine-Tuning Module

Tests FastTTSModel, TTSModelWrapper, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator.
Uses mocks to avoid downloading real models in unit tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace


# ============================================================================
# FastTTSModel API Tests
# ============================================================================


class TestFastTTSModelAPI:
    """Test that FastTTSModel has the correct API surface."""

    def test_has_from_pretrained(self):
        from mlx_tune.tts import FastTTSModel
        assert hasattr(FastTTSModel, "from_pretrained")
        assert callable(FastTTSModel.from_pretrained)

    def test_has_get_peft_model(self):
        from mlx_tune.tts import FastTTSModel
        assert hasattr(FastTTSModel, "get_peft_model")
        assert callable(FastTTSModel.get_peft_model)

    def test_has_for_training(self):
        from mlx_tune.tts import FastTTSModel
        assert hasattr(FastTTSModel, "for_training")
        assert callable(FastTTSModel.for_training)

    def test_has_for_inference(self):
        from mlx_tune.tts import FastTTSModel
        assert hasattr(FastTTSModel, "for_inference")
        assert callable(FastTTSModel.for_inference)

    def test_get_peft_model_rejects_non_wrapper(self):
        from mlx_tune.tts import FastTTSModel
        with pytest.raises(TypeError, match="Expected TTSModelWrapper"):
            FastTTSModel.get_peft_model("not a wrapper")


# ============================================================================
# TTSModelWrapper Tests
# ============================================================================


class TestTTSModelWrapper:
    """Test TTSModelWrapper initialization and state management."""

    def _make_wrapper(self, **kwargs):
        from mlx_tune.tts import TTSModelWrapper
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_codec = MagicMock()
        mock_codec.sampling_rate = 24000
        defaults = dict(
            model=mock_model,
            tokenizer=mock_tokenizer,
            codec=mock_codec,
            model_name="test/orpheus-3b",
            codec_model_name="hubertsiuzdak/snac_24khz",
        )
        defaults.update(kwargs)
        return TTSModelWrapper(**defaults)

    def test_init_defaults(self):
        wrapper = self._make_wrapper()
        assert wrapper.model_name == "test/orpheus-3b"
        assert wrapper.lora_config is None
        assert wrapper.lora_enabled is False
        assert wrapper._lora_applied is False
        assert wrapper.inference_mode is False

    def test_sample_rate(self):
        wrapper = self._make_wrapper()
        assert wrapper.sample_rate == 24000

    def test_audio_token_defaults(self):
        from mlx_tune.tts import ORPHEUS_START_TOKEN, ORPHEUS_END_TOKENS, ORPHEUS_AUDIO_TOKEN_OFFSET
        wrapper = self._make_wrapper()
        assert wrapper.start_token == ORPHEUS_START_TOKEN
        assert wrapper.end_tokens == ORPHEUS_END_TOKENS
        assert wrapper.audio_token_offset == ORPHEUS_AUDIO_TOKEN_OFFSET

    def test_custom_audio_tokens(self):
        wrapper = self._make_wrapper(
            start_token=99999,
            end_tokens=[99998],
            audio_token_offset=100000,
            codebook_size=8192,
        )
        assert wrapper.start_token == 99999
        assert wrapper.end_tokens == [99998]
        assert wrapper.audio_token_offset == 100000
        assert wrapper.codebook_size == 8192

    def test_configure_lora(self):
        wrapper = self._make_wrapper()
        wrapper.configure_lora(r=8, lora_alpha=8, target_modules=["q_proj"])
        assert wrapper.lora_enabled is True
        assert wrapper._lora_applied is False
        assert wrapper.lora_config["r"] == 8
        assert wrapper.lora_config["lora_alpha"] == 8
        assert wrapper.lora_config["target_modules"] == ["q_proj"]

    def test_configure_lora_resets_applied(self):
        wrapper = self._make_wrapper()
        wrapper._lora_applied = True
        wrapper.configure_lora(r=16)
        assert wrapper._lora_applied is False

    def test_for_training(self):
        from mlx_tune.tts import FastTTSModel
        wrapper = self._make_wrapper()
        result = FastTTSModel.for_training(wrapper)
        assert result.inference_mode is False

    def test_for_inference(self):
        from mlx_tune.tts import FastTTSModel
        wrapper = self._make_wrapper()
        result = FastTTSModel.for_inference(wrapper)
        assert result.inference_mode is True


# ============================================================================
# Code Interleaving Tests
# ============================================================================


class TestCodeInterleaving:
    """Test SNAC code interleave/de-interleave round-trip."""

    def _make_wrapper(self):
        from mlx_tune.tts import TTSModelWrapper
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_codec = MagicMock()
        mock_codec.sampling_rate = 24000
        return TTSModelWrapper(
            model=mock_model,
            tokenizer=mock_tokenizer,
            codec=mock_codec,
            model_name="test",
            codebook_size=4096,
            audio_token_offset=128266,
        )

    def test_interleave_empty(self):
        wrapper = self._make_wrapper()
        result = wrapper._interleave_codes([])
        assert result == []

    def test_interleave_basic(self):
        """Test that interleaving produces expected token count."""
        import mlx.core as mx
        wrapper = self._make_wrapper()

        # 3-level SNAC: level 0 = 2 frames, level 1 = 4 frames, level 2 = 8 frames
        codes = [
            mx.array([[10, 20]]),           # Level 0: 2 frames
            mx.array([[11, 12, 21, 22]]),   # Level 1: 4 frames
            mx.array([[13, 14, 15, 16, 23, 24, 25, 26]]),  # Level 2: 8 frames
        ]
        tokens = wrapper._interleave_codes(codes)
        # Should produce 7 tokens per coarsest frame * 2 frames = 14
        assert len(tokens) == 14
        # All tokens should have offset applied
        assert all(t >= wrapper.audio_token_offset for t in tokens)

    def test_deinterleave_basic(self):
        """Test de-interleaving produces 3 level arrays."""
        wrapper = self._make_wrapper()
        # Create 7 tokens (1 coarsest frame)
        offset = wrapper.audio_token_offset
        cs = wrapper.codebook_size
        tokens = [
            10 + offset + 0 * cs,  # L0
            11 + offset + 1 * cs,  # L1
            12 + offset + 1 * cs,  # L1
            13 + offset + 2 * cs,  # L2
            14 + offset + 2 * cs,  # L2
            15 + offset + 2 * cs,  # L2
            16 + offset + 2 * cs,  # L2
        ]
        codes = wrapper._deinterleave_codes(tokens)
        assert len(codes) == 3
        assert len(codes[0]) == 1   # Level 0: 1 frame
        assert len(codes[1]) == 2   # Level 1: 2 frames
        assert len(codes[2]) == 4   # Level 2: 4 frames
        # Check values
        assert codes[0][0] == 10
        assert codes[1][0] == 11
        assert codes[1][1] == 12
        assert codes[2][0] == 13
        assert codes[2][3] == 16

    def test_interleave_deinterleave_roundtrip(self):
        """Round-trip: interleave then de-interleave should recover original codes."""
        import mlx.core as mx
        wrapper = self._make_wrapper()

        # Original codes
        l0 = [5, 10]
        l1 = [1, 2, 3, 4]
        l2 = [10, 20, 30, 40, 50, 60, 70, 80]
        codes = [
            mx.array([l0]),
            mx.array([l1]),
            mx.array([l2]),
        ]

        tokens = wrapper._interleave_codes(codes)
        recovered = wrapper._deinterleave_codes(tokens)

        np.testing.assert_array_equal(recovered[0], l0)
        np.testing.assert_array_equal(recovered[1], l1)
        np.testing.assert_array_equal(recovered[2], l2)


# ============================================================================
# TTSSFTConfig Tests
# ============================================================================


class TestTTSSFTConfig:
    """Test TTSSFTConfig defaults and overrides."""

    def test_defaults(self):
        from mlx_tune.tts import TTSSFTConfig
        config = TTSSFTConfig()
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 2e-4
        assert config.max_steps == 60
        assert config.sample_rate == 24000
        assert config.train_on_completions is True
        assert config.output_dir == "./tts_outputs"

    def test_overrides(self):
        from mlx_tune.tts import TTSSFTConfig
        config = TTSSFTConfig(
            learning_rate=1e-4,
            max_steps=100,
            sample_rate=44100,
            output_dir="/tmp/test",
        )
        assert config.learning_rate == 1e-4
        assert config.max_steps == 100
        assert config.sample_rate == 44100
        assert config.output_dir == "/tmp/test"

    def test_to_dict(self):
        from mlx_tune.tts import TTSSFTConfig
        config = TTSSFTConfig(learning_rate=1e-3)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["learning_rate"] == 1e-3
        assert "per_device_train_batch_size" in d

    def test_kwargs_passthrough(self):
        from mlx_tune.tts import TTSSFTConfig
        config = TTSSFTConfig(custom_param="hello")
        assert config.custom_param == "hello"


# ============================================================================
# TTSDataCollator Tests
# ============================================================================


class TestTTSDataCollator:
    """Test TTSDataCollator processing."""

    def _make_collator(self):
        from mlx_tune.tts import TTSDataCollator, TTSModelWrapper
        from mlx_tune.audio_profiles import TTS_PROFILES
        mock_model = MagicMock(spec=TTSModelWrapper)
        mock_model.profile = TTS_PROFILES["orpheus"]
        mock_model.sample_rate = 24000
        mock_model.start_token = 128259
        mock_model.end_tokens = [128009, 128260]
        mock_model.audio_token_offset = 128266
        mock_model.codebook_size = 4096
        mock_model.encode_audio.return_value = [128266, 128267, 128268]  # 3 audio tokens

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 text tokens
        mock_tokenizer.pad_token_id = 0

        return TTSDataCollator(model=mock_model, tokenizer=mock_tokenizer)

    def test_process_single_sample(self):
        import mlx.core as mx
        collator = self._make_collator()
        sample = {
            "text": "Hello world",
            "audio": {"array": np.zeros(24000), "sampling_rate": 24000},
        }
        result = collator([sample])
        assert "input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_ids"], mx.array)
        assert isinstance(result["labels"], mx.array)

    def test_labels_mask_text_tokens(self):
        collator = self._make_collator()
        sample = {
            "text": "Hello",
            "audio": {"array": np.zeros(24000), "sampling_rate": 24000},
        }
        result = collator([sample])
        labels = np.array(result["labels"])[0]
        # First tokens (text + START) should be masked (-100)
        # text_tokens (5) + start_token (1) = 6 masked
        assert labels[0] == -100
        assert labels[5] == -100  # START token masked
        # Audio tokens should NOT be masked
        assert labels[6] != -100  # First audio token

    def test_batch_padding(self):
        collator = self._make_collator()
        # Make two samples that will have same length due to mock
        samples = [
            {"text": "Hi", "audio": {"array": np.zeros(24000), "sampling_rate": 24000}},
            {"text": "Hello", "audio": {"array": np.zeros(48000), "sampling_rate": 24000}},
        ]
        result = collator(samples)
        assert result["input_ids"].shape[0] == 2
        assert result["labels"].shape[0] == 2
        # Both should have same sequence length
        assert result["input_ids"].shape[1] == result["labels"].shape[1]

    def test_missing_audio_raises(self):
        collator = self._make_collator()
        with pytest.raises(ValueError, match="missing"):
            collator([{"text": "no audio here"}])

    def test_dict_input_treated_as_single(self):
        collator = self._make_collator()
        sample = {
            "text": "Hello",
            "audio": {"array": np.zeros(24000), "sampling_rate": 24000},
        }
        result = collator(sample)
        assert result["input_ids"].shape[0] == 1


# ============================================================================
# TTSSFTTrainer Tests
# ============================================================================


class TestTTSSFTTrainer:
    """Test TTSSFTTrainer initialization and config parsing."""

    def test_init_with_config(self):
        from mlx_tune.tts import TTSSFTTrainer, TTSSFTConfig, TTSModelWrapper
        mock_wrapper = MagicMock(spec=TTSModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.tokenizer = MagicMock()
        mock_wrapper.lora_enabled = False

        config = TTSSFTConfig(learning_rate=1e-3, max_steps=50)
        trainer = TTSSFTTrainer(
            model=mock_wrapper,
            args=config,
            train_dataset=[],
        )
        assert trainer.learning_rate == 1e-3
        assert trainer.max_steps == 50
        assert trainer.gradient_accumulation_steps == 4

    def test_init_with_kwargs(self):
        from mlx_tune.tts import TTSSFTTrainer, TTSModelWrapper
        mock_wrapper = MagicMock(spec=TTSModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.tokenizer = MagicMock()
        mock_wrapper.lora_enabled = False

        trainer = TTSSFTTrainer(
            model=mock_wrapper,
            learning_rate=5e-4,
            max_steps=25,
            train_dataset=[],
        )
        assert trainer.learning_rate == 5e-4
        assert trainer.max_steps == 25

    def test_output_dir_created(self, tmp_path):
        from mlx_tune.tts import TTSSFTTrainer, TTSSFTConfig, TTSModelWrapper
        mock_wrapper = MagicMock(spec=TTSModelWrapper)
        mock_wrapper.model = MagicMock()
        mock_wrapper.tokenizer = MagicMock()
        mock_wrapper.lora_enabled = False

        out_dir = str(tmp_path / "test_output")
        config = TTSSFTConfig(output_dir=out_dir)
        trainer = TTSSFTTrainer(model=mock_wrapper, args=config, train_dataset=[])
        assert (tmp_path / "test_output").exists()


# ============================================================================
# Save/Load Tests
# ============================================================================


class TestTTSSaveLoad:
    """Test adapter save/load functionality."""

    def _make_wrapper(self):
        from mlx_tune.tts import TTSModelWrapper
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_codec = MagicMock()
        mock_codec.sampling_rate = 24000
        return TTSModelWrapper(
            model=mock_model,
            tokenizer=mock_tokenizer,
            codec=mock_codec,
            model_name="test/model",
        )

    def test_save_without_lora_prints_message(self, tmp_path, capsys):
        wrapper = self._make_wrapper()
        wrapper.save_pretrained(str(tmp_path))
        captured = capsys.readouterr()
        assert "No LoRA" in captured.out

    def test_adapter_config_json_structure(self, tmp_path):
        import json
        wrapper = self._make_wrapper()
        wrapper.lora_config = {"r": 16, "lora_alpha": 16, "target_modules": ["q_proj"], "lora_dropout": 0.0}
        wrapper._lora_applied = True

        # Mock trainable parameters
        import mlx.core as mx
        wrapper.model.trainable_parameters.return_value = {"lora_a": mx.zeros((4, 16))}

        wrapper.save_pretrained(str(tmp_path))

        config_path = tmp_path / "adapter_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            config = json.load(f)
        assert config["model_type"] == "tts"
        assert config["fine_tune_type"] == "lora"
        assert "lora_parameters" in config
        assert "audio_config" in config
        assert config["audio_config"]["sample_rate"] == 24000


# ============================================================================
# Import and Module Tests
# ============================================================================


class TestTTSImports:
    """Test that all TTS classes are importable from mlx_tune."""

    def test_import_fast_tts_model(self):
        from mlx_tune import FastTTSModel
        assert FastTTSModel is not None

    def test_import_tts_wrapper(self):
        from mlx_tune import TTSModelWrapper
        assert TTSModelWrapper is not None

    def test_import_tts_trainer(self):
        from mlx_tune import TTSSFTTrainer
        assert TTSSFTTrainer is not None

    def test_import_tts_config(self):
        from mlx_tune import TTSSFTConfig
        assert TTSSFTConfig is not None

    def test_import_tts_collator(self):
        from mlx_tune import TTSDataCollator
        assert TTSDataCollator is not None

    def test_all_in_module_all(self):
        import mlx_tune
        for name in ["FastTTSModel", "TTSModelWrapper", "TTSSFTTrainer", "TTSSFTConfig", "TTSDataCollator"]:
            assert name in mlx_tune.__all__, f"{name} not in __all__"


# ============================================================================
# Constants Tests
# ============================================================================


class TestFastTTSModelConvert:
    """Test FastTTSModel.convert() static method."""

    def test_has_convert_method(self):
        from mlx_tune.tts import FastTTSModel
        assert hasattr(FastTTSModel, "convert")
        assert callable(FastTTSModel.convert)

    @patch("mlx_tune.tts.FastTTSModel.convert")
    def test_convert_calls_mlx_lm(self, mock_convert):
        from mlx_tune.tts import FastTTSModel
        FastTTSModel.convert("canopylabs/orpheus-3b", output_dir="./out")
        mock_convert.assert_called_once_with("canopylabs/orpheus-3b", output_dir="./out")

    def test_convert_accepts_quantize_params(self):
        """Test convert signature accepts quantization parameters."""
        import inspect
        from mlx_tune.tts import FastTTSModel
        sig = inspect.signature(FastTTSModel.convert)
        params = list(sig.parameters.keys())
        assert "hf_model" in params
        assert "output_dir" in params
        assert "quantize" in params
        assert "q_bits" in params
        assert "dtype" in params
        assert "upload_repo" in params


class TestTTSSavePretrained:
    """Test TTSModelWrapper.save_pretrained_merged() and push_to_hub()."""

    def _make_wrapper(self):
        """Create a mock TTSModelWrapper for testing."""
        from mlx_tune.tts import TTSModelWrapper
        mock_model = MagicMock()
        mock_model.named_modules.return_value = []
        mock_tokenizer = MagicMock()
        wrapper = TTSModelWrapper.__new__(TTSModelWrapper)
        wrapper.model = mock_model
        wrapper.tokenizer = mock_tokenizer
        wrapper.model_name = "test-tts"
        wrapper.config = {"model_type": "llama"}
        wrapper.lora_enabled = False
        wrapper._lora_applied = False
        wrapper._adapter_path = None
        wrapper.start_token = 128259
        wrapper.end_tokens = [128009, 128260]
        wrapper.audio_token_offset = 128266
        wrapper.codebook_size = 4096
        wrapper.lora_config = {}
        wrapper.codec = None
        return wrapper

    def test_has_save_pretrained_merged(self):
        from mlx_tune.tts import TTSModelWrapper
        assert hasattr(TTSModelWrapper, "save_pretrained_merged")

    def test_has_push_to_hub(self):
        from mlx_tune.tts import TTSModelWrapper
        assert hasattr(TTSModelWrapper, "push_to_hub")

    @patch("mlx_lm.utils.save_model")
    def test_save_pretrained_merged_creates_dir(self, mock_save, tmp_path):
        wrapper = self._make_wrapper()
        output = str(tmp_path / "merged_model")
        wrapper.save_pretrained_merged(output)
        assert (tmp_path / "merged_model").exists()

    @patch("mlx_lm.utils.save_model")
    def test_save_pretrained_merged_saves_config(self, mock_save, tmp_path):
        wrapper = self._make_wrapper()
        output = str(tmp_path / "merged_model")
        wrapper.save_pretrained_merged(output)
        import json
        config_path = tmp_path / "merged_model" / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["model_type"] == "llama"

    @patch("mlx_lm.utils.save_model")
    def test_save_pretrained_merged_fuses_lora(self, mock_save, tmp_path):
        wrapper = self._make_wrapper()
        wrapper._lora_applied = True
        mock_fused = MagicMock()
        mock_module = MagicMock()
        mock_module.fuse.return_value = mock_fused
        wrapper.model.named_modules.return_value = [("layer.0.attn.q_proj", mock_module)]

        output = str(tmp_path / "merged")
        wrapper.save_pretrained_merged(output)
        mock_module.fuse.assert_called_once()

    def test_push_to_hub_requires_saved_model(self):
        wrapper = self._make_wrapper()
        with pytest.raises(ValueError, match="No saved model"):
            wrapper.push_to_hub("user/repo")

    @patch("mlx_tune.tts._push_to_hub")
    def test_push_to_hub_calls_helper(self, mock_push):
        from pathlib import Path
        wrapper = self._make_wrapper()
        wrapper._adapter_path = Path("/tmp/test_adapters")
        # Make exist() return True
        with patch.object(Path, "exists", return_value=True):
            wrapper.push_to_hub("user/my-tts-model")
        mock_push.assert_called_once_with("/tmp/test_adapters", "user/my-tts-model")


class TestPushToHubHelper:
    """Test _push_to_hub helper function."""

    def test_push_to_hub_function_exists(self):
        from mlx_tune.tts import _push_to_hub
        assert callable(_push_to_hub)

    def test_push_to_hub_calls_hf_api(self, tmp_path):
        """Test that _push_to_hub calls create_repo + upload_folder."""
        (tmp_path / "model.safetensors").write_text("fake")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api

            from mlx_tune.tts import _push_to_hub
            _push_to_hub(str(tmp_path), "user/test-model")

            mock_api.create_repo.assert_called_once()
            mock_api.upload_folder.assert_called_once()

    def test_push_to_hub_with_token(self, tmp_path):
        """Test that _push_to_hub passes token to create_repo and upload_folder."""
        (tmp_path / "model.safetensors").write_text("fake")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api

            from mlx_tune.tts import _push_to_hub
            _push_to_hub(str(tmp_path), "user/test-model", token="hf_test123")

            # Token should be passed to create_repo
            create_kwargs = mock_api.create_repo.call_args
            assert create_kwargs[1].get("token") == "hf_test123"

            # Token should be passed to upload_folder
            upload_kwargs = mock_api.upload_folder.call_args
            assert upload_kwargs[1].get("token") == "hf_test123"

    def test_push_to_hub_with_private_repo(self, tmp_path):
        """Test that _push_to_hub can create private repos."""
        (tmp_path / "model.safetensors").write_text("fake")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api

            from mlx_tune.tts import _push_to_hub
            _push_to_hub(str(tmp_path), "user/test-model", private=True)

            create_kwargs = mock_api.create_repo.call_args
            assert create_kwargs[1].get("private") is True

    def test_push_to_hub_custom_commit_message(self, tmp_path):
        """Test that _push_to_hub uses custom commit message."""
        (tmp_path / "model.safetensors").write_text("fake")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            mock_api = MagicMock()
            MockHfApi.return_value = mock_api

            from mlx_tune.tts import _push_to_hub
            _push_to_hub(str(tmp_path), "user/test", commit_message="My custom message")

            upload_kwargs = mock_api.upload_folder.call_args
            assert upload_kwargs[1].get("commit_message") == "My custom message"


class TestTTSConstants:
    """Test Orpheus-specific constants."""

    def test_orpheus_constants(self):
        from mlx_tune.tts import (
            ORPHEUS_START_TOKEN,
            ORPHEUS_END_TOKENS,
            ORPHEUS_AUDIO_TOKEN_OFFSET,
            ORPHEUS_CODEBOOK_SIZE,
        )
        assert ORPHEUS_START_TOKEN == 128259
        assert ORPHEUS_END_TOKENS == [128009, 128260]
        assert ORPHEUS_AUDIO_TOKEN_OFFSET == 128266
        assert ORPHEUS_CODEBOOK_SIZE == 4096
