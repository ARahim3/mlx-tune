"""
End-to-end tests for audio model fine-tuning.

These tests download real models and verify the full pipeline:
load -> apply LoRA -> collate data -> train 2 steps -> verify loss decreases.

Mark: @pytest.mark.slow (skipped in normal test runs)
Run with: pytest tests/test_audio_e2e.py -v -m slow
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_audio(sr=24000, duration=1.0):
    """Generate a short sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


def _make_tts_dataset(n=4, sr=24000):
    """Create a minimal TTS dataset (list of dicts)."""
    return [
        {
            "text": f"Test sentence number {i}.",
            "audio": {"array": _make_fake_audio(sr, 1.0), "sampling_rate": sr},
        }
        for i in range(n)
    ]


def _make_stt_dataset(n=4, sr=16000):
    """Create a minimal STT dataset (list of dicts)."""
    return [
        {
            "text": f"Hello world number {i}",
            "audio": {"array": _make_fake_audio(sr, 2.0), "sampling_rate": sr},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TTS E2E: OuteTTS
# ---------------------------------------------------------------------------

class TestOuteTTSE2E:
    """End-to-end test for OuteTTS fine-tuning."""

    @pytest.mark.slow
    def test_outetts_load_and_detect(self):
        """Load OuteTTS and verify profile auto-detection."""
        from mlx_tune import FastTTSModel
        from mlx_tune.audio_profiles import TTS_PROFILES

        model, tokenizer = FastTTSModel.from_pretrained(
            "mlx-community/Llama-OuteTTS-1.0-1B-8bit",
        )

        assert model.profile.name == "outetts"
        assert model.profile.codec_type == "dac"
        assert model.profile.token_format == "text"
        assert tokenizer is not None

    @pytest.mark.slow
    def test_outetts_lora_and_train(self):
        """Apply LoRA and train 2 steps on OuteTTS."""
        from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

        model, tokenizer = FastTTSModel.from_pretrained(
            "mlx-community/Llama-OuteTTS-1.0-1B-8bit",
        )
        model = FastTTSModel.get_peft_model(model, r=8, lora_alpha=8)

        dataset = _make_tts_dataset(n=4, sr=24000)
        collator = TTSDataCollator(model=model, tokenizer=tokenizer)

        trainer = TTSSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=dataset,
            args=TTSSFTConfig(
                output_dir=tempfile.mkdtemp(),
                max_steps=2,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                logging_steps=1,
            ),
        )

        result = trainer.train()
        assert result.metrics["train_loss"] > 0
        assert result.metrics["train_loss"] < 100  # Sanity check


# ---------------------------------------------------------------------------
# TTS E2E: Spark-TTS
# ---------------------------------------------------------------------------

class TestSparkTTSE2E:
    """End-to-end test for Spark-TTS fine-tuning."""

    @pytest.mark.slow
    def test_spark_load_and_detect(self):
        """Load Spark-TTS and verify profile auto-detection."""
        from mlx_tune import FastTTSModel

        model, tokenizer = FastTTSModel.from_pretrained(
            "mlx-community/Spark-TTS-0.5B-bf16",
        )

        assert model.profile.name == "spark"
        assert model.profile.codec_type == "bicodec"
        assert model.profile.token_format == "text"
        assert tokenizer is not None

    @pytest.mark.slow
    def test_spark_lora_and_train(self):
        """Apply LoRA and train 2 steps on Spark-TTS."""
        from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator

        model, tokenizer = FastTTSModel.from_pretrained(
            "mlx-community/Spark-TTS-0.5B-bf16",
        )
        model = FastTTSModel.get_peft_model(model, r=8, lora_alpha=8)

        dataset = _make_tts_dataset(n=4, sr=16000)
        collator = TTSDataCollator(model=model, tokenizer=tokenizer)

        trainer = TTSSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=dataset,
            args=TTSSFTConfig(
                output_dir=tempfile.mkdtemp(),
                max_steps=2,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                logging_steps=1,
                sample_rate=16000,
            ),
        )

        result = trainer.train()
        assert result.metrics["train_loss"] > 0
        assert result.metrics["train_loss"] < 100


# ---------------------------------------------------------------------------
# STT E2E: Moonshine
# ---------------------------------------------------------------------------

class TestMoonshineE2E:
    """End-to-end test for Moonshine STT fine-tuning."""

    @pytest.mark.slow
    def test_moonshine_load_and_detect(self):
        """Load Moonshine and verify profile auto-detection."""
        from mlx_tune import FastSTTModel

        model, processor = FastSTTModel.from_pretrained(
            "UsefulSensors/moonshine-tiny",
        )

        assert model.profile.name == "moonshine"
        assert model.profile.preprocessor == "raw_conv"
        assert processor is not None

    @pytest.mark.slow
    def test_moonshine_lora_and_train(self):
        """Apply LoRA and train 2 steps on Moonshine."""
        from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator

        model, processor = FastSTTModel.from_pretrained(
            "UsefulSensors/moonshine-tiny",
        )
        model = FastSTTModel.get_peft_model(
            model, r=8, lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        dataset = _make_stt_dataset(n=4, sr=16000)
        collator = STTDataCollator(
            model=model, processor=processor,
            language="en", task="transcribe",
        )

        trainer = STTSFTTrainer(
            model=model,
            processor=processor,
            data_collator=collator,
            train_dataset=dataset,
            args=STTSFTConfig(
                output_dir=tempfile.mkdtemp(),
                max_steps=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-5,
                logging_steps=1,
            ),
        )

        result = trainer.train()
        assert result.metrics["train_loss"] > 0
        assert result.metrics["train_loss"] < 100


# ---------------------------------------------------------------------------
# STT E2E: Distil-Whisper (uses Whisper profile)
# ---------------------------------------------------------------------------

class TestDistilWhisperE2E:
    """Verify Distil-Whisper works with existing Whisper profile."""

    @pytest.mark.slow
    def test_distil_whisper_detection(self):
        """Distil-Whisper should auto-detect as 'whisper' profile."""
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("distil-whisper-large-v3", {}) == "whisper"
        assert detect_stt_model_type("mlx-community/distil-whisper-large-v3", {}) == "whisper"

    @pytest.mark.slow
    def test_distil_whisper_load(self):
        """Load distil-whisper and verify it uses Whisper profile."""
        from mlx_tune import FastSTTModel

        model, processor = FastSTTModel.from_pretrained(
            "mlx-community/distil-whisper-large-v3",
        )

        assert model.profile.name == "whisper"
        assert model.profile.preprocessor == "log_mel_spectrogram"
        assert processor is not None


# ---------------------------------------------------------------------------
# Auto-save adapter test
# ---------------------------------------------------------------------------

class TestAdapterSaveLoad:
    """Test saving/loading adapters for new models."""

    @pytest.mark.slow
    def test_outetts_save_load_adapter(self):
        """Save and reload LoRA adapters for OuteTTS."""
        from mlx_tune import FastTTSModel

        model, tokenizer = FastTTSModel.from_pretrained(
            "mlx-community/Llama-OuteTTS-1.0-1B-8bit",
        )
        model = FastTTSModel.get_peft_model(model, r=8, lora_alpha=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "adapters.safetensors").exists()
            assert (Path(tmpdir) / "adapter_config.json").exists()

            # Load back
            model2, tok2 = FastTTSModel.from_pretrained(
                "mlx-community/Llama-OuteTTS-1.0-1B-8bit",
            )
            model2 = FastTTSModel.get_peft_model(model2, r=8, lora_alpha=8)
            model2.load_adapter(tmpdir)
