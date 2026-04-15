"""
Tests for audio model profiles and codec adapters.

These tests verify that:
1. Profile dataclasses are correctly constructed and immutable
2. Auto-detection functions match model names to profiles
3. Codec adapters wrap operations correctly (SNAC, DAC, BiCodec, Mimi)
4. Profile registries contain expected entries
5. Profile field values match the hardcoded constants they replaced
6. Text-token format profiles have correct audio_token_formats
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# TTSModelProfile tests - Orpheus
# ---------------------------------------------------------------------------

class TestOrpheusProfile:
    """Test Orpheus TTSModelProfile."""

    def test_orpheus_profile_exists(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert "orpheus" in TTS_PROFILES

    def test_orpheus_profile_name(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.name == "orpheus"

    def test_orpheus_profile_architecture(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.architecture == "decoder_only"

    def test_orpheus_profile_start_token(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.start_token == 128259

    def test_orpheus_profile_end_tokens(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.end_tokens == (128009, 128260)

    def test_orpheus_profile_audio_token_offset(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.audio_token_offset == 128266

    def test_orpheus_profile_codebook_size(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.codebook_size == 4096

    def test_orpheus_profile_codec_type(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.codec_type == "snac"

    def test_orpheus_profile_codec_repo(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.codec_repo == "mlx-community/snac_24khz"

    def test_orpheus_profile_sample_rate(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.sample_rate == 24000

    def test_orpheus_profile_num_codebooks(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.num_codebooks == 3

    def test_orpheus_profile_interleave_pattern(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.interleave_pattern == (1, 2, 4)
        assert sum(profile.interleave_pattern) == 7  # tokens per frame

    def test_orpheus_profile_prompt_template(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        rendered = profile.prompt_template.format(speaker="tara", text="Hello")
        assert "tara" in rendered
        assert "Hello" in rendered
        assert "<custom_token_3>" in rendered

    def test_orpheus_profile_default_speaker(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.default_speaker == "tara"

    def test_orpheus_profile_lora_target_modules(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert "q_proj" in profile.lora_target_modules
        assert "k_proj" in profile.lora_target_modules
        assert "v_proj" in profile.lora_target_modules
        assert "o_proj" in profile.lora_target_modules
        assert "gate_proj" in profile.lora_target_modules

    def test_orpheus_profile_lora_module_mapping(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.lora_module_mapping["q_proj"] == "self_attn.q_proj"
        assert profile.lora_module_mapping["gate_proj"] == "mlp.gate_proj"

    def test_orpheus_profile_loader(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.loader == "mlx_lm"

    def test_orpheus_numeric_token_format(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        assert profile.token_format == "numeric"
        assert profile.audio_token_formats == ()

    def test_profile_is_frozen(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["orpheus"]
        with pytest.raises(AttributeError):
            profile.name = "modified"

    def test_profile_field_access(self):
        from mlx_tune.audio_profiles import TTSModelProfile
        profile = TTSModelProfile(
            name="test",
            architecture="test_arch",
            codec_type="snac",
            codec_repo="test/repo",
            sample_rate=16000,
            start_token=1,
            end_tokens=(2, 3),
            audio_token_offset=100,
            codebook_size=1024,
            num_codebooks=2,
            interleave_pattern=(1, 2),
            prompt_template="{speaker}: {text}",
            default_speaker="default",
            lora_target_modules=("q_proj",),
            lora_module_mapping={"q_proj": "self_attn.q_proj"},
            loader="test_loader",
        )
        assert profile.name == "test"
        assert profile.sample_rate == 16000


# ---------------------------------------------------------------------------
# TTSModelProfile tests - OuteTTS
# ---------------------------------------------------------------------------

class TestOuteTTSProfile:
    """Test OuteTTS TTSModelProfile."""

    def test_outetts_profile_exists(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert "outetts" in TTS_PROFILES

    def test_outetts_profile_name(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["outetts"].name == "outetts"

    def test_outetts_codec_type(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["outetts"].codec_type == "dac"

    def test_outetts_text_token_format(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["outetts"]
        assert profile.token_format == "text"

    def test_outetts_audio_token_formats(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["outetts"]
        assert len(profile.audio_token_formats) == 2
        assert profile.audio_token_formats[0].format(code=42) == "<|c1_42|>"
        assert profile.audio_token_formats[1].format(code=7) == "<|c2_7|>"

    def test_outetts_num_codebooks(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["outetts"].num_codebooks == 2

    def test_outetts_loader(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["outetts"].loader == "mlx_audio_tts"

    def test_outetts_inner_model_attr(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["outetts"].inner_model_attr == "model"

    def test_outetts_prompt_template(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["outetts"]
        rendered = profile.prompt_template.format(speaker="", text="Hello")
        assert "<|text_start|>" in rendered
        assert "Hello" in rendered
        assert "<|audio_start|>" in rendered


# ---------------------------------------------------------------------------
# TTSModelProfile tests - Spark
# ---------------------------------------------------------------------------

class TestSparkProfile:
    """Test Spark-TTS TTSModelProfile."""

    def test_spark_profile_exists(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert "spark" in TTS_PROFILES

    def test_spark_codec_type(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["spark"].codec_type == "bicodec"

    def test_spark_text_token_format(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["spark"]
        assert profile.token_format == "text"

    def test_spark_audio_token_formats(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["spark"]
        assert len(profile.audio_token_formats) == 2
        assert profile.audio_token_formats[0].format(code=5) == "<|bicodec_global_5|>"
        assert profile.audio_token_formats[1].format(code=10) == "<|bicodec_semantic_10|>"

    def test_spark_sample_rate(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["spark"].sample_rate == 16000

    def test_spark_loader(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["spark"].loader == "mlx_audio_tts"

    def test_spark_prompt_template(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["spark"]
        rendered = profile.prompt_template.format(speaker="", text="Hello")
        assert "<|tts|>" in rendered
        assert "Hello" in rendered


# ---------------------------------------------------------------------------
# TTSModelProfile tests - Sesame
# ---------------------------------------------------------------------------

class TestSesameProfile:
    """Test Sesame/CSM TTSModelProfile."""

    def test_sesame_profile_exists(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert "sesame" in TTS_PROFILES

    def test_sesame_architecture(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["sesame"].architecture == "backbone_decoder"

    def test_sesame_codec_type(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["sesame"].codec_type == "mimi"

    def test_sesame_num_codebooks(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["sesame"].num_codebooks == 32

    def test_sesame_codebook_size(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        assert TTS_PROFILES["sesame"].codebook_size == 2048

    def test_sesame_numeric_token_format(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["sesame"]
        assert profile.token_format == "numeric"
        assert profile.audio_token_formats == ()

    def test_sesame_interleave_pattern(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        profile = TTS_PROFILES["sesame"]
        assert len(profile.interleave_pattern) == 32
        assert all(x == 1 for x in profile.interleave_pattern)


# ---------------------------------------------------------------------------
# STTModelProfile tests - Whisper
# ---------------------------------------------------------------------------

class TestWhisperProfile:
    """Test Whisper STTModelProfile."""

    def test_whisper_profile_exists(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert "whisper" in STT_PROFILES

    def test_whisper_profile_name(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].name == "whisper"

    def test_whisper_profile_sample_rate(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].sample_rate == 16000

    def test_whisper_profile_n_mels(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].n_mels == 80

    def test_whisper_profile_max_audio_samples(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].max_audio_samples == 480000

    def test_whisper_profile_encoder_block_path(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].encoder_block_path == "encoder.blocks"

    def test_whisper_profile_decoder_block_path(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].decoder_block_path == "decoder.blocks"

    def test_whisper_profile_attn_names(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        profile = STT_PROFILES["whisper"]
        assert profile.attn_names["self_attn"] == "attn"
        assert profile.attn_names["cross_attn"] == "cross_attn"

    def test_whisper_profile_cross_attn_attr(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].cross_attn_attr == "cross_attn"

    def test_whisper_profile_sot_token_id(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].sot_token_id == 50258

    def test_whisper_profile_lora_target_modules(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        profile = STT_PROFILES["whisper"]
        assert "query" in profile.lora_target_modules
        assert "key" in profile.lora_target_modules
        assert "value" in profile.lora_target_modules
        assert "out" in profile.lora_target_modules

    def test_whisper_profile_loader(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].loader == "mlx_audio_stt"

    def test_whisper_profile_preprocessor(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["whisper"].preprocessor == "log_mel_spectrogram"

    def test_whisper_profile_is_frozen(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        with pytest.raises(AttributeError):
            STT_PROFILES["whisper"].name = "modified"


# ---------------------------------------------------------------------------
# STTModelProfile tests - Moonshine
# ---------------------------------------------------------------------------

class TestMoonshineProfile:
    """Test Moonshine STTModelProfile."""

    def test_moonshine_profile_exists(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert "moonshine" in STT_PROFILES

    def test_moonshine_profile_name(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].name == "moonshine"

    def test_moonshine_sample_rate(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].sample_rate == 16000

    def test_moonshine_preprocessor(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].preprocessor == "raw_conv"

    def test_moonshine_no_mels(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].n_mels == 0

    def test_moonshine_variable_length(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].max_audio_samples == 0

    def test_moonshine_encoder_block_path(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].encoder_block_path == "encoder.layers"

    def test_moonshine_decoder_block_path(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].decoder_block_path == "decoder.layers"

    def test_moonshine_attn_names(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        profile = STT_PROFILES["moonshine"]
        assert profile.attn_names["self_attn"] == "self_attn"
        assert profile.attn_names["cross_attn"] == "encoder_attn"

    def test_moonshine_cross_attn_attr(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].cross_attn_attr == "encoder_attn"

    def test_moonshine_lora_target_modules(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        profile = STT_PROFILES["moonshine"]
        assert "q_proj" in profile.lora_target_modules
        assert "v_proj" in profile.lora_target_modules

    def test_moonshine_sot_token_id(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert STT_PROFILES["moonshine"].sot_token_id == 1


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------

class TestAutoDetection:
    """Test model type auto-detection from model names."""

    # TTS detection
    def test_detect_orpheus_basic(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("orpheus-3b-0.1", {}) == "orpheus"

    def test_detect_orpheus_with_org(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("mlx-community/orpheus-3b-0.1-ft-bf16", {}) == "orpheus"

    def test_detect_orpheus_canopylabs(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("canopylabs/orpheus-3b-0.1-ft", {}) == "orpheus"

    def test_detect_outetts(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("OuteAI/OuteTTS-0.3-500M", {}) == "outetts"

    def test_detect_outetts_lowercase(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("mlx-community/outetts-0.3-500m-bf16", {}) == "outetts"

    def test_detect_spark_tts(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("SparkAudio/spark-tts-0.5b", {}) == "spark"

    def test_detect_spark_hyphen(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("spark-tts-model", {}) == "spark"

    def test_detect_spark_underscore(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("spark_tts_model", {}) == "spark"

    def test_detect_sesame(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("sesame-csm-1b", {}) == "sesame"

    def test_detect_csm(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("csm-1b-mlx", {}) == "sesame"

    def test_detect_tts_unknown(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("some-random-model", {}) is None

    def test_detect_tts_from_config(self):
        from mlx_tune.audio_profiles import detect_tts_model_type
        assert detect_tts_model_type("unknown-model", {"model_type": "orpheus"}) == "orpheus"

    # STT detection
    def test_detect_whisper_basic(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("whisper-large-v3", {}) == "whisper"

    def test_detect_whisper_with_org(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("mlx-community/whisper-large-v3-turbo", {}) == "whisper"

    def test_detect_whisper_openai(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("openai/whisper-tiny", {}) == "whisper"

    def test_detect_whisper_case_insensitive(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("Whisper-Large-V3", {}) == "whisper"

    def test_detect_distil_whisper(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("distil-whisper-large-v3", {}) == "whisper"

    def test_detect_moonshine(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("useful-sensors/moonshine-base", {}) == "moonshine"

    def test_detect_moonshine_basic(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("moonshine-tiny", {}) == "moonshine"

    def test_detect_qwen3_asr(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("mlx-community/Qwen3-ASR-1.7B-8bit", {}) == "qwen3_asr"

    def test_detect_qwen3_asr_lowercase(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("qwen3-asr-0.6b", {}) == "qwen3_asr"

    def test_detect_canary(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("nvidia/canary-1b-v2", {}) == "canary"

    def test_detect_voxtral(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("mlx-community/Voxtral-Mini-3B-2507-bf16", {}) == "voxtral"

    def test_detect_voxtral_lowercase(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        # Regular Voxtral matches voxtral; Realtime now matches its own profile
        # (since v0.4.21 — was None before that).
        assert detect_stt_model_type("mistral/voxtral-mini", {}) == "voxtral"
        assert (
            detect_stt_model_type("mistral/voxtral-realtime", {}) == "voxtral_realtime"
        )

    def test_detect_stt_unknown(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("some-random-stt-model", {}) is None

    def test_detect_stt_from_config(self):
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert detect_stt_model_type("unknown-model", {"model_type": "whisper"}) == "whisper"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistries:
    """Test profile registries."""

    def test_tts_registry_has_all_models(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        expected = {"orpheus", "outetts", "spark", "sesame", "qwen3_tts"}
        assert set(TTS_PROFILES.keys()) == expected

    def test_stt_registry_has_all_models(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        expected = {
            "whisper",
            "moonshine",
            "qwen3_asr",
            "canary",
            "voxtral",
            "voxtral_realtime",
            "parakeet_tdt",
        }
        assert set(STT_PROFILES.keys()) == expected

    def test_tts_registry_values_are_profiles(self):
        from mlx_tune.audio_profiles import TTS_PROFILES, TTSModelProfile
        for profile in TTS_PROFILES.values():
            assert isinstance(profile, TTSModelProfile)

    def test_stt_registry_values_are_profiles(self):
        from mlx_tune.audio_profiles import STT_PROFILES, STTModelProfile
        for profile in STT_PROFILES.values():
            assert isinstance(profile, STTModelProfile)

    def test_all_tts_profiles_have_required_fields(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        for name, profile in TTS_PROFILES.items():
            assert profile.name == name
            assert profile.codec_type in ("snac", "dac", "bicodec", "mimi", "qwen3_speech")
            assert profile.sample_rate > 0
            assert profile.num_codebooks >= 1
            assert len(profile.lora_target_modules) > 0

    def test_all_stt_profiles_have_required_fields(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        for name, profile in STT_PROFILES.items():
            assert profile.name == name
            assert profile.sample_rate > 0
            assert profile.preprocessor in (
                "log_mel_spectrogram",
                "raw_conv",
                "canary_mel",
                "parakeet_mel",
            )
            assert len(profile.lora_target_modules) > 0
            assert profile.architecture in (
                "encoder_decoder",
                "audio_llm",
                "voxtral_realtime",
                "parakeet_tdt",
            )

    def test_audio_llm_profiles_have_audio_token_id(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        for name, profile in STT_PROFILES.items():
            if profile.architecture == "audio_llm":
                assert profile.audio_token_id is not None, f"{name} missing audio_token_id"

    def test_qwen3_asr_profile(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["qwen3_asr"]
        assert p.architecture == "audio_llm"
        assert p.n_mels == 128
        assert p.audio_token_id == 151676
        assert "q_proj" in p.lora_target_modules
        assert p.encoder_lora_targets is not None
        assert "out_proj" in p.encoder_lora_targets

    def test_canary_profile(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["canary"]
        assert p.architecture == "encoder_decoder"
        assert p.n_mels == 128
        assert "conformer" in p.encoder_block_path
        assert p.encoder_lora_targets is not None
        assert "linear_q" in p.encoder_lora_targets

    def test_voxtral_profile(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["voxtral"]
        assert p.architecture == "audio_llm"
        assert p.audio_token_id == 24
        assert "language_model" in p.decoder_block_path


class TestVoxtralRealtimeProfile:
    """Tests for the Voxtral Realtime STT profile (third architecture type)."""

    def test_voxtral_realtime_profile_exists(self):
        from mlx_tune.audio_profiles import STT_PROFILES
        assert "voxtral_realtime" in STT_PROFILES
        p = STT_PROFILES["voxtral_realtime"]
        assert p.name == "voxtral_realtime"
        assert p.sample_rate == 16000
        assert p.n_mels == 128
        assert p.decoder_block_path == "decoder.layers"
        assert p.encoder_block_path == "encoder.transformer_layers"

    def test_voxtral_realtime_architecture_field(self):
        """Ensure the third architecture type is registered correctly."""
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["voxtral_realtime"]
        assert p.architecture == "voxtral_realtime"
        # Confirm it's distinct from the existing two
        assert p.architecture != "encoder_decoder"
        assert p.architecture != "audio_llm"

    def test_voxtral_realtime_lora_targets_use_w_prefix(self):
        """Regression: Voxtral Realtime uses Mistral-internal naming
        (wq/wk/wv/wo) NOT q_proj/k_proj/v_proj/o_proj. Future devs may try to
        'fix' this — keep the test to prevent that."""
        from mlx_tune.audio_profiles import STT_PROFILES
        p = STT_PROFILES["voxtral_realtime"]
        assert p.lora_target_modules == ("wq", "wk", "wv", "wo")
        assert p.encoder_lora_targets == ("wq", "wk", "wv", "wo")
        assert p.decoder_ffn_targets == (
            "feed_forward_w1",
            "feed_forward_w2",
            "feed_forward_w3",
        )
        # Sanity: should NOT contain q_proj or gate_proj
        assert "q_proj" not in p.lora_target_modules
        assert "gate_proj" not in (p.decoder_ffn_targets or ())

    def test_voxtral_realtime_detection_positive(self):
        """Realtime variants should detect as voxtral_realtime, not voxtral."""
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert (
            detect_stt_model_type("mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16")
            == "voxtral_realtime"
        )
        assert (
            detect_stt_model_type("mistralai/Voxtral-Mini-4B-Realtime-2602")
            == "voxtral_realtime"
        )
        assert (
            detect_stt_model_type("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
            == "voxtral_realtime"
        )

    def test_voxtral_realtime_detection_negative(self):
        """Regression: regular Voxtral must NOT match voxtral_realtime."""
        from mlx_tune.audio_profiles import detect_stt_model_type
        assert (
            detect_stt_model_type("mlx-community/Voxtral-Mini-3B-2507-bf16")
            == "voxtral"
        )
        assert (
            detect_stt_model_type("mistralai/Voxtral-Mini-3B-2507")
            == "voxtral"
        )
        assert (
            detect_stt_model_type("mlx-community/Voxtral-Small-24B-2507")
            == "voxtral"
        )


# ---------------------------------------------------------------------------
# Codec Adapter tests - SNAC
# ---------------------------------------------------------------------------

class TestSNACCodecAdapter:
    """Test SNACCodecAdapter creation and interface."""

    def test_create_codec_snac(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import create_codec

        mock_snac = MagicMock()
        mock_snac.sampling_rate = 24000

        adapter = create_codec(TTS_PROFILES["orpheus"], mock_snac)
        assert adapter.sample_rate == 24000
        assert adapter.num_codebooks == 3

    def test_create_codec_unsupported(self):
        from mlx_tune.audio_profiles import TTSModelProfile
        from mlx_tune.audio_codecs import create_codec

        fake_profile = TTSModelProfile(
            name="fake",
            architecture="fake",
            codec_type="unsupported_codec",
            codec_repo="fake/repo",
            sample_rate=16000,
        )

        with pytest.raises(ValueError, match="Unsupported codec type"):
            create_codec(fake_profile, MagicMock())

    def test_snac_adapter_interleave_empty(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import SNACCodecAdapter

        mock_snac = MagicMock()
        adapter = SNACCodecAdapter(TTS_PROFILES["orpheus"], mock_snac)

        result = adapter.interleave([])
        assert result == []

    def test_snac_adapter_deinterleave_roundtrip(self):
        """Verify interleave -> deinterleave is identity (modulo offset)."""
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import SNACCodecAdapter

        mock_snac = MagicMock()
        adapter = SNACCodecAdapter(TTS_PROFILES["orpheus"], mock_snac)

        # Create fake hierarchical codes: 2 coarsest frames
        level0 = np.array([10, 20])
        level1 = np.array([30, 31, 40, 41])
        level2 = np.array([50, 51, 52, 53, 60, 61, 62, 63])

        tokens = adapter.interleave([level0, level1, level2])
        assert len(tokens) == 14  # 2 frames * 7 tokens/frame

        recovered = adapter.deinterleave(tokens)
        assert len(recovered) == 3
        np.testing.assert_array_equal(recovered[0], level0)
        np.testing.assert_array_equal(recovered[1], level1)
        np.testing.assert_array_equal(recovered[2], level2)

    def test_snac_adapter_isinstance_protocol(self):
        from mlx_tune.audio_codecs import SNACCodecAdapter, CodecAdapter
        from mlx_tune.audio_profiles import TTS_PROFILES

        mock_snac = MagicMock()
        adapter = SNACCodecAdapter(TTS_PROFILES["orpheus"], mock_snac)
        assert isinstance(adapter, CodecAdapter)


# ---------------------------------------------------------------------------
# Codec Adapter tests - DAC
# ---------------------------------------------------------------------------

class TestDACCodecAdapter:
    """Test DACCodecAdapter for OuteTTS."""

    def test_create_dac_adapter(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import create_codec

        mock_dac = MagicMock()
        adapter = create_codec(TTS_PROFILES["outetts"], mock_dac)
        assert adapter.num_codebooks == 2

    def test_dac_interleave_flat(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import DACCodecAdapter

        mock_dac = MagicMock()
        adapter = DACCodecAdapter(TTS_PROFILES["outetts"], mock_dac)

        # 2 codebooks, 3 time steps each
        c1 = np.array([10, 20, 30])
        c2 = np.array([40, 50, 60])

        tokens = adapter.interleave([c1, c2])
        # Flat interleave: [c1_t0, c2_t0, c1_t1, c2_t1, c1_t2, c2_t2]
        assert tokens == [10, 40, 20, 50, 30, 60]

    def test_dac_deinterleave_roundtrip(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import DACCodecAdapter

        mock_dac = MagicMock()
        adapter = DACCodecAdapter(TTS_PROFILES["outetts"], mock_dac)

        c1 = np.array([10, 20, 30])
        c2 = np.array([40, 50, 60])

        tokens = adapter.interleave([c1, c2])
        recovered = adapter.deinterleave(tokens)
        assert len(recovered) == 2
        np.testing.assert_array_equal(recovered[0], c1)
        np.testing.assert_array_equal(recovered[1], c2)

    def test_dac_interleave_empty(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import DACCodecAdapter

        mock_dac = MagicMock()
        adapter = DACCodecAdapter(TTS_PROFILES["outetts"], mock_dac)
        assert adapter.interleave([]) == []


# ---------------------------------------------------------------------------
# Codec Adapter tests - BiCodec
# ---------------------------------------------------------------------------

class TestBiCodecAdapter:
    """Test BiCodecAdapter for Spark-TTS."""

    def test_create_bicodec_adapter(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import create_codec

        mock_bicodec = MagicMock()
        adapter = create_codec(TTS_PROFILES["spark"], mock_bicodec)
        assert adapter.num_codebooks == 2

    def test_bicodec_interleave_concatenates(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import BiCodecAdapter

        mock_bicodec = MagicMock()
        adapter = BiCodecAdapter(TTS_PROFILES["spark"], mock_bicodec)

        global_tokens = np.array([1, 2, 3])
        semantic_tokens = np.array([10, 20, 30])

        tokens = adapter.interleave([global_tokens, semantic_tokens])
        # Concatenated: global first, then semantic
        assert tokens == [1, 2, 3, 10, 20, 30]

    def test_bicodec_deinterleave_splits_at_midpoint(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import BiCodecAdapter

        mock_bicodec = MagicMock()
        adapter = BiCodecAdapter(TTS_PROFILES["spark"], mock_bicodec)

        tokens = [1, 2, 3, 10, 20, 30]
        recovered = adapter.deinterleave(tokens)
        assert len(recovered) == 2
        np.testing.assert_array_equal(recovered[0], [1, 2, 3])
        np.testing.assert_array_equal(recovered[1], [10, 20, 30])


# ---------------------------------------------------------------------------
# Codec Adapter tests - Mimi
# ---------------------------------------------------------------------------

class TestMimiCodecAdapter:
    """Test MimiCodecAdapter for Sesame/CSM."""

    def test_create_mimi_adapter(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import create_codec

        mock_mimi = MagicMock()
        adapter = create_codec(TTS_PROFILES["sesame"], mock_mimi)
        assert adapter.num_codebooks == 32

    def test_mimi_interleave_with_offset(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import MimiCodecAdapter

        mock_mimi = MagicMock()
        adapter = MimiCodecAdapter(TTS_PROFILES["sesame"], mock_mimi)

        # 3 codebooks for simplicity (sesame has 32 but logic is same)
        # 2 time steps, codebook_size=2048
        c0 = np.array([5, 10])
        c1 = np.array([15, 20])
        c2 = np.array([25, 30])

        tokens = adapter.interleave([c0, c1, c2])
        # Frame 0: c0[0]+0*2048, c1[0]+1*2048, c2[0]+2*2048
        # Frame 1: c0[1]+0*2048, c1[1]+1*2048, c2[1]+2*2048
        assert tokens[0] == 5        # 5 + 0*2048
        assert tokens[1] == 15 + 2048  # 15 + 1*2048
        assert tokens[2] == 25 + 4096  # 25 + 2*2048
        assert tokens[3] == 10       # 10 + 0*2048
        assert tokens[4] == 20 + 2048  # 20 + 1*2048
        assert tokens[5] == 30 + 4096  # 30 + 2*2048

    def test_mimi_deinterleave_roundtrip(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import MimiCodecAdapter

        mock_mimi = MagicMock()
        adapter = MimiCodecAdapter(TTS_PROFILES["sesame"], mock_mimi)

        # Use profile's 32 codebooks, 1 time step
        codes = [np.array([i * 10]) for i in range(32)]
        tokens = adapter.interleave(codes)
        assert len(tokens) == 32

        recovered = adapter.deinterleave(tokens)
        assert len(recovered) == 32
        for i in range(32):
            np.testing.assert_array_equal(recovered[i], codes[i])

    def test_mimi_interleave_empty(self):
        from mlx_tune.audio_profiles import TTS_PROFILES
        from mlx_tune.audio_codecs import MimiCodecAdapter

        mock_mimi = MagicMock()
        adapter = MimiCodecAdapter(TTS_PROFILES["sesame"], mock_mimi)
        assert adapter.interleave([]) == []


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestExports:
    """Test that profiles are exported from mlx_tune."""

    def test_import_tts_model_profile(self):
        from mlx_tune import TTSModelProfile
        assert TTSModelProfile is not None

    def test_import_stt_model_profile(self):
        from mlx_tune import STTModelProfile
        assert STTModelProfile is not None

    def test_import_tts_profiles(self):
        from mlx_tune import TTS_PROFILES
        assert "orpheus" in TTS_PROFILES
        assert "outetts" in TTS_PROFILES

    def test_import_stt_profiles(self):
        from mlx_tune import STT_PROFILES
        assert "whisper" in STT_PROFILES
        assert "moonshine" in STT_PROFILES


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test that existing constants still exist in tts.py and stt.py."""

    def test_orpheus_constants_still_exist(self):
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

    def test_whisper_constants_still_exist(self):
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

    def test_orpheus_profile_matches_constants(self):
        from mlx_tune.tts import (
            ORPHEUS_START_TOKEN,
            ORPHEUS_END_TOKENS,
            ORPHEUS_AUDIO_TOKEN_OFFSET,
            ORPHEUS_CODEBOOK_SIZE,
        )
        from mlx_tune.audio_profiles import TTS_PROFILES

        profile = TTS_PROFILES["orpheus"]
        assert profile.start_token == ORPHEUS_START_TOKEN
        assert list(profile.end_tokens) == ORPHEUS_END_TOKENS
        assert profile.audio_token_offset == ORPHEUS_AUDIO_TOKEN_OFFSET
        assert profile.codebook_size == ORPHEUS_CODEBOOK_SIZE

    def test_whisper_profile_matches_constants(self):
        from mlx_tune.stt import (
            WHISPER_SAMPLE_RATE,
            WHISPER_N_SAMPLES,
            WHISPER_N_MELS,
        )
        from mlx_tune.audio_profiles import STT_PROFILES

        profile = STT_PROFILES["whisper"]
        assert profile.sample_rate == WHISPER_SAMPLE_RATE
        assert profile.max_audio_samples == WHISPER_N_SAMPLES
        assert profile.n_mels == WHISPER_N_MELS
