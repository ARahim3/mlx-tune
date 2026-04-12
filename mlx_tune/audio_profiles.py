"""
Audio Model Profiles for MLX-Tune

Declarative profiles that capture all model-specific constants and behaviors
for TTS and STT models. This replaces hardcoded values scattered throughout
tts.py and stt.py with centralized, extensible configuration.

Usage:
    from mlx_tune.audio_profiles import TTS_PROFILES, STT_PROFILES

    orpheus = TTS_PROFILES["orpheus"]
    whisper = STT_PROFILES["whisper"]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import re


# ---------------------------------------------------------------------------
# TTS Model Profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTSModelProfile:
    """
    Declarative profile for a TTS model architecture.

    Captures all model-specific constants (token IDs, codec config,
    prompt format, LoRA targets, etc.) in a single frozen dataclass.
    """

    # Identity
    name: str
    architecture: str  # "decoder_only" | "backbone_decoder"

    # Audio codec
    codec_type: str  # "snac" | "dac" | "bicodec" | "mimi" | "qwen3_speech"
    codec_repo: str  # default codec model repo
    sample_rate: int  # codec output sample rate

    # Special tokens (Optional for text-token models like OuteTTS/Spark)
    start_token: Optional[int] = None
    end_tokens: Tuple[int, ...] = ()
    audio_token_offset: int = 0
    codebook_size: int = 0

    # VQ codec settings
    num_codebooks: int = 1
    interleave_pattern: Tuple[int, ...] = (1,)

    # Token format: "numeric" (SNAC/Mimi style) or "text" (OuteTTS/Spark style)
    token_format: str = "numeric"

    # Per-codebook text format for text-token models (e.g., "<|c1_{code}|>")
    # One format string per codebook, empty for numeric-token models
    audio_token_formats: Tuple[str, ...] = ()

    # Prompt format
    prompt_template: str = "{speaker}: {text}"
    default_speaker: str = ""

    # LoRA
    lora_target_modules: Tuple[str, ...] = ()
    lora_module_mapping: Dict[str, str] = field(default_factory=dict)

    # Model loader identifier
    loader: str = "mlx_lm"  # "mlx_lm" | "mlx_audio_tts"

    # Inner model path for mlx_audio models (e.g. "model" to access model.model)
    inner_model_attr: Optional[str] = None


# ---------------------------------------------------------------------------
# STT Model Profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class STTModelProfile:
    """
    Declarative profile for an STT model architecture.

    Captures all model-specific constants (sample rate, mel config,
    block paths, LoRA targets, etc.) in a single frozen dataclass.
    """

    # Identity
    name: str

    # Architecture type:
    #   "encoder_decoder" — Whisper, Canary, Moonshine
    #   "audio_llm"        — Qwen3-ASR, Voxtral (audio tower + LLM with placeholder tokens)
    #   "voxtral_realtime" — Voxtral Realtime (causal encoder + decoder with additive
    #                        embedding fusion, AdaRMSNorm time conditioning, no LLM split)
    architecture: str = "encoder_decoder"

    # Audio preprocessing
    sample_rate: int = 16000
    preprocessor: str = "log_mel_spectrogram"  # "log_mel_spectrogram" | "raw_conv"
    n_mels: int = 80
    max_audio_samples: int = 480000  # e.g. 480000 for 30s at 16kHz

    # Model structure paths
    encoder_block_path: str = "encoder.blocks"
    decoder_block_path: str = "decoder.blocks"
    attn_names: Dict[str, str] = field(default_factory=lambda: {"self_attn": "attn", "cross_attn": "cross_attn"})
    cross_attn_attr: str = "cross_attn"

    # Tokenizer
    sot_token_id: int = 50258

    # LoRA — target module names within attention blocks
    lora_target_modules: Tuple[str, ...] = ("query", "key", "value", "out")

    # Separate encoder LoRA targets (if encoder uses different names than decoder)
    encoder_lora_targets: Optional[Tuple[str, ...]] = None

    # Optional FFN/MLP target names within decoder layers (e.g. SwiGLU w1/w2/w3)
    # Used by Voxtral Realtime where the FFN modules live alongside attention.
    decoder_ffn_targets: Optional[Tuple[str, ...]] = None

    # Audio-LLM specific
    audio_token_id: Optional[int] = None  # Placeholder token replaced by audio features

    # Inner model path for wrapper models (e.g., "_model" for Qwen3-ASR)
    inner_model_attr: Optional[str] = None

    # Model loader
    loader: str = "mlx_audio_stt"


# ---------------------------------------------------------------------------
# Llama-style LoRA module mapping (shared by Orpheus, OuteTTS, Spark, Sesame)
# ---------------------------------------------------------------------------

_LLAMA_LORA_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

_LLAMA_LORA_MAPPING = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


# ---------------------------------------------------------------------------
# TTS Profiles
# ---------------------------------------------------------------------------

_ORPHEUS_PROFILE = TTSModelProfile(
    name="orpheus",
    architecture="decoder_only",
    codec_type="snac",
    codec_repo="mlx-community/snac_24khz",
    sample_rate=24000,
    start_token=128259,  # <custom_token_3>
    end_tokens=(128009, 128260),  # <|eot_id|>, <custom_token_4>
    audio_token_offset=128266,
    codebook_size=4096,
    num_codebooks=3,
    interleave_pattern=(1, 2, 4),  # L0: 1 code, L1: 2 codes, L2: 4 codes per frame
    token_format="numeric",
    prompt_template="<custom_token_3><|begin_of_text|>{speaker}: {text}<|eot_id|>",
    default_speaker="tara",
    lora_target_modules=_LLAMA_LORA_MODULES,
    lora_module_mapping=_LLAMA_LORA_MAPPING,
    loader="mlx_lm",
)

_OUTETTS_PROFILE = TTSModelProfile(
    name="outetts",
    architecture="decoder_only",
    codec_type="dac",
    codec_repo="mlx-community/dac-speech-24khz-1.5kbps",
    sample_rate=24000,
    num_codebooks=2,
    interleave_pattern=(1, 1),  # 2 codebooks, flat sequential
    token_format="text",  # Uses <|c1_X|> and <|c2_X|> text tokens
    audio_token_formats=("<|c1_{code}|>", "<|c2_{code}|>"),
    prompt_template="<|text_start|>{text}<|text_end|><|audio_start|>",
    default_speaker="",
    lora_target_modules=_LLAMA_LORA_MODULES,
    lora_module_mapping=_LLAMA_LORA_MAPPING,
    loader="mlx_audio_tts",
    inner_model_attr="model",  # OuteTTS wraps Llama/Qwen as model.model
)

_SPARK_PROFILE = TTSModelProfile(
    name="spark",
    architecture="decoder_only",
    codec_type="bicodec",
    codec_repo="",  # BiCodec is bundled with Spark model
    sample_rate=16000,
    num_codebooks=2,  # global + semantic
    interleave_pattern=(1, 1),
    token_format="text",  # Uses <|bicodec_global_X|> and <|bicodec_semantic_X|>
    audio_token_formats=("<|bicodec_global_{code}|>", "<|bicodec_semantic_{code}|>"),
    prompt_template="<|tts|><|start_content|>{text}<|end_content|>",
    default_speaker="",
    lora_target_modules=_LLAMA_LORA_MODULES,
    lora_module_mapping=_LLAMA_LORA_MAPPING,
    loader="mlx_audio_tts",
    inner_model_attr="model",  # Spark wraps Qwen2 as model.model
)

_SESAME_PROFILE = TTSModelProfile(
    name="sesame",
    architecture="backbone_decoder",  # Dual Llama: backbone + depth decoder
    codec_type="mimi",
    codec_repo="",  # Mimi is typically bundled or loaded separately
    sample_rate=24000,
    codebook_size=2048,  # audio_vocab_size per codebook
    num_codebooks=32,
    interleave_pattern=(1,) * 32,  # One token per codebook per frame
    token_format="numeric",  # Numeric IDs with offset: token + codebook * vocab_size
    prompt_template="{text}",
    default_speaker="",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    lora_module_mapping={
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
    },
    loader="mlx_audio_tts",
    inner_model_attr="model",  # SesameModel has backbone and decoder
)

_QWEN3_TTS_PROFILE = TTSModelProfile(
    name="qwen3_tts",
    architecture="decoder_only",
    codec_type="qwen3_speech",  # Built-in speech tokenizer (16 codebooks, 12.5Hz)
    codec_repo="",  # Codec is part of the model (speech_tokenizer)
    sample_rate=24000,
    start_token=2149,       # codec_bos_id
    end_tokens=(2150,),     # codec_eos_token_id
    audio_token_offset=0,
    codebook_size=2048,
    num_codebooks=16,       # All 16 codebooks needed for input embedding construction
    interleave_pattern=(1,),  # Only code_0 is predicted by the talker
    token_format="numeric",
    prompt_template="{text}",  # Text handled via separate embedding path
    default_speaker="",
    lora_target_modules=_LLAMA_LORA_MODULES,
    lora_module_mapping=_LLAMA_LORA_MAPPING,
    loader="mlx_audio_tts",
    inner_model_attr="talker",  # LoRA applied to talker's transformer layers
)


# ---------------------------------------------------------------------------
# STT Profiles
# ---------------------------------------------------------------------------

_WHISPER_PROFILE = STTModelProfile(
    name="whisper",
    sample_rate=16000,
    preprocessor="log_mel_spectrogram",
    n_mels=80,
    max_audio_samples=480000,  # 30 seconds at 16kHz
    encoder_block_path="encoder.blocks",
    decoder_block_path="decoder.blocks",
    attn_names={"self_attn": "attn", "cross_attn": "cross_attn"},
    cross_attn_attr="cross_attn",
    sot_token_id=50258,
    lora_target_modules=("query", "key", "value", "out"),
    loader="mlx_audio_stt",
)

_MOONSHINE_PROFILE = STTModelProfile(
    name="moonshine",
    sample_rate=16000,
    preprocessor="raw_conv",  # Raw waveform through conv frontend, no mel
    n_mels=0,  # Not applicable
    max_audio_samples=0,  # Variable length (no fixed pad_or_trim)
    encoder_block_path="encoder.layers",
    decoder_block_path="decoder.layers",
    attn_names={"self_attn": "self_attn", "cross_attn": "encoder_attn"},
    cross_attn_attr="encoder_attn",
    sot_token_id=1,  # decoder_start_token_id
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    loader="mlx_audio_stt",
)

_QWEN3_ASR_PROFILE = STTModelProfile(
    name="qwen3_asr",
    architecture="audio_llm",
    sample_rate=16000,
    preprocessor="log_mel_spectrogram",
    n_mels=128,
    max_audio_samples=0,  # Variable length
    encoder_block_path="_model.audio_tower.layers",
    decoder_block_path="_model.model.layers",
    attn_names={"self_attn": "self_attn"},
    cross_attn_attr="",  # No cross-attention in LLM decoder
    sot_token_id=0,  # Not used for audio-LLM
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    encoder_lora_targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    audio_token_id=151676,
    inner_model_attr=None,  # Model wrapper delegates via __call__
    loader="mlx_audio_stt",
)

_CANARY_PROFILE = STTModelProfile(
    name="canary",
    architecture="encoder_decoder",
    sample_rate=16000,
    preprocessor="canary_mel",  # Uses parakeet's log_mel_spectrogram via model._preprocess_audio
    n_mels=128,
    max_audio_samples=0,  # Variable length
    encoder_block_path="encoder.conformer.layers",
    decoder_block_path="decoder.blocks",
    attn_names={"self_attn": "self_attn", "cross_attn": "cross_attn"},
    cross_attn_attr="cross_attn",
    sot_token_id=0,
    lora_target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
    encoder_lora_targets=("linear_q", "linear_k", "linear_v", "linear_out"),
    loader="mlx_audio_stt",
)

_VOXTRAL_PROFILE = STTModelProfile(
    name="voxtral",
    architecture="audio_llm",
    sample_rate=16000,
    preprocessor="log_mel_spectrogram",
    n_mels=128,
    max_audio_samples=0,  # Variable length
    encoder_block_path="audio_tower.layers",
    decoder_block_path="language_model.model.layers",
    attn_names={"self_attn": "self_attn"},
    cross_attn_attr="",  # No cross-attention in LLM decoder
    sot_token_id=0,
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    encoder_lora_targets=("q_proj", "k_proj", "v_proj", "out_proj"),
    audio_token_id=24,  # Voxtral audio placeholder token
    loader="mlx_audio_stt",
)

_VOXTRAL_REALTIME_PROFILE = STTModelProfile(
    name="voxtral_realtime",
    # Third architecture type — streaming ASR with causal encoder + additive embedding
    # fusion + AdaRMSNorm time conditioning. No audio_tower/language_model split.
    architecture="voxtral_realtime",
    sample_rate=16000,
    preprocessor="log_mel_spectrogram",
    n_mels=128,
    max_audio_samples=0,  # Variable length
    # decoder.layers[i].attention.{wq,wk,wv,wo} + .feed_forward_w{1,2,3}
    decoder_block_path="decoder.layers",
    # encoder.transformer_layers[i].attention.{wq,wk,wv,wo}  (frozen by default)
    encoder_block_path="encoder.transformer_layers",
    attn_names={"self_attn": "attention"},  # inner attn module is named .attention
    cross_attn_attr="",  # No cross-attention in this architecture
    sot_token_id=1,  # bos_token_id
    # Mistral-internal naming (NOT q_proj/k_proj — keep as wq/wk/wv/wo)
    lora_target_modules=("wq", "wk", "wv", "wo"),
    encoder_lora_targets=("wq", "wk", "wv", "wo"),
    decoder_ffn_targets=("feed_forward_w1", "feed_forward_w2", "feed_forward_w3"),
    loader="mlx_audio_stt",
)


# ---------------------------------------------------------------------------
# Profile Registries
# ---------------------------------------------------------------------------

TTS_PROFILES: Dict[str, TTSModelProfile] = {
    "orpheus": _ORPHEUS_PROFILE,
    "outetts": _OUTETTS_PROFILE,
    "spark": _SPARK_PROFILE,
    "sesame": _SESAME_PROFILE,
    "qwen3_tts": _QWEN3_TTS_PROFILE,
}

STT_PROFILES: Dict[str, STTModelProfile] = {
    "whisper": _WHISPER_PROFILE,
    "moonshine": _MOONSHINE_PROFILE,
    "qwen3_asr": _QWEN3_ASR_PROFILE,
    "canary": _CANARY_PROFILE,
    # voxtral_realtime MUST be checked before voxtral so the realtime variant
    # gets matched first. The voxtral pattern also has a negative lookahead as
    # belt-and-braces, but explicit ordering is the primary guard.
    "voxtral_realtime": _VOXTRAL_REALTIME_PROFILE,
    "voxtral": _VOXTRAL_PROFILE,
}


# ---------------------------------------------------------------------------
# Auto-detection functions
# ---------------------------------------------------------------------------

_TTS_PATTERNS: Dict[str, List[str]] = {
    "orpheus": [r"orpheus", r"canopylabs.*orpheus"],
    "outetts": [r"outetts", r"outeai.*outetts"],
    "spark": [r"spark[-_]?tts", r"sparkaudio"],
    "sesame": [r"sesame", r"csm[-_]?1b", r"marvis"],
    "qwen3_tts": [r"qwen3[-_]?tts", r"Qwen3[-_]?TTS"],
}

_STT_PATTERNS: Dict[str, List[str]] = {
    "whisper": [
        r"whisper",
        r"openai.*whisper",
        r"distil[-_]?whisper",
        r"whisper[-_](tiny|base|small|medium|large)",
    ],
    "moonshine": [r"moonshine", r"useful[-_]?sensors.*moonshine"],
    "qwen3_asr": [r"qwen3[-_]?asr", r"Qwen3[-_]?ASR"],
    "canary": [r"canary", r"nvidia.*canary"],
    # Voxtral Realtime — must be checked BEFORE the regular voxtral pattern.
    # Dict iteration in Python 3.7+ preserves insertion order, so this works.
    "voxtral_realtime": [r"voxtral.*realtime", r"Voxtral.*Realtime"],
    # Match regular Voxtral but NOT Voxtral Realtime (different streaming architecture).
    # Negative lookahead is defensive in case detection is called with a custom dict order.
    "voxtral": [r"voxtral(?!.*realtime)", r"mistral.*voxtral(?!.*realtime)", r"Voxtral(?!.*Realtime)"],
}


def detect_tts_model_type(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Auto-detect TTS model type from model name and/or config.

    Returns:
        Profile key (e.g. "orpheus", "outetts") or None if unrecognized
    """
    name_lower = model_name.lower()

    for profile_key, patterns in _TTS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return profile_key

    # Check config model_type field as fallback
    if config:
        mt = config.get("model_type", "").lower()
        if mt in TTS_PROFILES:
            return mt

    return None


def detect_stt_model_type(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Auto-detect STT model type from model name and/or config.

    Returns:
        Profile key (e.g. "whisper", "moonshine") or None if unrecognized
    """
    name_lower = model_name.lower()

    for profile_key, patterns in _STT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return profile_key

    if config:
        mt = config.get("model_type", "").lower()
        if mt in STT_PROFILES:
            return mt

    return None
