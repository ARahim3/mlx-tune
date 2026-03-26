"""
MLX-Tune: MLX-powered LLM fine-tuning for Apple Silicon

A drop-in replacement for Unsloth that uses Apple's MLX framework instead of CUDA/Triton kernels.

Supported Training Methods:
- SFT (Supervised Fine-Tuning)
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization) - DeepSeek R1 style
- KTO (Kahneman-Tversky Optimization)
- SimPO (Simple Preference Optimization)
- VLM (Vision Language Model) fine-tuning
- TTS (Text-to-Speech) fine-tuning
- STT (Speech-to-Text) fine-tuning
"""

__version__ = "0.4.10"  # Qwen3-TTS fine-tuning support

from mlx_tune.model import (
    FastLanguageModel,
    ReferencePolicy,
    RLModelRoles,
    RewardModel,
    ValueModel,
    build_value_model,
    create_rl_model_roles,
)
from mlx_tune.trainer import (
    prepare_dataset,
    format_chat_template,
    create_training_data,
    save_model_hf_format,
    export_to_gguf,
    get_training_config,
)
from mlx_tune.sft_trainer import SFTTrainer, SFTConfig, TrainingArguments
from mlx_tune.rl_api import (
    RLCheckpointBundle,
    PreparedRLDataset,
    prepare_rl_dataset,
    build_reference_policy,
    build_reward_model,
    create_reward_function,
    resume_from_checkpoint,
)

# RL Trainers
from mlx_tune.rl_trainers import (
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
    KTOConfig,
    SimPOConfig,
    KTOTrainer,
    SimPOTrainer,
    prepare_reward_dataset,
    prepare_preference_dataset,
    score_reward_model,
)

# Loss functions for custom training
from mlx_tune.losses import (
    compute_log_probs,
    compute_log_probs_with_lengths,
    compute_completion_log_probs,
    dpo_loss,
    orpo_loss,
    kto_loss,
    simpo_loss,
    sft_loss,
    grpo_loss,
    grpo_batch_loss,
    compute_reference_logprobs,
    pairwise_reward_loss,
    reward_model_pairwise_loss,
    reward_model_regression_loss,
    value_regression_loss,
    value_model_regression_loss,
    scalar_loss_metrics,
    pairwise_ranking_accuracy,
    precompute_preference_reference_logprobs,
    precompute_kto_reference_logprobs,
    ppo_sequence_loss,
)

# Vision Language Models
from mlx_tune.vlm import (
    FastVisionModel,
    VLMSFTTrainer,
    VLMSFTConfig,
    VLMModelWrapper,
    UnslothVisionDataCollator,
    load_vlm_dataset,
)

# Text-to-Speech Models
from mlx_tune.tts import (
    FastTTSModel,
    TTSModelWrapper,
    TTSSFTTrainer,
    TTSSFTConfig,
    TTSDataCollator,
)

# Speech-to-Text Models
from mlx_tune.stt import (
    FastSTTModel,
    STTModelWrapper,
    STTSFTTrainer,
    STTSFTConfig,
    STTDataCollator,
    STTProcessor,
)

# Audio Profiles and Codec Adapters
from mlx_tune.audio_profiles import (
    TTSModelProfile,
    STTModelProfile,
    TTS_PROFILES,
    STT_PROFILES,
    detect_tts_model_type,
    detect_stt_model_type,
)
from mlx_tune.audio_codecs import (
    CodecAdapter,
    create_codec,
)

# Chat Templates and Dataset Formatting (Unsloth-compatible)
from mlx_tune.chat_templates import (
    # Dataset format detection and conversion
    detect_dataset_format,
    standardize_sharegpt,
    standardize_sharegpt_enhanced,
    convert_to_mlx_format,
    get_formatting_func,
    apply_chat_template_to_sample,
    alpaca_to_text,
    # Chat template functions (Unsloth-compatible)
    get_chat_template,
    list_chat_templates,
    get_template_info,
    get_template_for_model,
    # Response-only training (Unsloth-compatible)
    train_on_responses_only,
    # Template registry
    CHAT_TEMPLATES,
    TEMPLATE_ALIASES,
    DEFAULT_SYSTEM_MESSAGES,
    ChatTemplateEntry,
    # Multi-turn conversation merging (Unsloth-compatible)
    to_sharegpt,
    # Column mapping (Unsloth-compatible)
    apply_column_mapping,
    infer_column_mapping,
    # HF dataset config (Unsloth-compatible)
    HFDatasetConfig,
    load_dataset_with_config,
)
from mlx_tune.trl_compat import PatchFastRL

__all__ = [
    # Core
    "FastLanguageModel",
    "ReferencePolicy",
    "RLModelRoles",
    "RewardModel",
    "ValueModel",
    "build_reference_policy",
    "build_reward_model",
    "build_value_model",
    "create_rl_model_roles",
    "PreparedRLDataset",
    "RLCheckpointBundle",
    "prepare_rl_dataset",
    "resume_from_checkpoint",
    "PatchFastRL",
    "__version__",
    # SFT Training
    "SFTTrainer",
    "SFTConfig",
    "TrainingArguments",
    # RL Trainers
    "DPOTrainer",
    "DPOConfig",
    "ORPOTrainer",
    "ORPOConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "RewardTrainer",
    "RewardConfig",
    "PPOTrainer",
    "PPOConfig",
    "OnlineDPOTrainer",
    "OnlineDPOConfig",
    "KTOConfig",
    "SimPOConfig",
    "KTOTrainer",
    "SimPOTrainer",
    # Vision Models
    "FastVisionModel",
    "VLMSFTTrainer",
    "VLMSFTConfig",
    "VLMModelWrapper",
    "UnslothVisionDataCollator",
    # Text-to-Speech Models
    "FastTTSModel",
    "TTSModelWrapper",
    "TTSSFTTrainer",
    "TTSSFTConfig",
    "TTSDataCollator",
    # Speech-to-Text Models
    "FastSTTModel",
    "STTModelWrapper",
    "STTSFTTrainer",
    "STTSFTConfig",
    "STTDataCollator",
    "STTProcessor",
    # Audio Profiles and Codec Adapters
    "TTSModelProfile",
    "STTModelProfile",
    "TTS_PROFILES",
    "STT_PROFILES",
    "detect_tts_model_type",
    "detect_stt_model_type",
    "CodecAdapter",
    "create_codec",
    # Loss Functions
    "compute_log_probs",
    "compute_log_probs_with_lengths",
    "compute_completion_log_probs",
    "dpo_loss",
    "orpo_loss",
    "kto_loss",
    "simpo_loss",
    "sft_loss",
    "grpo_loss",
    "grpo_batch_loss",
    "compute_reference_logprobs",
    "pairwise_reward_loss",
    "reward_model_pairwise_loss",
    "reward_model_regression_loss",
    "value_regression_loss",
    "value_model_regression_loss",
    "scalar_loss_metrics",
    "pairwise_ranking_accuracy",
    "precompute_preference_reference_logprobs",
    "precompute_kto_reference_logprobs",
    "ppo_sequence_loss",
    # Utilities
    "prepare_dataset",
    "prepare_reward_dataset",
    "prepare_preference_dataset",
    "format_chat_template",
    "create_training_data",
    "save_model_hf_format",
    "export_to_gguf",
    "get_training_config",
    "create_reward_function",
    "score_reward_model",
    "load_vlm_dataset",
    # Chat Templates and Dataset Formatting
    "detect_dataset_format",
    "standardize_sharegpt",
    "standardize_sharegpt_enhanced",
    "convert_to_mlx_format",
    "get_formatting_func",
    "apply_chat_template_to_sample",
    "alpaca_to_text",
    # Chat Template Functions (Unsloth-compatible)
    "get_chat_template",
    "list_chat_templates",
    "get_template_info",
    "get_template_for_model",
    # Response-only Training (Unsloth-compatible)
    "train_on_responses_only",
    # Template Registry
    "CHAT_TEMPLATES",
    "TEMPLATE_ALIASES",
    "DEFAULT_SYSTEM_MESSAGES",
    "ChatTemplateEntry",
    # Multi-turn Conversation Merging (Unsloth-compatible)
    "to_sharegpt",
    # Column Mapping (Unsloth-compatible)
    "apply_column_mapping",
    "infer_column_mapping",
    # HF Dataset Config (Unsloth-compatible)
    "HFDatasetConfig",
    "load_dataset_with_config",
]
