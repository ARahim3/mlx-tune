/* ===== mlx-tune Docs — Pre-built Search Index ===== */
/* Regenerate by scanning HTML pages for headings and key phrases */

window.MLX_SEARCH_INDEX = [
  // --- Home / Getting Started ---
  { title: "Getting Started", section: "Installation", url: "index.html#quick-start", tags: "install pip setup requirements apple silicon mac", page: "Home" },
  { title: "Quick Start", section: "First fine-tune in 5 lines", url: "index.html#quick-start", tags: "quick start hello world first example", page: "Home" },
  { title: "Training Methods", section: "Supported training methods", url: "index.html#methods", tags: "sft dpo grpo kto simpo orpo training methods", page: "Home" },

  // --- LLM Fine-Tuning ---
  { title: "LLM Fine-Tuning", section: "Overview", url: "llm.html", tags: "llm language model text fine-tune lora", page: "LLM" },
  { title: "FastLanguageModel", section: "Model Loading", url: "llm.html#fast-language-model", tags: "fastlanguagemodel load model from_pretrained huggingface", page: "LLM" },
  { title: "SFTTrainer", section: "Supervised Fine-Tuning", url: "llm.html#sft", tags: "sfttrainer sftconfig supervised fine-tuning training", page: "LLM" },
  { title: "DPOTrainer", section: "Direct Preference Optimization", url: "llm.html#dpo", tags: "dpo preference optimization dpoconfig dpotrainer", page: "LLM" },
  { title: "ORPOTrainer", section: "ORPO Training", url: "llm.html#orpo", tags: "orpo odds ratio preference optimization", page: "LLM" },
  { title: "GRPOTrainer", section: "Group Relative Policy Optimization", url: "llm.html#grpo", tags: "grpo group relative policy reward function rl", page: "LLM" },
  { title: "KTOTrainer", section: "Kahneman-Tversky Optimization", url: "llm.html#kto", tags: "kto binary feedback kahneman tversky", page: "LLM" },
  { title: "SimPOTrainer", section: "Simple Preference Optimization", url: "llm.html#simpo", tags: "simpo simple preference no reference model", page: "LLM" },
  { title: "Chat Templates", section: "get_chat_template()", url: "llm.html#chat-templates", tags: "chat template llama gemma qwen phi mistral deepseek format", page: "LLM" },
  { title: "Response-Only Training", section: "train_on_responses_only()", url: "llm.html#response-only", tags: "response only train on responses prompt masking assistant", page: "LLM" },
  { title: "Dataset Utilities", section: "to_sharegpt / column mapping", url: "llm.html#datasets", tags: "dataset sharegpt alpaca column mapping hfdatasetconfig", page: "LLM" },
  { title: "MoE Fine-Tuning", section: "Mixture of Experts", url: "llm.html#moe", tags: "moe mixture experts switch linear qwen phi mixtral gemma4", page: "LLM" },
  { title: "Continual Pretraining", section: "CPTTrainer / CPTConfig", url: "llm.html#cpt", tags: "cpt continual pretraining raw text domain language adaptation embedding learning rate", page: "LLM" },
  { title: "CPTTrainer", section: "Continual Pretraining Trainer", url: "llm.html#CPTTrainer", tags: "cpttrainer cptconfig continual pretraining embed_tokens lm_head decoupled lr", page: "LLM" },
  { title: "LFM2 Models", section: "Liquid Foundation Models", url: "llm.html#cpt", tags: "lfm2 lfm2.5 liquid ai hybrid gated conv gqa attention in_proj out_proj w1 w2 w3", page: "LLM" },
  { title: "Embedding Models", section: "Contrastive Learning", url: "llm.html#embedding", tags: "embedding sentence transformer bert modernbert qwen3 harrier infonce contrastive", page: "LLM" },
  { title: "Save & Export", section: "GGUF, LoRA, Merged", url: "llm.html#save", tags: "save export gguf lora merge push hub save_pretrained_merged convert", page: "LLM" },

  // --- VLM (Vision) ---
  { title: "Vision Fine-Tuning", section: "Overview", url: "vlm.html", tags: "vlm vision language model image multimodal", page: "VLM" },
  { title: "FastVisionModel", section: "Model Loading", url: "vlm.html#quick-start", tags: "fastvisionmodel vision model load qwen gemma4 paligemma", page: "VLM" },
  { title: "VLMSFTTrainer", section: "Vision SFT Training", url: "vlm.html#vlm-sft", tags: "vlmsfttrainer vision sft training image text", page: "VLM" },
  { title: "UnslothVisionDataCollator", section: "Data Collation", url: "vlm.html#data-collator", tags: "unslothvisiondatacollator collator data image batch", page: "VLM" },
  { title: "VLMGRPOTrainer", section: "Vision GRPO", url: "vlm.html#vlm-grpo", tags: "vlmgrpo vision grpo reward function rl reasoning", page: "VLM" },
  { title: "Gemma 4", section: "Gemma 4 VLM Support", url: "vlm.html", tags: "gemma 4 gemma4 e2b e4b 26b 31b google vision multimodal", page: "VLM" },
  { title: "Gemma 4 Audio", section: "Audio Fine-Tuning (STT/ASR)", url: "vlm.html#tips", tags: "gemma 4 audio stt asr conformer speech transcribe finetune_audio_layers", page: "VLM" },

  // --- OCR ---
  { title: "OCR Fine-Tuning", section: "Overview", url: "ocr.html", tags: "ocr document understanding text recognition", page: "OCR" },
  { title: "FastOCRModel", section: "Model Loading", url: "ocr.html#quick-start", tags: "fastocrmodel ocr model deepseek glm olmocr", page: "OCR" },
  { title: "OCR Metrics", section: "CER / WER / Exact Match", url: "ocr.html#metrics", tags: "cer wer exact match character error rate word error", page: "OCR" },
  { title: "OCR GRPO", section: "RL for OCR", url: "ocr.html#grpo", tags: "ocr grpo rl reward cer reward", page: "OCR" },

  // --- Audio ---
  { title: "Audio Fine-Tuning", section: "TTS & STT Overview", url: "audio.html", tags: "audio tts stt text speech voice", page: "Audio" },
  { title: "TTS Fine-Tuning", section: "Text-to-Speech", url: "audio.html#tts", tags: "tts text speech orpheus outetts spark sesame qwen3-tts", page: "Audio" },
  { title: "FastTTSModel", section: "TTS Model Loading", url: "audio.html#tts", tags: "fastttsmodel tts load orpheus outetts", page: "Audio" },
  { title: "STT Fine-Tuning", section: "Speech-to-Text", url: "audio.html#stt", tags: "stt speech text whisper moonshine qwen3-asr canary voxtral voxtral-realtime streaming", page: "Audio" },
  { title: "FastSTTModel", section: "STT Model Loading", url: "audio.html#stt", tags: "faststtmodel stt load whisper moonshine voxtral", page: "Audio" },
  { title: "Voxtral Realtime", section: "Streaming ASR", url: "audio.html#stt-voxtral-realtime", tags: "voxtral realtime streaming asr mistral causal encoder ada rms norm tekkenizer wq wk wv wo feed_forward_w1 third architecture", page: "Audio" },
  { title: "Audio Codecs", section: "SNAC / DAC / BiCodec / Mimi", url: "audio.html#codecs", tags: "codec snac dac bicodec mimi audio tokenizer", page: "Audio" },
  { title: "Gemma 4 Audio (VLM)", section: "Conformer STT via VLM pipeline", url: "audio.html#models", tags: "gemma 4 audio conformer stt asr vlm fastvisionmodel", page: "Audio" },

  // --- Workflow ---
  { title: "Unsloth to mlx-tune", section: "Migration Guide", url: "workflow.html", tags: "unsloth migration import translate convert cuda mlx", page: "Workflow" },
  { title: "Import Translation", section: "Change imports", url: "workflow.html#imports", tags: "import from unsloth from mlx_tune translation", page: "Workflow" },
  { title: "Config Translation", section: "SFTConfig differences", url: "workflow.html#config", tags: "config sftconfig parameters differences", page: "Workflow" },

  // --- Examples ---
  { title: "Examples", section: "All Examples", url: "examples.html", tags: "examples code sample tutorial notebook", page: "Examples" },
  { title: "Gemma 4 Vision", section: "Example 38", url: "examples.html#vlm", tags: "gemma 4 vision example 38 fine-tune", page: "Examples" },
  { title: "Gemma 4 Text-to-SQL", section: "Example 39", url: "examples.html#vlm", tags: "gemma 4 text sql example 39 google", page: "Examples" },
  { title: "Gemma 4 MoE", section: "Example 40", url: "examples.html#vlm", tags: "gemma 4 moe example 40 26b experts", page: "Examples" },
  { title: "Vision GRPO", section: "Example 26", url: "examples.html#vlm", tags: "vision grpo rl example 26 reward", page: "Examples" },
  { title: "Orpheus TTS", section: "Example 12", url: "examples.html#audio", tags: "orpheus tts speech example 12", page: "Examples" },
  { title: "Whisper STT", section: "Example 13", url: "examples.html#audio", tags: "whisper stt speech example 13", page: "Examples" },
  { title: "Embedding Fine-Tuning", section: "Example 27", url: "examples.html#embedding", tags: "embedding example 27 sentence bert infonce", page: "Examples" },
  { title: "MoE Fine-Tuning", section: "Example 29", url: "examples.html#moe", tags: "moe example 29 qwen3.5 mixture experts", page: "Examples" },
  { title: "LFM2 SFT", section: "Example 41", url: "examples.html#sft", tags: "lfm2 liquid ai sft example 41 hybrid gated conv", page: "Examples" },
  { title: "LFM2.5 Thinking", section: "Example 42", url: "examples.html#sft", tags: "lfm2 liquid ai thinking reasoning example 42 chain-of-thought", page: "Examples" },
  { title: "CPT Language Adaptation", section: "Example 43", url: "examples.html#cpt", tags: "cpt continual pretraining language adaptation example 43 turkish", page: "Examples" },
  { title: "CPT Domain Knowledge", section: "Example 44", url: "examples.html#cpt", tags: "cpt domain knowledge medical scientific example 44", page: "Examples" },
  { title: "CPT Code Capabilities", section: "Example 45", url: "examples.html#cpt", tags: "cpt code programming capabilities example 45 mlx", page: "Examples" },
  { title: "LFM2 + CPT", section: "Example 46", url: "examples.html#cpt", tags: "lfm2 cpt continual pretraining domain adaptation example 46", page: "Examples" },
  { title: "Gemma 4 Audio ASR", section: "Example 47", url: "examples.html#vlm", tags: "gemma 4 audio asr stt example 47 conformer transcribe", page: "Examples" },
  { title: "Gemma 4 Audio Understanding", section: "Example 48", url: "examples.html#vlm", tags: "gemma 4 audio understanding qa example 48 tower lora", page: "Examples" },
  { title: "Voxtral Realtime Streaming STT", section: "Example 49", url: "examples.html#ex49", tags: "voxtral realtime streaming stt example 49 mistral librispeech 4-bit causal encoder ada rms norm", page: "Examples" },

  // --- Troubleshooting ---
  { title: "Troubleshooting", section: "Common Issues", url: "troubleshooting.html", tags: "troubleshoot error fix help debug issue", page: "Help" },
  { title: "Model Not Found", section: "HuggingFace errors", url: "troubleshooting.html#model-not-found", tags: "model not found 404 huggingface gated token", page: "Help" },
  { title: "Out of Memory", section: "OOM errors", url: "troubleshooting.html#oom", tags: "oom out memory killed crash ram unified", page: "Help" },
  { title: "GGUF Export", section: "Export issues", url: "troubleshooting.html#gguf", tags: "gguf export quantized limitation ollama llama.cpp", page: "Help" },
  { title: "NaN Loss", section: "Training produces NaN", url: "troubleshooting.html#nan", tags: "nan loss training gradient explode infinity", page: "Help" },

  // --- API Reference ---
  { title: "API Reference", section: "Complete API", url: "api.html", tags: "api reference documentation class method function", page: "API" },
  { title: "FastLanguageModel API", section: "from_pretrained / get_peft_model", url: "api.html#fastlanguagemodel", tags: "api fastlanguagemodel from_pretrained get_peft_model", page: "API" },
  { title: "FastVisionModel API", section: "VLM API", url: "api.html#fastvisionmodel", tags: "api fastvisionmodel vision", page: "API" },
  { title: "Chat Template API", section: "get_chat_template / train_on_responses_only", url: "api.html#chat-templates", tags: "api chat template get_chat_template train_on_responses_only", page: "API" },
];
