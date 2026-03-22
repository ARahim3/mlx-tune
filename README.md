<p align="center">
  <img src="https://raw.githubusercontent.com/ARahim3/mlx-tune/main/mlx-tune-logo.png" alt="MLX-Tune Logo" width="300"/>
</p>

<p align="center">
  <strong>Fine-tune LLMs on your Mac with Apple Silicon</strong><br>
  <em>SFT, DPO, GRPO, and Vision fine-tuning — natively on MLX. Unsloth-compatible API.</em>
</p>

<p align="center">
  <a href="https://github.com/ARahim3/mlx-tune"><img src="https://img.shields.io/github/stars/arahim3/mlx-tune?style=social" alt="GitHub stars"></a>
  <a href="https://pepy.tech/projects/unsloth-mlx"><img src="https://static.pepy.tech/personalized-badge/unsloth-mlx?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads" alt="PyPI Downloads"></a>
  <a href="https://github.com/ARahim3/mlx-tune"><img alt="GitHub forks" src="https://img.shields.io/github/forks/arahim3/mlx-tune"></a>
  <br>
  <a href="#installation"><img src="https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple" alt="Platform"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/MLX-0.20+-green" alt="MLX"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-orange" alt="License"></a>
</p>

<p align="center">
  <a href="https://arahim3.github.io/mlx-tune/">Documentation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#supported-training-methods">Training Methods</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#project-status">Status</a>
</p>

---

> [!NOTE]
> **Name Change**: This project was originally called `unsloth-mlx`. Since it's not an official Unsloth project and to avoid any confusion, it has been renamed to `mlx-tune`. The vision remains the same — bringing the Unsloth experience to Mac users via MLX. If you were using `unsloth-mlx`, simply switch to `pip install mlx-tune` and update your imports from `unsloth_mlx` to `mlx_tune`.

> [!NOTE]
> **Why I Built This (A Personal Note)**
>
> I rely on Unsloth for my daily fine-tuning on cloud GPUs—it's the gold standard for me. But recently, I started working on a MacBook M4 and hit a friction point: I wanted to prototype locally on my Mac, then scale up to the cloud without rewriting my entire training script.
>
> Since Unsloth relies on Triton (which Macs don't have, yet), I couldn't use it locally. I built `mlx-tune` to solve this specific "Context Switch" problem. It wraps Apple's native MLX framework in an Unsloth-compatible API.
>
> **The goal isn't to replace Unsloth or claim superior performance.** The goal is **code portability**: allowing you to write `FastLanguageModel` code once on your Mac, test it, and then push that *exact same script* to a CUDA cluster. It solves a workflow problem, not just a hardware one.
>
> This is an "unofficial" project built by a fan, for fans who happen to use Macs. It's helping me personally, and if it helps others like me, then I'll have my satisfaction.

## Why MLX-Tune?

Bringing the [Unsloth](https://github.com/unslothai/unsloth) experience to Mac users via Apple's [MLX](https://github.com/ml-explore/mlx) framework.

- 🚀 **Fine-tune LLMs locally** on your Mac (M1/M2/M3/M4/M5)
- 💾 **Leverage unified memory** (up to 512GB on Mac Studio)
- 🔄 **Unsloth-compatible API** - your existing training scripts just work!
- 📦 **Export anywhere** - HuggingFace format, GGUF for Ollama/llama.cpp

```python
# Unsloth (CUDA)                        # MLX-Tune (Apple Silicon)
from unsloth import FastLanguageModel   from mlx_tune import FastLanguageModel
from trl import SFTTrainer              from mlx_tune import SFTTrainer

# Rest of your code stays exactly the same!
```

## What This Is (and Isn't)

**This is NOT** a replacement for Unsloth or an attempt to compete with it. Unsloth is incredible - it's the gold standard for efficient LLM fine-tuning on CUDA.

**This IS** a bridge for Mac users who want to:
- 🧪 **Prototype locally** - Experiment with fine-tuning before committing to cloud GPU costs
- 📚 **Learn & iterate** - Develop your training pipeline with fast local feedback loops
- 🔄 **Then scale up** - Move to cloud NVIDIA GPUs + original Unsloth for production training

```
Local Mac (MLX-Tune)       →     Cloud GPU (Unsloth)
   Prototype & experiment          Full-scale training
   Small datasets                  Large datasets
   Quick iterations                Production runs
```

## Project Status

> 🚀 **v0.4.6** - Fix VLM save/load/export + mlx-vlm 0.4.0 compatibility

| Feature | Status | Notes |
|---------|--------|-------|
| SFT Training | ✅ Stable | Native MLX training |
| Model Loading | ✅ Stable | Any HuggingFace model (quantized & non-quantized) |
| Save/Export | ✅ Stable | HF format, GGUF ([see limitations](#known-limitations)) |
| DPO Training | ✅ Stable | **Full DPO loss** |
| ORPO Training | ✅ Stable | **Full ORPO loss** |
| GRPO Training | ✅ Stable | **Multi-generation + reward** |
| KTO/SimPO | ✅ Stable | Proper loss implementations |
| Chat Templates | ✅ Stable | 15 models (llama, gemma, qwen, phi, mistral) |
| Response-Only Training | ✅ Stable | `train_on_responses_only()` |
| Multi-turn Merging | ✅ Stable | `to_sharegpt()` + `conversation_extension` |
| Column Mapping | ✅ Stable | `apply_column_mapping()` auto-rename |
| Dataset Config | ✅ Stable | `HFDatasetConfig` structured loading |
| **Vision Models** | ✅ **NEW** | **Full VLM fine-tuning** via mlx-vlm |
| PyPI Package | ✅ Available | `uv pip install mlx-tune` |

## Installation

```bash
# Using uv (recommended - faster and more reliable)
uv pip install mlx-tune

# Or using pip
pip install mlx-tune

# From source (for development)
git clone https://github.com/ARahim3/mlx-tune.git
cd mlx-tune
uv pip install -e .
```

## Quick Start

```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset

# Load any HuggingFace model (1B model for quick start)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Load a dataset (or create your own)
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

# Train with SFTTrainer (same API as TRL!)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=50,
    ),
)
trainer.train()

# Save (same API as Unsloth!)
model.save_pretrained("lora_model")  # Adapters only
model.save_pretrained_merged("merged", tokenizer)  # Full model
model.save_pretrained_gguf("model", tokenizer)  # GGUF (see note below)
```

> [!NOTE]
> **GGUF Export**: Works with non-quantized base models. If using a 4-bit model (like above),
> see [Known Limitations](#known-limitations) for workarounds.

### Chat Templates & Response-Only Training

```python
from mlx_tune import get_chat_template, train_on_responses_only

# Apply chat template (supports llama-3, gemma, qwen, phi, mistral, etc.)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# Or auto-detect from model name
tokenizer = get_chat_template(tokenizer, chat_template="auto")

# Train only on responses (not prompts) - more efficient!
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

### Vision Model Fine-Tuning (NEW!)

Fine-tune vision-language models like Qwen3.5 on image+text tasks:

```python
from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig

# Load a vision model
model, processor = FastVisionModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-bf16",
)

# Add LoRA (same params as Unsloth!)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16, lora_alpha=16,
)

# Train on image-text data
FastVisionModel.for_training(model)
trainer = VLMSFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=dataset,
    args=VLMSFTConfig(max_steps=30, learning_rate=2e-4),
)
trainer.train()
```

See [`examples/10_qwen35_vision_finetuning.py`](examples/10_qwen35_vision_finetuning.py) for the full workflow, or [`examples/11_qwen35_text_finetuning.py`](examples/11_qwen35_text_finetuning.py) for text-only fine-tuning on Qwen3.5.

## Supported Training Methods

| Method | Trainer | Implementation | Use Case |
|--------|---------|----------------|----------|
| **SFT** | `SFTTrainer` | ✅ Native MLX | Instruction fine-tuning |
| **DPO** | `DPOTrainer` | ✅ Native MLX | Preference learning (proper log-prob loss) |
| **ORPO** | `ORPOTrainer` | ✅ Native MLX | Combined SFT + odds ratio preference |
| **GRPO** | `GRPOTrainer` | ✅ Native MLX | Reasoning with multi-generation (DeepSeek R1 style) |
| **KTO** | `KTOTrainer` | ✅ Native MLX | Kahneman-Tversky optimization |
| **SimPO** | `SimPOTrainer` | ✅ Native MLX | Simple preference optimization |
| **VLM SFT** | `VLMSFTTrainer` | ✅ Native MLX | Vision-Language model fine-tuning |

## Examples

Check [`examples/`](examples/) for working code:
- Basic model loading and inference
- Complete SFT fine-tuning pipeline
- RL training methods (DPO, GRPO, ORPO)

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4/M5)
- **OS**: macOS 13.0+
- **Memory**: 8GB+ unified RAM (16GB+ recommended)
- **Python**: 3.9+

## Comparison with Unsloth

| Feature | Unsloth (CUDA) | MLX-Tune |
|---------|----------------|----------|
| Platform | NVIDIA GPUs | Apple Silicon |
| Backend | Triton Kernels | MLX Framework |
| Memory | VRAM (limited) | Unified (up to 512GB) |
| API | Original | 100% Compatible |
| Best For | Production training | Local dev, large models |

## Known Limitations

### GGUF Export from Quantized Models

**The Issue**: GGUF export (`save_pretrained_gguf`) doesn't work directly with quantized (4-bit) base models. This is a [known limitation in mlx-lm](https://github.com/ml-explore/mlx-lm/issues/353), not an mlx-tune bug.

**What Works**:
- ✅ Training with quantized models (QLoRA) - works perfectly
- ✅ Saving adapters (`save_pretrained`) - works
- ✅ Saving merged model (`save_pretrained_merged`) - works
- ✅ Inference with trained model - works
- ❌ GGUF export from quantized base model - mlx-lm limitation

**Workarounds**:

1. **Use a non-quantized base model** (recommended for GGUF export):
   ```python
   # Use fp16 model instead of 4-bit
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="mlx-community/Llama-3.2-1B-Instruct",  # NOT -4bit
       max_seq_length=2048,
       load_in_4bit=False,  # Train in fp16
   )
   # Train normally, then export
   model.save_pretrained_gguf("model", tokenizer)  # Works!
   ```

2. **Dequantize during export** (results in large fp16 file):
   ```python
   model.save_pretrained_gguf("model", tokenizer, dequantize=True)
   # Then re-quantize with llama.cpp:
   # ./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
   ```

3. **Skip GGUF, use MLX format**: If you only need the model for MLX/Python inference, just use `save_pretrained_merged()` - no GGUF needed.

**Related Issues**:
- [mlx-lm #353](https://github.com/ml-explore/mlx-lm/issues/353) - MLX to GGUF conversion
- [mlx-examples #1382](https://github.com/ml-explore/mlx-examples/issues/1382) - Quantized to GGUF

## Contributing

Contributions welcome! Areas that need help:
- Custom MLX kernels for even faster training
- More comprehensive test coverage
- Documentation and examples
- Testing on different M-series chips (M1, M2, M3, M4, M5)
- VLM training improvements

## License

Apache 2.0 - See [LICENSE](LICENSE) file.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - The original, incredible CUDA library
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [MLX-LM](https://github.com/ml-explore/mlx-lm) - LLM utilities for MLX
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) - Vision model support

---

<p align="center">
  <strong>Community project, not affiliated with Unsloth AI or Apple.</strong><br>
  ⭐ Star this repo if you find it useful!
</p>
