"""
Training utilities for MLX-Tune

Provides helper functions for loading datasets, training models, and saving results
in standard HuggingFace format.
"""

from typing import Optional, Dict, Any, Union, List
from datasets import load_dataset
import json
from pathlib import Path


def prepare_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: str = "train",
    formatting_func: Optional[callable] = None,
    **kwargs
):
    """
    Load and prepare dataset for fine-tuning.

    This function provides Unsloth-compatible dataset loading with support for
    HuggingFace datasets library.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "timdettmers/openassistant-guanaco")
        dataset_path: Local path to dataset (JSONL or JSON)
        split: Dataset split to load ("train", "test", "validation")
        formatting_func: Function to format dataset samples
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Loaded dataset object

    Examples:
        >>> # Load from HuggingFace Hub
        >>> dataset = prepare_dataset("timdettmers/openassistant-guanaco")
        >>>
        >>> # Load from local file
        >>> dataset = prepare_dataset(dataset_path="data/train.jsonl")
        >>>
        >>> # Load with custom split
        >>> dataset = prepare_dataset(
        ...     "yahma/alpaca-cleaned",
        ...     split="train[:1000]"
        ... )
    """

    if dataset_name:
        # Load from HuggingFace Hub
        print(f"Loading dataset '{dataset_name}' from HuggingFace Hub...")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        print(f"✓ Loaded {len(dataset)} examples")
        return dataset

    elif dataset_path:
        # Load from local file
        dataset_path = Path(dataset_path)
        print(f"Loading dataset from '{dataset_path}'...")

        if dataset_path.suffix == '.jsonl':
            # Load JSONL file
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        elif dataset_path.suffix == '.json':
            # Load JSON file
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        print(f"✓ Loaded {len(dataset)} examples")
        return dataset

    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")


def format_chat_template(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    """
    Format messages using the model's chat template.

    This function provides Unsloth-compatible chat template formatting, supporting
    different LLM formats (Llama, Mistral, Qwen, etc.).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer with chat template support
        add_generation_prompt: Whether to add generation prompt at the end

    Returns:
        Formatted prompt string

    Examples:
        >>> messages = [
        ...     {"role": "user", "content": "What is AI?"},
        ...     {"role": "assistant", "content": "AI stands for..."}
        ... ]
        >>> prompt = format_chat_template(messages, tokenizer)
    """

    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False
        )
    else:
        # Fallback to simple formatting if no chat template
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
            elif role == 'system':
                formatted += f"System: {content}\n"
        if add_generation_prompt:
            formatted += "Assistant: "
        return formatted


def create_training_data(
    dataset: Any,
    tokenizer: Any,
    output_path: str,
    format_type: str = "chat",
    text_field: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> str:
    """
    Create training data file in MLX-LM compatible format.

    Args:
        dataset: Dataset object (from HuggingFace datasets)
        tokenizer: Tokenizer for chat template formatting
        output_path: Path to save formatted data (JSONL format)
        format_type: Data format ("chat", "text", "completions")
        text_field: Field name containing text (for "text" format)
        max_samples: Maximum number of samples to process

    Returns:
        Path to created training data file

    Examples:
        >>> dataset = load_dataset("timdettmers/openassistant-guanaco")
        >>> create_training_data(
        ...     dataset,
        ...     tokenizer,
        ...     "train.jsonl",
        ...     format_type="chat"
        ... )
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples_written = 0
    with open(output_path, 'w') as f:
        for idx, sample in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break

            # Format based on type
            if format_type == "chat":
                # Expect 'messages' field or format it
                if 'messages' in sample:
                    formatted_sample = {"messages": sample['messages']}
                elif 'conversations' in sample:
                    formatted_sample = {"messages": sample['conversations']}
                else:
                    # Try to construct messages from text field
                    continue

            elif format_type == "text":
                # Simple text format
                if text_field and text_field in sample:
                    formatted_sample = {"text": sample[text_field]}
                elif 'text' in sample:
                    formatted_sample = {"text": sample['text']}
                else:
                    continue

            elif format_type == "completions":
                # Prompt-completion format
                if 'prompt' in sample and 'completion' in sample:
                    formatted_sample = {
                        "prompt": sample['prompt'],
                        "completion": sample['completion']
                    }
                else:
                    continue

            else:
                raise ValueError(f"Unsupported format_type: {format_type}")

            f.write(json.dumps(formatted_sample) + '\n')
            samples_written += 1

    print(f"✓ Created training data: {output_path} ({samples_written} samples)")
    return str(output_path)


def save_model_hf_format(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    save_method: str = "merged_16bit",
    **kwargs
):
    """
    Save fine-tuned model in standard HuggingFace format.

    This saves the model so that anyone can use it with transformers library,
    not just MLX. Essential for sharing your fine-tuned models!

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir: Directory to save model
        push_to_hub: Whether to upload to HuggingFace Hub
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        save_method: Unsloth-compatible merge mode. ``"merged_16bit"`` (default)
            dequantizes a quantized base and saves a full-precision merged model
            so the fine-tune is preserved exactly. ``"merged_4bit"`` keeps the
            base quantization and re-quantizes the fused weights (smaller on disk,
            but small LoRA deltas can be rounded away — see note below). Pass an
            explicit ``dequantize=`` to override the mode-derived default.
        **kwargs: Additional arguments (``dequantize``, hub upload options).

    Note:
        Merging a LoRA adapter into a quantized base and re-quantizing
        (``merged_4bit``) rounds the merged weights back onto the low-bit grid.
        When the fine-tune is weak (low LR / few steps) the delta can be smaller
        than the quantization step and effectively disappear, so the reloaded
        model behaves like the base. ``merged_16bit`` avoids this by saving
        full-precision weights; alternatively keep the adapter separate and use
        ``model.load_adapter(...)`` at inference time.

    Examples:
        >>> # Save locally in HF format (16-bit merge — fine-tune preserved)
        >>> save_model_hf_format(model, tokenizer, "my-finetuned-model")
        >>>
        >>> # Save and push to HuggingFace Hub
        >>> save_model_hf_format(
        ...     model, tokenizer,
        ...     "my-finetuned-model",
        ...     push_to_hub=True,
        ...     repo_id="username/my-model"
        ... )
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")

    # Resolve whether to dequantize. An explicit ``dequantize=`` always wins;
    # otherwise derive it from the Unsloth-style save_method:
    #   merged_16bit -> dequantize=True  (full-precision merge, no rounding loss)
    #   merged_4bit  -> dequantize=False (keep base quant, re-quantize fused weights)
    if "dequantize" in kwargs:
        dequantize = bool(kwargs.pop("dequantize"))
    else:
        dequantize = save_method != "merged_4bit"

    # For MLX models, we need to use mlx_lm utilities to save
    # This will save in a format compatible with HuggingFace
    try:
        from mlx_lm.utils import save_model, save_config, dequantize_model
        from mlx.utils import tree_unflatten

        # Get the underlying MLX model
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        # CRITICAL: Fuse LoRA adapters into base weights before saving.
        # Without this, LoRA layers are saved as-is and won't load properly.
        # When dequantize=True, fuse() returns plain nn.Linear for the fused
        # (LoRA-wrapped) layers; the *non*-LoRA quantized layers are converted
        # separately by dequantize_model() below.
        fused_linears = [
            (n, m.fuse(dequantize=dequantize))
            for n, m in actual_model.named_modules()
            if hasattr(m, "fuse")
        ]

        if fused_linears:
            print(f"  Fusing {len(fused_linears)} LoRA layers into base model "
                  f"({'16-bit / dequantized' if dequantize else '4-bit / re-quantized'})...")
            actual_model.update_modules(tree_unflatten(fused_linears))
        else:
            print("  No LoRA layers to fuse (saving base model as-is)")

        # Resolve the config dict (carries the `quantization` field for a
        # quantized base). Prefer the in-memory config captured at load time.
        config = None
        if getattr(model, 'config', None):
            config = dict(model.config)
        else:
            src = getattr(model, 'model_path', None) or getattr(model, 'model_name', None)
            if src and Path(src).exists():
                src_config = Path(src) / "config.json"
                if src_config.exists():
                    with open(src_config) as f:
                        config = json.load(f)

        if dequantize:
            # Convert any remaining quantized layers (embeddings, untargeted
            # linears, lm_head) to full precision, and drop the quantization
            # metadata so the fp16 weights reload cleanly. Mirrors mlx_lm.fuse.
            actual_model = dequantize_model(actual_model)
            if config is not None:
                config.pop("quantization", None)
                config.pop("quantization_config", None)

        # Save weights. donate_model=False keeps the in-memory model usable
        # after the call (e.g. for follow-up inference).
        save_model(str(output_dir), actual_model, donate_model=False)

        # Save tokenizer separately
        tokenizer.save_pretrained(str(output_dir))

        # Save config.json (needed for loading and GGUF export)
        if config is not None:
            save_config(config, config_path=output_dir / "config.json")

        print(f"✓ Model saved to {output_dir}")

        if push_to_hub and repo_id:
            print(f"Uploading to HuggingFace Hub: {repo_id}")
            from mlx_lm.utils import upload_to_hub
            upload_to_hub(str(output_dir), repo_id, **kwargs)
            print(f"✓ Model uploaded to {repo_id}")

    except ImportError:
        print("Warning: mlx_lm.utils not available. Attempting alternative save method...")
        # Alternative: save tokenizer at minimum
        tokenizer.save_pretrained(str(output_dir))
        print(f"✓ Tokenizer saved to {output_dir}")


# ---------------------------------------------------------------------------
# GGUF export via llama.cpp
# ---------------------------------------------------------------------------

# Quantization types accepted by ``llama-quantize`` that we expose. "f16",
# "bf16" and "f32" are produced directly by the converter (no quantize step).
_GGUF_UNQUANTIZED_TYPES = {"f16", "bf16", "f32"}
_GGUF_QUANT_TYPES = {
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_xs", "iq3_s",
    "iq3_m", "iq4_nl", "iq4_xs", "tq1_0", "tq2_0",
}
# Unsloth-compatible aliases
_GGUF_ALIASES = {"not_quantized": "f16", "fast_quantized": "q8_0", "quantized": "q4_k_m"}

_LLAMA_CPP_SETUP_HELP = """\
GGUF export requires llama.cpp (the converter script plus the llama-quantize binary).

  1. Get llama.cpp:
       git clone https://github.com/ggml-org/llama.cpp
       brew install llama.cpp          # provides llama-quantize
     (or build llama-quantize yourself: cmake -B build && cmake --build build --target llama-quantize)

  2. Give the converter a Python with its dependencies:
       pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

  3. Point mlx-tune at it (any of):
       export LLAMA_CPP_PATH=/path/to/llama.cpp
       model.save_pretrained_gguf(..., llama_cpp_path="/path/to/llama.cpp")
     Optionally, if the converter's deps live in a different environment:
       export LLAMA_CPP_PYTHON=/path/to/python
"""


class LlamaCppNotFoundError(RuntimeError):
    """Raised when the llama.cpp toolchain needed for GGUF export is missing."""


def _normalize_gguf_quantization(quantization: Optional[str], qat: bool = False) -> str:
    """Resolve a user-supplied GGUF quantization name to a llama.cpp type."""
    if qat:
        # QAT checkpoints (e.g. Gemma QAT) are trained against Q4_0's scale
        # grid; any other scheme re-rounds the weights and discards that.
        if quantization and str(quantization).lower() not in ("q4_0", "quantized", "q4_k_m"):
            print(f"  Note: qat=True overrides quantization='{quantization}' -> 'q4_0'")
        return "q4_0"
    q = (quantization or "q4_k_m").lower()
    q = _GGUF_ALIASES.get(q, q)
    if q not in _GGUF_QUANT_TYPES and q not in _GGUF_UNQUANTIZED_TYPES:
        supported = ", ".join(sorted(_GGUF_QUANT_TYPES | _GGUF_UNQUANTIZED_TYPES))
        raise ValueError(f"Unsupported GGUF quantization '{quantization}'. Supported: {supported}")
    return q


def _find_llama_cpp(llama_cpp_path: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Locate ``convert_hf_to_gguf.py`` and ``llama-quantize``.

    Search order: explicit ``llama_cpp_path`` arg, ``$LLAMA_CPP_PATH``,
    ``./llama.cpp``, ``~/llama.cpp``. ``llama-quantize`` is additionally
    looked up on ``$PATH`` (e.g. Homebrew's llama.cpp).
    """
    import os
    import shutil

    candidates = []
    for p in (llama_cpp_path, os.environ.get("LLAMA_CPP_PATH")):
        if p:
            candidates.append(Path(p).expanduser())
    candidates += [Path.cwd() / "llama.cpp", Path.home() / "llama.cpp"]

    convert_script = None
    quantize_bin = None
    for root in candidates:
        if convert_script is None and (root / "convert_hf_to_gguf.py").is_file():
            convert_script = str(root / "convert_hf_to_gguf.py")
        if quantize_bin is None:
            for rel in ("build/bin/llama-quantize", "llama-quantize", "bin/llama-quantize"):
                if (root / rel).is_file() and os.access(root / rel, os.X_OK):
                    quantize_bin = str(root / rel)
                    break
    if quantize_bin is None:
        quantize_bin = shutil.which("llama-quantize")

    return {"convert_script": convert_script, "quantize_bin": quantize_bin}


def _copy_tokenizer_assets(model_path: Union[str, Path], out_dir: Path) -> None:
    """Copy tokenizer files HF tokenizers don't re-emit but llama.cpp may need.

    Only files that exist in the *source* checkpoint are copied, so a model that
    ships just ``tokenizer.json`` (e.g. Gemma 4) is converted from that.
    """
    import shutil
    from mlx_lm.utils import _download

    src = Path(model_path) if Path(model_path).exists() else _download(str(model_path))
    for name in ("tokenizer.model", "tokenizer.model.v3", "tekken.json"):
        if (src / name).is_file() and not (out_dir / name).exists():
            shutil.copy2(src / name, out_dir / name)


def _release_mlx_memory() -> None:
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _merge_for_gguf(model_path: Union[str, Path], adapter_path: Optional[str], out_dir: Path) -> Path:
    """Load base (+ adapters) from disk, fuse, dequantize, save full-precision HF weights.

    Used by the standalone ``export_to_gguf(model_path, adapter_path=...)``. When a
    model is already in memory, ``save_model_hf_format`` is used instead so the
    weights are not loaded twice.
    """
    from mlx_lm import load as mlx_load
    from mlx_lm.utils import save_model, save_config, dequantize_model
    from mlx.utils import tree_unflatten

    print(f"  Loading {model_path}" + (f" with adapters from {adapter_path}" if adapter_path else ""))
    model, tokenizer, config = mlx_load(str(model_path), adapter_path=adapter_path, return_config=True)

    fused = [(n, m.fuse(dequantize=True)) for n, m in model.named_modules() if hasattr(m, "fuse")]
    if fused:
        print(f"  Fusing {len(fused)} LoRA layers (dequantized)")
        model.update_modules(tree_unflatten(fused))
    model = dequantize_model(model)
    config = dict(config)
    config.pop("quantization", None)
    config.pop("quantization_config", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(str(out_dir), model, donate_model=True)
    save_config(config, config_path=out_dir / "config.json")
    tokenizer.save_pretrained(str(out_dir))
    del model
    _copy_tokenizer_assets(model_path, out_dir)
    return out_dir


def export_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "q4_k_m",
    adapter_path: Optional[str] = None,
    qat: bool = False,
    llama_cpp_path: Optional[str] = None,
    **kwargs
):
    """
    Export a model (optionally with LoRA adapters) to GGUF for llama.cpp, Ollama, LM Studio.

    Pipeline (same approach as Unsloth):
      1. Load the base model with mlx-lm, fuse adapters, and dequantize to full
         precision (works for 4-bit/8-bit bases).
      2. Convert to GGUF with llama.cpp's ``convert_hf_to_gguf.py`` (supports every
         architecture llama.cpp supports: Llama, Gemma, Qwen, Phi, Mistral, ...).
      3. Quantize with ``llama-quantize`` to the requested type.

    Args:
        model_path: Base model path or HuggingFace ID (quantized or not).
        output_path: Output ``.gguf`` path (defaults to ./model.gguf).
        quantization: llama.cpp type: q4_k_m (default), q4_0, q5_k_m, q6_k, q8_0,
            f16, bf16, ... Unsloth aliases (``quantized``, ``fast_quantized``,
            ``not_quantized``) are accepted.
        adapter_path: LoRA adapter directory to fuse before export.
        qat: Force strict ``q4_0``, matching the scale grid of Quantization-Aware
            Training checkpoints (e.g. Gemma QAT). Token embeddings keep
            llama.cpp's default higher-precision type, as in Google's own QAT GGUFs.
        llama_cpp_path: Path to a llama.cpp checkout (else ``$LLAMA_CPP_PATH``,
            ``./llama.cpp``, ``~/llama.cpp``).
        **kwargs:
            - llama_cpp_python: Python used to run the converter (else
              ``$LLAMA_CPP_PYTHON``, else the current interpreter).
            - keep_intermediates: keep merged weights / f16 GGUF (default False).
            - model / tokenizer: an in-memory (LoRA-applied) model to merge instead
              of reloading ``model_path`` from disk; used by ``save_pretrained_gguf``
              so the weights are only held once. The LoRA layers are fused into this
              model in place (same as ``save_pretrained_merged``).
            - dequantize: accepted for backward compatibility (always dequantizes).

    Raises:
        LlamaCppNotFoundError: llama.cpp tools are not installed/located.

    Examples:
        >>> export_to_gguf("mlx-community/gemma-2-2b-it-4bit", adapter_path="./adapters",
        ...                output_path="model-q4_k_m.gguf")
        >>> export_to_gguf("./gemma-qat-unquantized", adapter_path="./adapters", qat=True)
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    quant = _normalize_gguf_quantization(quantization, qat=qat)
    output_path = Path(output_path) if output_path else Path("./model.gguf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tools = _find_llama_cpp(llama_cpp_path)
    missing = []
    if tools["convert_script"] is None:
        missing.append("convert_hf_to_gguf.py (llama.cpp checkout)")
    if quant not in _GGUF_UNQUANTIZED_TYPES and tools["quantize_bin"] is None:
        missing.append("llama-quantize binary")
    if missing:
        raise LlamaCppNotFoundError(
            "Could not find: " + ", ".join(missing) + "\n\n" + _LLAMA_CPP_SETUP_HELP
        )

    converter_python = kwargs.get("llama_cpp_python") or os.environ.get("LLAMA_CPP_PYTHON") or sys.executable
    keep = bool(kwargs.get("keep_intermediates", False))

    print("Exporting model to GGUF (llama.cpp)...")
    print(f"  Model: {model_path}")
    if adapter_path:
        print(f"  Adapters: {adapter_path}")
    print(f"  Quantization: {quant}" + (" (QAT)" if qat else ""))
    print(f"  Output: {output_path}")

    in_memory_model = kwargs.get("model")
    work = Path(tempfile.mkdtemp(prefix="mlx_tune_gguf_", dir=str(output_path.parent)))
    try:
        print("[1/3] Fusing adapters and dequantizing to full precision...")
        if in_memory_model is not None:
            merged_dir = work / "merged"
            save_model_hf_format(in_memory_model, kwargs.get("tokenizer"), str(merged_dir),
                                 save_method="merged_16bit")
            _copy_tokenizer_assets(model_path, merged_dir)
        else:
            merged_dir = _merge_for_gguf(model_path, adapter_path, work / "merged")
        _release_mlx_memory()

        converted = output_path if quant in _GGUF_UNQUANTIZED_TYPES else work / "model-f16.gguf"
        print(f"[2/3] Converting to GGUF ({'final ' + quant if converted == output_path else 'f16 intermediate'})...")
        conv_cmd = [converter_python, tools["convert_script"], str(merged_dir),
                    "--outfile", str(converted),
                    "--outtype", quant if converted == output_path else "f16"]
        result = subprocess.run(conv_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
            hint = ""
            if "No module named" in tail:
                hint = ("\nThe converter's Python is missing dependencies. Install them with\n"
                        "  pip install -r <llama.cpp>/requirements/requirements-convert_hf_to_gguf.txt\n"
                        "or set LLAMA_CPP_PYTHON to an interpreter that has them.")
            raise RuntimeError(f"convert_hf_to_gguf.py failed (exit {result.returncode}):\n{tail}{hint}")

        if converted != output_path:
            print(f"[3/3] Quantizing to {quant.upper()} with llama-quantize...")
            q_cmd = [tools["quantize_bin"], str(converted), str(output_path), quant.upper()]
            result = subprocess.run(q_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
                raise RuntimeError(f"llama-quantize failed (exit {result.returncode}):\n{tail}")

        size_mb = output_path.stat().st_size / 1e6
        print(f"✓ Model exported to {output_path} ({size_mb:.0f} MB, {quant})")
        return str(output_path)
    finally:
        if keep:
            print(f"  Intermediates kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def get_training_config(
    output_dir: str = "./lora_finetuned",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    **kwargs
) -> Dict[str, Any]:
    """
    Get recommended training configuration.

    Returns a configuration dict compatible with MLX-LM training.

    Args:
        output_dir: Directory to save trained model
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        **kwargs: Additional training arguments

    Returns:
        Training configuration dict

    Examples:
        >>> config = get_training_config(
        ...     num_train_epochs=5,
        ...     learning_rate=1e-4
        ... )
    """

    config = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": kwargs.get("lora_dropout", 0.05),
        "warmup_steps": kwargs.get("warmup_steps", 100),
        "max_seq_length": kwargs.get("max_seq_length", 2048),
        "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
        "save_steps": kwargs.get("save_steps", 500),
        "logging_steps": kwargs.get("logging_steps", 10),
    }

    config.update(kwargs)
    return config
