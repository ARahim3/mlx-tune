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
        **kwargs: Additional arguments for pushing to hub

    Examples:
        >>> # Save locally in HF format
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

    # For MLX models, we need to use mlx_lm utilities to save
    # This will save in a format compatible with HuggingFace
    try:
        from mlx_lm.utils import save_model
        from mlx.utils import tree_unflatten

        # Get the underlying MLX model
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        # CRITICAL: Fuse LoRA adapters into base weights before saving
        # Without this, LoRA layers are saved as-is and won't load properly
        fused_linears = []
        for n, m in actual_model.named_modules():
            if not hasattr(m, "fuse"):
                continue
            try:
                fused = m.fuse(dequantize=kwargs.get('dequantize', False))
            except TypeError as exc:
                if "dequantize" not in str(exc):
                    raise
                fused = m.fuse()
            fused_linears.append((n, fused))

        if fused_linears:
            print(f"  Fusing {len(fused_linears)} LoRA layers into base model...")
            actual_model.update_modules(tree_unflatten(fused_linears))
        else:
            print("  No LoRA layers to fuse (saving base model as-is)")

        # mlx_lm.utils.save_model only takes (save_path, model) - tokenizer is saved separately
        save_model(str(output_dir), actual_model)

        # Save tokenizer separately
        tokenizer.save_pretrained(str(output_dir))

        # Save config.json if available (needed for loading and GGUF export)
        if hasattr(model, 'config') and model.config is not None:
            config_path = output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(model.config, f, indent=2)
        elif hasattr(model, 'model_path') and model.model_path:
            # Try to copy config from original model path
            src_config = Path(model.model_path) / "config.json"
            if src_config.exists():
                import shutil
                shutil.copy(src_config, output_dir / "config.json")

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


def export_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "q4_k_m",
    adapter_path: Optional[str] = None,
    **kwargs
):
    """
    Export model to GGUF format for use with llama.cpp, Ollama, etc.

    This function uses mlx_lm.fuse to merge adapters (if any) and export to GGUF.

    Args:
        model_path: Path to the base model or HuggingFace model ID
            (e.g., "mlx-community/Llama-3.2-1B-Instruct-4bit" or "./my_model")
        output_path: Path for output GGUF file (defaults to ./model.gguf)
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16, etc.)
            Note: mlx_lm exports in fp16 precision
        adapter_path: Path to LoRA adapters to fuse before export
        **kwargs: Additional export options:
            - dequantize: bool - Dequantize model before export (required for quantized models)

    Examples:
        >>> # Export base model to GGUF
        >>> export_to_gguf("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>>
        >>> # Export fine-tuned model with adapters
        >>> export_to_gguf(
        ...     "mlx-community/Llama-3.2-1B-Instruct-4bit",
        ...     adapter_path="./adapters",
        ...     output_path="my-model.gguf",
        ... )

    Note:
        GGUF export is only supported for Llama, Mistral, and Mixtral architectures.
        Quantized models need dequantize=True to export properly.
    """
    import subprocess

    # Determine if model_path is a HuggingFace model ID or local path
    # HF model IDs typically contain "/" but don't exist as local paths
    model_path_str = str(model_path)
    is_hf_model = (
        "/" in model_path_str and
        not Path(model_path_str).exists()
    )

    # Keep as string for HF models, convert to Path for local paths
    if not is_hf_model:
        model_path = Path(model_path_str)

    # Handle output path
    if output_path is None:
        output_path = Path("./model.gguf")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if model appears to be quantized (warn user about mlx_lm limitation)
    quantized_indicators = ['4bit', '8bit', '3bit', '2bit', '-q4', '-q8', 'int4', 'int8', 'bnb']
    model_name_lower = model_path_str.lower()
    is_likely_quantized = any(ind in model_name_lower for ind in quantized_indicators)

    if is_likely_quantized and not kwargs.get('dequantize', False):
        print("\n" + "=" * 70)
        print("⚠️  WARNING: Quantized model detected!")
        print("=" * 70)
        print(f"Model '{model_path}' appears to be quantized.")
        print("GGUF export from quantized models is NOT supported by mlx_lm.")
        print("This is an upstream limitation: https://github.com/ml-explore/mlx-lm/issues/353")
        print("\nOptions:")
        print("  1. Use dequantize=True (creates large fp16, re-quantize with llama.cpp)")
        print("  2. Use a non-quantized base model for training")
        print("  3. Use save_pretrained_merged() for MLX-only inference")
        print("=" * 70 + "\n")

    print(f"Exporting model to GGUF format...")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    if adapter_path:
        print(f"  Adapters: {adapter_path}")

    # Build mlx_lm.fuse command
    cmd = [
        "mlx_lm.fuse",
        "--model", str(model_path),
        "--export-gguf",
        "--gguf-path", str(output_path),
    ]

    # Add adapter path if provided
    if adapter_path:
        cmd.extend(["--adapter-path", str(adapter_path)])

    # Add dequantize flag for quantized models (required for proper GGUF export)
    if kwargs.get('dequantize', False) or kwargs.get('de_quantize', False):
        cmd.append("--dequantize")

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✓ Model exported to {output_path}")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"Error during GGUF export: {error_msg}")

        # Provide helpful error messages
        if "adapter_config.json" in error_msg.lower():
            print("\n⚠️  Adapter config not found. This usually means:")
            print("   1. The adapter path is missing adapter_config.json")
            print("   2. Training was done with an older version of mlx-tune")
            print(f"\n   To fix, either:")
            print(f"   a) Re-train with mlx-tune >= 0.3.4 (saves adapter_config.json)")
            print(f"   b) Export without adapters (base model only):")
            print(f"      model.save_pretrained_gguf('model', tokenizer)")
            if adapter_path:
                print(f"\n   Adapter path checked: {adapter_path}")
        elif "config.json" in error_msg.lower() or "FileNotFoundError" in str(e):
            print("\n⚠️  Config file not found. This usually means:")
            print("   1. The model path is incorrect")
            print("   2. The model hasn't been downloaded yet")
            print(f"\n   Try loading the model first with mlx_lm:")
            print(f"   python -c \"from mlx_lm import load; load('{model_path}')\"")
        elif "quantized" in error_msg.lower():
            print("\n⚠️  Quantized model detected. Try with dequantize=True:")
            print(f"   export_to_gguf('{model_path}', dequantize=True)")

        # Try alternative method using convert
        print("\nTrying alternative export method...")
        try:
            alt_cmd = [
                "mlx_lm.convert",
                "--hf-path", str(model_path),
                "-q",  # Quantize
                "--export-gguf",
            ]
            subprocess.run(alt_cmd, check=True)
            print(f"✓ Model exported using alternative method")
            return str(output_path)
        except Exception as alt_e:
            print(f"Alternative method also failed: {alt_e}")
            print("\nManual export command:")
            print(f"  mlx_lm.fuse --model {model_path} --export-gguf --gguf-path {output_path}")
            raise


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
