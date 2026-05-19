"""
Example 2: LoRA Configuration

This example demonstrates how to add LoRA adapters to a model,
showing compatibility with Unsloth's get_peft_model API.
"""

from mlx_tune import FastLanguageModel

def main():
    print("=" * 60)
    print("MLX-Tune Example: LoRA Configuration")
    print("=" * 60)

    # Load a model
    print("\n1. Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print("✓ Model loaded!")

    # Add LoRA adapters
    print("\n2. Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        # use_gradient_checkpointing defaults to False (fast).
        # Pass "unsloth" or True if you OOM at long ctx or with a large model.
        random_state=3407,
    )
    print("✓ LoRA adapters configured!")

    # Check LoRA configuration
    print("\n3. LoRA Configuration:")
    if hasattr(model, 'lora_config') and model.lora_config:
        print(f"   LoRA Rank: {model.lora_config['r']}")
        print(f"   LoRA Alpha: {model.lora_config['lora_alpha']}")
        print(f"   LoRA Dropout: {model.lora_config['lora_dropout']}")
        print(f"   Target Modules: {model.lora_config['target_modules']}")
        print(f"   Bias: {model.lora_config['bias']}")
        print(f"   LoRA Enabled: {model.lora_enabled}")
    else:
        print("   LoRA configuration not found")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Ready for fine-tuning!")
    print("=" * 60)


if __name__ == "__main__":
    main()
