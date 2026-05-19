"""
Example 4: Simple Fine-Tuning with LoRA

This example demonstrates a complete fine-tuning workflow using MLX-Tune.
Note: This example shows the API setup. Actual training would require using
MLX's training utilities (mlx_lm.lora command or custom training loop).

For actual fine-tuning, you would typically use:
1. MLX-LM's command-line tool: `mlx_lm.lora --model ... --train --data ...`
2. Or implement a custom training loop with MLX

This example shows how to set up the model with Unsloth-compatible API.
"""

from mlx_tune import FastLanguageModel
import json


def create_sample_dataset():
    """Create a sample dataset for demonstration"""

    # Sample training data (chat format)
    train_data = [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is deep learning?"},
                {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is a neural network?"},
                {"role": "assistant", "content": "A neural network is a computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers."}
            ]
        },
    ]

    return train_data


def main():
    print("=" * 60)
    print("MLX-Tune Example: Fine-Tuning Setup")
    print("=" * 60)

    # Step 1: Load the base model
    print("\n1. Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print("✓ Model loaded!")

    # Step 2: Configure LoRA adapters
    print("\n2. Configuring LoRA adapters for parameter-efficient fine-tuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank - lower rank = fewer parameters
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,  # LoRA scaling factor
        lora_dropout=0.05,  # Dropout for regularization
        bias="none",
        # use_gradient_checkpointing defaults to False — opt in with "unsloth"
        # only when you OOM (e.g. ctx≥4096 or large model on tight-RAM Mac).
    )
    print("✓ LoRA configured!")
    print(f"   LoRA Rank: {model.lora_config['r']}")
    print(f"   Target Modules: {len(model.lora_config['target_modules'])} modules")

    # Step 3: Create sample dataset
    print("\n3. Creating sample training dataset...")
    train_data = create_sample_dataset()
    print(f"✓ Created dataset with {len(train_data)} examples")

    # Save dataset to file
    with open("sample_train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print("✓ Saved to sample_train.jsonl")

    # Step 4: Display training information
    print("\n4. Ready for Fine-Tuning!")
    print("   " + "-" * 56)
    print("   Model is now configured with LoRA adapters.")
    print("   ")
    print("   To actually fine-tune, you can:")
    print("   ")
    print("   A. Use MLX-LM command-line tool:")
    print("      $ mlx_lm.lora \\")
    print("          --model mlx-community/Llama-3.2-1B-Instruct-4bit \\")
    print("          --train \\")
    print("          --data sample_train.jsonl \\")
    print("          --iters 100 \\")
    print("          --learning-rate 1e-5 \\")
    print("          --lora-layers 16")
    print("   ")
    print("   B. Or implement custom training loop with MLX")
    print("   ")
    print("   After training, you can:")
    print("   - Load the fine-tuned adapters")
    print("   - Merge adapters with base model")
    print("   - Export to GGUF format")
    print("   - Upload to HuggingFace Hub")
    print("   " + "-" * 56)

    # Step 5: Test inference before training (baseline)
    print("\n5. Testing inference (baseline - before training)...")
    FastLanguageModel.for_inference(model)

    test_prompt = "What is machine learning?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    print(f"   Prompt: {test_prompt}")
    print("   Generating response...")

    from mlx_lm import generate
    response = generate(
        model.model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=100,
        verbose=False,
    )

    print(f"   Response: {response}")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Prepare your training dataset in JSONL format")
    print("2. Run fine-tuning using mlx_lm.lora command")
    print("3. Load the fine-tuned adapters")
    print("4. Test the improved model")
    print("5. Export and deploy!")


if __name__ == "__main__":
    main()
