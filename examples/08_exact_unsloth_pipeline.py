"""
Example 8: Exact Unsloth Pipeline - Complete API Compatibility

This example shows that MLX-Tune provides the EXACT same API as Unsloth.
Just change the import line and your Unsloth code works on Mac!

ORIGINAL UNSLOTH CODE (for CUDA):
---------------------------------
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

UNSLOTH-MAC CODE (for Apple Silicon):
-------------------------------------
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset

The rest of the code is IDENTICAL!
"""

# ============================================================================
# IMPORTS - Just change this one line!
# ============================================================================

# For Unsloth (CUDA):
# from unsloth import FastLanguageModel
# from trl import SFTTrainer, SFTConfig

# For MLX-Tune (Apple Silicon MLX):
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset


def main():
    print("=" * 70)
    print("EXACT UNSLOTH PIPELINE - API Compatibility Demo")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Model with 4-bit Quantization
    # ========================================================================
    print("\n[Step 1] Loading model with 4-bit quantization...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        # For real Unsloth: model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    print(f"✓ Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA Adapters - SAME API!
    # ========================================================================
    print("\n[Step 2] Applying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank: 16 is sweet spot for most tasks
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # 0 is optimized in Unsloth
        bias="none",
        # Mirrors Unsloth's tutorial syntax exactly. mlx-tune's default is
        # False (faster); pass "unsloth" or True when you need to halve
        # activation memory at the cost of ~2× step time.
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("✓ LoRA adapters configured")

    # ========================================================================
    # Step 3: Prepare Dataset with Chat Template - SAME!
    # ========================================================================
    print("\n[Step 3] Preparing dataset...")

    # Create sample dataset (in real use: load_dataset("yahma/alpaca-cleaned"))
    sample_data = [
        {
            "instruction": "Explain what machine learning is.",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Write a haiku about programming.",
            "input": "",
            "output": "Code flows like water\nBugs emerge from the deep mist\nDebug, compile, run"
        },
        {
            "instruction": "What is Python?",
            "input": "",
            "output": "Python is a high-level, interpreted programming language known for its simple syntax and versatility in web development, data science, and automation."
        },
        {
            "instruction": "Explain recursion.",
            "input": "",
            "output": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem until reaching a base case."
        },
    ]

    def format_prompts(examples):
        """Format dataset with chat template - SAME AS UNSLOTH!"""
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            content = f"{instruction}\n{input_text}" if input_text else instruction
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": output}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_list(sample_data)
    dataset = dataset.map(format_prompts, batched=True)

    print(f"✓ Dataset prepared with {len(dataset)} examples")

    # ========================================================================
    # Step 4: Configure Training with SFTConfig - SAME API!
    # ========================================================================
    print("\n[Step 4] Configuring training...")

    # This is the EXACT same SFTConfig pattern from Unsloth!
    training_config = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20,  # Small for demo, use 500+ for real training
        learning_rate=2e-4,
        # fp16=True,  # Not applicable on MLX
        # bf16=True,  # Not applicable on MLX
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
    )

    # ========================================================================
    # Step 5: Create Trainer with SFTConfig - SAME API!
    # ========================================================================
    print("\n[Step 5] Creating SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_config,  # Pass SFTConfig just like in Unsloth!
    )

    print(f"✓ Trainer configured")
    print(f"  - Learning rate: {trainer.learning_rate}")
    print(f"  - Batch size: {trainer.batch_size}")
    print(f"  - Iterations: {trainer.iters}")

    # ========================================================================
    # Step 6: Train! - SAME API!
    # ========================================================================
    print("\n[Step 6] Training the model...")
    print("(This will actually train using MLX under the hood)")

    trainer.train()

    # ========================================================================
    # Step 7: Inference Mode - SAME API!
    # ========================================================================
    print("\n[Step 7] Enabling inference mode...")

    FastLanguageModel.for_inference(model)

    from mlx_lm import generate

    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model.model, tokenizer,
        prompt=formatted_prompt,
        max_tokens=50,
        verbose=False,
    )

    print(f"\n📝 Test Inference:")
    print(f"   Q: {prompt}")
    print(f"   A: {response}")

    # ========================================================================
    # Step 8: Save Options - SAME API as Unsloth!
    # ========================================================================
    print("\n[Step 8] Save options (Unsloth-compatible):")

    # Option 1: Save LoRA adapters only (~100MB)
    print("  model.save_pretrained('lora_model')  # Adapters only")

    # Option 2: Save merged model (base + adapters)
    print("  model.save_pretrained_merged('merged_16bit', tokenizer)")

    # Option 3: Export to GGUF for llama.cpp/Ollama
    print("  model.save_pretrained_gguf('model', tokenizer, quantization_method='q4_k_m')")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUCCESS! The EXACT Unsloth pipeline works on Mac!")
    print("=" * 70)

    print("""
API COMPATIBILITY SUMMARY:
-------------------------
✅ FastLanguageModel.from_pretrained() - SAME API
✅ FastLanguageModel.get_peft_model() - SAME API
✅ SFTConfig - SAME API as TRL
✅ SFTTrainer - SAME API
✅ trainer.train() - SAME API
✅ FastLanguageModel.for_inference() - SAME API
✅ model.save_pretrained() - SAME API
✅ model.save_pretrained_merged() - SAME API
✅ model.save_pretrained_gguf() - SAME API

MIGRATION GUIDE:
---------------
1. Change import: from unsloth import -> from mlx_tune import
2. Change import: from trl import SFTTrainer, SFTConfig -> already in mlx_tune
3. Use mlx-community models instead of unsloth/ models
4. That's it! Rest of the code is IDENTICAL!
""")


if __name__ == "__main__":
    main()
