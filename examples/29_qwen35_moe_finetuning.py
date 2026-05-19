"""
Example 29: Qwen3.5 MoE Fine-Tuning — Mixture of Experts on Apple Silicon

Fine-tune Qwen3.5-35B-A3B, a Mixture of Experts model with 35B total
parameters but only 3B active per token. MLX-Tune automatically detects
MoE architectures and applies LoRA to expert layers via LoRASwitchLinear.

This is the SAME API as Unsloth — just change the import!

ORIGINAL UNSLOTH CODE (CUDA):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen3-30B-A3B")

MLX-TUNE CODE (Apple Silicon):
    from mlx_tune import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("mlx-community/Qwen3.5-35B-A3B-4bit")

Requirements: ~20GB unified memory for 4-bit model + LoRA training.
"""

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("MoE FINE-TUNING: Qwen3.5-35B-A3B (35B total, 3B active)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load MoE Model
    # ========================================================================
    print("\n[Step 1] Loading Qwen3.5-35B-A3B MoE model (4-bit)...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Qwen3.5-35B-A3B-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA — MoE expert layers detected automatically!
    # ========================================================================
    print("\n[Step 2] Applying LoRA (MoE-aware)...")

    # Same target_modules as any Unsloth model — MLX-Tune auto-resolves
    # gate_proj/up_proj/down_proj → mlp.switch_mlp.* (expert layers)
    #                              + mlp.shared_expert.* (shared expert)
    #                              + mlp.gate_proj/... (dense layers if mixed)
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # Required: Qwen3.5-35B-A3B-4bit barely fits without GC even at ctx=2048.
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA adapters configured")

    # ========================================================================
    # Step 3: Prepare Dataset
    # ========================================================================
    print("\n[Step 3] Preparing dataset...")

    sample_data = [
        {"instruction": "What is a mixture of experts model?",
         "output": "A Mixture of Experts (MoE) model uses multiple specialized sub-networks (experts) with a router that selects which experts process each token, enabling large model capacity with efficient computation."},
        {"instruction": "Explain sparse activation in neural networks.",
         "output": "Sparse activation means only a subset of neurons or parameters are active for each input. In MoE models, the router selects top-K experts per token, so most expert weights are inactive, reducing compute while maintaining capacity."},
        {"instruction": "What are the benefits of MoE models?",
         "output": "MoE models offer better scaling efficiency: they can have many more parameters than dense models while using similar compute per token. This enables larger knowledge capacity without proportional increase in training or inference cost."},
        {"instruction": "How does LoRA work with MoE models?",
         "output": "LoRA for MoE applies low-rank adaptation to each expert's projection layers independently. MLX uses LoRASwitchLinear which maintains per-expert LoRA matrices and uses gather_mm for efficient expert-specific updates."},
        {"instruction": "Compare Qwen3.5-35B-A3B to a dense 3B model.",
         "output": "Qwen3.5-35B-A3B has 35B total parameters but only activates 3B per token via top-K expert routing. This gives it much more knowledge capacity than a dense 3B model while having similar inference speed."},
    ]

    def format_prompts(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(sample_data)
    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset prepared: {len(dataset)} examples")

    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print("\n[Step 4] Training...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-5,
            logging_steps=1,
            output_dir="outputs_moe",
        ),
    )

    trainer.train()

    # ========================================================================
    # Step 5: Test Inference
    # ========================================================================
    print("\n[Step 5] Testing inference...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "What makes MoE models efficient?"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model.model, tokenizer, prompt=formatted, max_tokens=100, verbose=False)
    print(f"Q: {prompt}")
    print(f"A: {response}")

    # ========================================================================
    # Step 6: Save Adapters
    # ========================================================================
    print("\n[Step 6] Saving adapters...")
    model.save_pretrained("outputs_moe/saved_adapters")
    print("Adapters saved to outputs_moe/saved_adapters")

    print("\n" + "=" * 70)
    print("MoE fine-tuning complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
