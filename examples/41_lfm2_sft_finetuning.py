"""
Example 41: LFM2 SFT Fine-Tuning - Liquid Foundation Models

Fine-tune Liquid AI's LFM2 models on Apple Silicon with MLX-Tune.
LFM2 uses a hybrid architecture: gated convolutions + grouped query attention.

Model: LFM2-2.6B (or LFM2.5-1.2B-Instruct)
Target Modules: q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3
Chat Format: ChatML (auto-detected)

NOTE: Requires downloading the model (~1.5-3GB for quantized versions).
"""

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("LFM2 SFT Fine-Tuning - Liquid Foundation Models")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load LFM2 Model
    # ========================================================================
    print("\n[Step 1] Loading LFM2 model...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/LFM2-350M-4bit",
        # Alternative: "lmstudio-community/LFM2.5-1.2B-Instruct-MLX-8bit"
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA with LFM2-Specific Target Modules
    # ========================================================================
    print("\n[Step 2] Applying LoRA adapters...")

    # LFM2 uses different module names than standard transformers:
    #   - out_proj/in_proj: attention projections (instead of o_proj)
    #   - w1, w2, w3: gated convolution MLP (instead of gate/up/down_proj)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",  # Attention
            "in_proj",                                   # Input projection
            "w1", "w2", "w3",                           # Gated conv MLP
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA configured with LFM2-specific target modules")

    # ========================================================================
    # Step 3: Prepare Dataset
    # ========================================================================
    print("\n[Step 3] Preparing dataset...")

    sample_data = [
        {
            "instruction": "What makes Liquid Foundation Models different from standard transformers?",
            "input": "",
            "output": "Liquid Foundation Models (LFM) use a hybrid architecture combining gated short convolutions with grouped query attention. This design achieves linear complexity for sequence processing while maintaining strong reasoning capabilities, making them efficient for edge deployment on devices like smartphones and laptops."
        },
        {
            "instruction": "Explain the concept of dynamical systems in the context of neural networks.",
            "input": "",
            "output": "Dynamical systems theory provides the foundation for LFM's gated convolution blocks. Instead of using attention for every layer, these blocks apply linear operations modulated by input-dependent gates, inspired by signal processing principles. This approach processes sequential information more efficiently while maintaining model quality."
        },
        {
            "instruction": "What are the advantages of on-device AI models?",
            "input": "",
            "output": "On-device AI models offer privacy (data stays on device), low latency (no network round-trip), offline capability, and reduced costs (no cloud inference fees). Models like LFM2 are specifically optimized for on-device deployment with efficient memory usage and fast inference on mobile and edge hardware."
        },
        {
            "instruction": "How does grouped query attention improve efficiency?",
            "input": "",
            "output": "Grouped query attention (GQA) shares key-value heads across multiple query heads, reducing the KV cache size and computational requirements. In LFM2, only 6 of 16 layers use GQA attention, with the remaining 10 using gated convolutions, creating an optimal balance of efficiency and capability."
        },
    ]

    def format_prompts(examples):
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

    dataset = Dataset.from_list(sample_data)
    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset prepared with {len(dataset)} examples")

    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print("\n[Step 4] Training...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir="outputs_lfm2_sft",
            weight_decay=0.01,
            lr_scheduler_type="linear",
        ),
    )

    trainer.train()

    # ========================================================================
    # Step 5: Inference
    # ========================================================================
    print("\n[Step 5] Testing inference...")

    FastLanguageModel.for_inference(model)

    from mlx_lm import generate

    prompt = "What is the benefit of hybrid SSM-attention architectures?"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model.model, tokenizer,
        prompt=formatted,
        max_tokens=100,
        verbose=False,
    )

    print(f"\nQ: {prompt}")
    print(f"A: {response}")

    # ========================================================================
    # Step 6: Save
    # ========================================================================
    print("\n[Step 6] Save options:")
    print("  model.save_pretrained('lfm2_lora')  # LoRA adapters")
    print("  model.save_pretrained_merged('lfm2_merged', tokenizer)  # Full model")

    print("\nLFM2 SFT fine-tuning complete!")


if __name__ == "__main__":
    main()
