"""
Example 30: Phi-3.5 MoE Fine-Tuning — Microsoft's Mixture of Experts

Fine-tune Phi-3.5-MoE-instruct, Microsoft's MoE model with 42B total
parameters (16 experts, top-2 routing, ~6.6B active per token).

This demonstrates MoE fine-tuning with a non-Qwen provider, showing that
MLX-Tune's MoE support is architecture-agnostic. PhiMoE uses a different
MoE structure (block_sparse_moe) than Qwen (mlp) — both are handled
automatically by the dynamic path resolver.

Requirements: ~22GB unified memory (4-bit quantized).
"""

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("MoE FINE-TUNING: Phi-3.5-MoE-instruct (42B total, ~6.6B active)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Phi-3.5-MoE
    # ========================================================================
    print("\n[Step 1] Loading Phi-3.5-MoE-instruct (4-bit)...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Phi-3.5-MoE-instruct-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA — MoE detected automatically
    # ========================================================================
    print("\n[Step 2] Applying LoRA (MoE-aware)...")

    # Same target_modules as any model — MLX-Tune auto-resolves:
    # gate_proj → block_sparse_moe.switch_mlp.gate_proj (expert layers)
    # q_proj → self_attn.q_proj (attention, same as dense)
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        # Required: Phi-3.5-MoE 41.9B-A6.6B 4-bit needs GC for activation memory.
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA adapters configured")

    # ========================================================================
    # Step 3: Prepare Dataset
    # ========================================================================
    print("\n[Step 3] Preparing dataset...")

    sample_data = [
        {"instruction": "What is Phi-3.5-MoE?",
         "output": "Phi-3.5-MoE is Microsoft's Mixture of Experts language model. It uses 16 experts with top-2 routing, giving it 42B total parameters but only about 6.6B active per token. It excels at reasoning, math, and code tasks."},
        {"instruction": "Compare MoE and dense model architectures.",
         "output": "Dense models activate all parameters for every token, while MoE models use a router to select a subset of expert sub-networks. This gives MoE models larger capacity with similar compute cost, enabling better scaling efficiency."},
        {"instruction": "What is the role of the gate/router in MoE?",
         "output": "The gate (or router) is a learned linear layer that computes scores for each expert given an input token. It selects the top-K scoring experts and produces weights for combining their outputs. The router is typically not fine-tuned."},
        {"instruction": "Explain SwitchLinear in MLX.",
         "output": "SwitchLinear is MLX's efficient implementation of MoE expert layers. It stacks all expert weights into a single tensor and uses gather_mm for batched expert-specific matrix multiplications, avoiding the overhead of iterating over individual experts."},
        {"instruction": "How does LoRASwitchLinear work?",
         "output": "LoRASwitchLinear extends SwitchLinear with per-expert LoRA adapters. It maintains expert-indexed low-rank matrices (lora_a and lora_b of shape [num_experts, ...]) and uses gather_mm to apply the correct adapter for each token-expert pair."},
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
            output_dir="outputs_phi_moe",
        ),
    )

    trainer.train()

    # ========================================================================
    # Step 5: Test Inference
    # ========================================================================
    print("\n[Step 5] Testing inference...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "Explain the benefits of MoE models in one paragraph."
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

    adapter_path = "outputs_phi_moe/adapters"
    model.save_pretrained(adapter_path)
    print(f"Adapters saved to {adapter_path}")

    from pathlib import Path
    adapter_dir = Path(adapter_path)
    for f in sorted(adapter_dir.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 70)
    print("Phi-3.5-MoE fine-tuning complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
