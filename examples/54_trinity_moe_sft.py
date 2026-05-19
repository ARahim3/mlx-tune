"""
Example 54: Arcee Trinity-Nano (AFMoE) — Instruction SFT on FineTome-100k

Fine-tune Arcee's Trinity-Nano-Preview, a 6B-total / 1B-active Mixture-of-Experts
model using the AFMoE architecture (128 experts + 1 shared expert, sigmoid
routing, gated attention, interleaved sliding/global attention).

Dataset: mlabonne/FineTome-100k — a high-quality ShareGPT-format instruction
mix that Unsloth popularized for chat fine-tunes. We take a small slice here
for a quick demo — swap `SLICE` for the full split when you're ready to run
the real recipe.

mlx-tune automatically:
  - detects the MoE architecture (QuantizedSwitchLinear experts)
  - applies per-expert LoRA via mlx-lm's LoRASwitchLinear
  - also covers the shared expert and gated-attention gate_proj

After training this example:
  1. saves the adapter
  2. reloads it on a fresh base model (round-trip verification)
  3. merges the adapter into the base weights via save_pretrained_merged
  4. runs a final generation sample

Model: mlx-community/Trinity-Nano-Preview-4bit (~3.3 GB)

Requires ~8-10 GB unified memory for 4-bit model + LoRA training.
"""

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import load_dataset


MODEL_ID = "mlx-community/Trinity-Nano-Preview-4bit"
DATASET_ID = "mlabonne/FineTome-100k"
SLICE = "train[:200]"   # tiny slice for a fast demo; use "train" for the real thing


def _sharegpt_to_chatml(example):
    """FineTome stores conversations as [{'from': 'human'/'gpt', 'value': ...}, ...]."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = []
    for turn in example["conversations"]:
        role = role_map.get(turn.get("from", "").lower())
        if role is None:
            continue
        msgs.append({"role": role, "content": turn.get("value", "")})
    return msgs


def main():
    print("=" * 70)
    print("Trinity-Nano SFT — AFMoE on FineTome-100k")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer
    # ------------------------------------------------------------------
    print("\n[1] Loading Trinity-Nano-Preview (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # ------------------------------------------------------------------
    # 2. Apply MoE-aware LoRA
    #    _resolve_target_modules() walks the model and returns every path
    #    ending in each short name — so gate_proj/up_proj/down_proj hit
    #    both mlp.experts.* (SwitchLinear) and mlp.shared_experts.*.
    #    The attention's gate_proj (gated-attention) is also included.
    # ------------------------------------------------------------------
    print("\n[2] Applying LoRA (per-expert via LoRASwitchLinear)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        # Trinity-Nano is 6B MoE / 1B active — fits without GC at ctx=2048.
        # Set "unsloth" if you push to longer contexts.
        random_state=3407,
    )

    # ------------------------------------------------------------------
    # 3. Real HF dataset → ChatML text
    # ------------------------------------------------------------------
    print(f"\n[3] Loading {DATASET_ID} ({SLICE})...")
    raw = load_dataset(DATASET_ID, split=SLICE)
    print(f"Loaded {len(raw)} conversations (columns: {raw.column_names})")

    def format_fn(batch):
        out = []
        for convo in batch["conversations"]:
            msgs = _sharegpt_to_chatml({"conversations": convo})
            if len(msgs) < 2:       # need at least one user + assistant turn
                out.append("")
                continue
            out.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
        return {"text": out}

    dataset = raw.map(
        format_fn,
        batched=True,
        remove_columns=raw.column_names,
    ).filter(lambda r: bool(r["text"]))
    print(f"Prepared {len(dataset)} ChatML samples")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("\n[4] Training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=3,
            max_steps=20,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs_trinity_sft",
        ),
    )
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Inference with the trained LoRA
    # ------------------------------------------------------------------
    print("\n[5] Testing inference with trained LoRA...")
    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "Give me a one-sentence summary of how AFMoE routes tokens."
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    response = generate(
        model.model, tokenizer, prompt=formatted,
        max_tokens=120, verbose=False,
    )
    print(f"Q: {prompt}")
    print(f"A: {response}")

    # ------------------------------------------------------------------
    # 6. Save adapter + reload roundtrip
    # ------------------------------------------------------------------
    print("\n[6] Saving adapter + reload roundtrip...")
    adapter_dir = "outputs_trinity_sft/saved_adapters"
    model.save_pretrained(adapter_dir)

    fresh, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True,
    )
    fresh.load_adapter(adapter_dir)
    print(f"✓ Reloaded adapter from {adapter_dir} onto a fresh base model")

    # ------------------------------------------------------------------
    # 7. Merge LoRA into base weights (save_pretrained_merged)
    #    Produces a standalone MLX model directory you can share or serve
    #    without the LoRA adapter files.
    # ------------------------------------------------------------------
    print("\n[7] Merging LoRA into base weights...")
    merged_dir = "outputs_trinity_sft/merged_model"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer=tokenizer)
        print(f"✓ Merged model saved to {merged_dir}")
    except Exception as e:
        # Quantized LoRA fusion requires dequant-then-requant; if the
        # installed mlx-lm doesn't support it for QuantizedSwitchLinear,
        # surface the error but don't fail the example.
        print(f"(merged-model export skipped: {type(e).__name__}: {e})")

    print("\n" + "=" * 70)
    print("Trinity-Nano SFT complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
