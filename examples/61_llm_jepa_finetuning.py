"""
Example 61: LLM-JEPA Fine-Tuning

LLM-JEPA (Huang, LeCun & Balestriero, arXiv 2509.14252) augments standard
next-token prediction with a Joint-Embedding Predictive objective: each item has
two "views" of the same content (here a natural-language description ``text`` and
its regex ``code``), and the JEPA term aligns their last-token embeddings —
``L = L_NTP + lambda * d(Pred(Enc(text)), Enc(code))``.

The fine-tuned artifact is a **normal LoRA-fine-tuned LLM**: you save / merge /
generate exactly as with SFT. The JEPA term just shapes the representation during
training (reported to improve accuracy and resist overfitting). With the default
``num_predictors=0`` the predictor is the identity.

Run it:
    python examples/61_llm_jepa_finetuning.py

For real benchmarks use paired-view datasets like NL-RX (regex), Spider (SQL) or
GSM8K (math), which are what the paper evaluates.
"""

from mlx_tune import FastLanguageModel, LLMJEPATrainer, LLMJEPAConfig

# Set True to also train a pure-NTP baseline (jepa_lambda=0) for an A/B read.
RUN_AB_BASELINE = False


def regex_pairs():
    """Small NL -> regex view pairs (offline, self-contained)."""
    return [
        {"text": "match one or more digits", "code": r"\d+"},
        {"text": "match a word boundary", "code": r"\b"},
        {"text": "match any whitespace character", "code": r"\s"},
        {"text": "match a lowercase letter", "code": r"[a-z]"},
        {"text": "match an uppercase letter", "code": r"[A-Z]"},
        {"text": "match exactly three digits", "code": r"\d{3}"},
        {"text": "match an email-like token", "code": r"\w+@\w+\.\w+"},
        {"text": "match a hex color code", "code": r"#[0-9a-fA-F]{6}"},
        {"text": "match a US zip code", "code": r"\d{5}(-\d{4})?"},
        {"text": "match a 24-hour time", "code": r"([01]\d|2[0-3]):[0-5]\d"},
        {"text": "match an IPv4 octet", "code": r"25[0-5]|2[0-4]\d|1?\d?\d"},
        {"text": "match a non-digit character", "code": r"\D"},
    ]


def train(jepa_lambda: float, output_dir: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        max_seq_length=512,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
    )

    config = LLMJEPAConfig(
        jepa_lambda=jepa_lambda,   # 0.1 = LLM-JEPA, 0.0 = pure-NTP baseline
        jepa_distance="cosine",    # cosine | l2 | mse | infonce
        num_predictors=0,          # k=0 -> predictor is identity (paper default)
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        max_steps=40,
        logging_steps=5,
        max_seq_length=512,
        output_dir=output_dir,
    )

    trainer = LLMJEPATrainer(
        model=model,
        train_dataset=regex_pairs(),
        tokenizer=tokenizer,
        args=config,
    )
    result = trainer.train()
    return model, tokenizer, result


def main():
    print("=" * 70)
    print("LLM-JEPA Fine-Tuning — End-to-End")
    print("=" * 70)

    # --- LLM-JEPA run -------------------------------------------------------
    model, tokenizer, result = train(jepa_lambda=0.1, output_dir="./llm_jepa_output")
    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")

    # --- The artifact is a normal LLM: generate from it ---------------------
    print("\n--- Generation from the fine-tuned model ---")
    prompt = "match one or more digits"
    out = model.generate(prompt=prompt, max_tokens=16, verbose=False)
    print(f"  prompt: {prompt!r}")
    print(f"  output: {out!r}")

    # Merge LoRA into the base for a standalone model (standard SFT flow):
    #   model.save_pretrained_merged("./llm_jepa_merged")

    # --- Optional A/B: pure-NTP baseline ------------------------------------
    if RUN_AB_BASELINE:
        print("\n--- Baseline (jepa_lambda=0, pure NTP) for comparison ---")
        train(jepa_lambda=0.0, output_dir="./llm_jepa_baseline")
        print("Compare the NTP curves / downstream accuracy of the two runs.")


if __name__ == "__main__":
    main()
