"""
Example 42: LFM2.5-Thinking Reasoning Fine-Tuning

Fine-tune LFM2.5-1.2B-Thinking for chain-of-thought reasoning tasks.
This model uses <think>...</think> tags for internal reasoning (like Qwen3.5).

Model: LFM2.5-1.2B-Thinking
Target Modules: q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3

NOTE: Requires downloading the model.
"""

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset
import re


def main():
    print("=" * 70)
    print("LFM2.5-Thinking - Reasoning Fine-Tuning")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Model
    # ========================================================================
    print("\n[Step 1] Loading LFM2.5-Thinking model...")

    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/LFM2.5-1.2B-Thinking-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA
    # ========================================================================
    print("\n[Step 2] Applying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "in_proj", "w1", "w2", "w3",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ========================================================================
    # Step 3: Prepare Reasoning Dataset
    # ========================================================================
    print("\n[Step 3] Preparing reasoning dataset...")

    # Chain-of-thought reasoning examples with <think> tags
    reasoning_data = [
        {
            "messages": [
                {"role": "user", "content": "If a train travels 120 km in 2 hours, what is its average speed? And how far will it travel in 5 hours at the same speed?"},
                {"role": "assistant", "content": "<think>\nFirst, I need to find the average speed.\nSpeed = Distance / Time = 120 km / 2 hours = 60 km/h\n\nNow for 5 hours at 60 km/h:\nDistance = Speed x Time = 60 x 5 = 300 km\n</think>\n\nThe average speed is 60 km/h. At this speed, the train will travel 300 km in 5 hours."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "A store offers a 20% discount on a $150 jacket. If sales tax is 8%, what is the final price?"},
                {"role": "assistant", "content": "<think>\nStep 1: Calculate the discount\nDiscount = 20% of $150 = 0.20 x 150 = $30\n\nStep 2: Price after discount\n$150 - $30 = $120\n\nStep 3: Calculate sales tax\nTax = 8% of $120 = 0.08 x 120 = $9.60\n\nStep 4: Final price\n$120 + $9.60 = $129.60\n</think>\n\nThe final price is $129.60. The jacket costs $120 after the 20% discount, plus $9.60 in sales tax."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "In a class of 30 students, 18 play soccer and 15 play basketball. If 8 play both, how many play neither?"},
                {"role": "assistant", "content": "<think>\nI'll use the inclusion-exclusion principle.\n\nStudents playing at least one sport:\n|Soccer OR Basketball| = |Soccer| + |Basketball| - |Both|\n= 18 + 15 - 8 = 25\n\nStudents playing neither:\nTotal - At least one = 30 - 25 = 5\n</think>\n\n5 students play neither sport. Using inclusion-exclusion: 18 + 15 - 8 = 25 play at least one sport, leaving 30 - 25 = 5."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the sum of all prime numbers less than 20?"},
                {"role": "assistant", "content": "<think>\nLet me list all primes less than 20:\n2, 3, 5, 7, 11, 13, 17, 19\n\nNow sum them:\n2 + 3 = 5\n5 + 5 = 10\n10 + 7 = 17\n17 + 11 = 28\n28 + 13 = 41\n41 + 17 = 58\n58 + 19 = 77\n</think>\n\nThe sum of all prime numbers less than 20 is 77. The primes are: 2, 3, 5, 7, 11, 13, 17, 19."},
            ]
        },
    ]

    def format_reasoning(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(reasoning_data)
    dataset = dataset.map(format_reasoning, batched=True)
    print(f"Dataset prepared with {len(dataset)} reasoning examples")

    # ========================================================================
    # Step 4: Train
    # ========================================================================
    print("\n[Step 4] Training on reasoning data...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir="outputs_lfm2_thinking",
            lr_scheduler_type="cosine",
        ),
    )

    trainer.train()

    # ========================================================================
    # Step 5: Inference with Think Tag Handling
    # ========================================================================
    print("\n[Step 5] Testing reasoning inference...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "If you have 3 red balls and 5 blue balls in a bag, what is the probability of drawing 2 red balls in a row without replacement?"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model.model, tokenizer,
        prompt=formatted,
        max_tokens=300,
        verbose=False,
    )

    # Strip <think> tags for clean display
    clean_response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

    print(f"\nQ: {prompt}")
    print(f"A (full): {response[:200]}...")
    print(f"A (clean): {clean_response[:200]}")

    print("\nLFM2.5-Thinking fine-tuning complete!")


if __name__ == "__main__":
    main()
