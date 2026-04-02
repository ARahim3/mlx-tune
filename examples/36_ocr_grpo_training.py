"""
OCR GRPO Training (Reinforcement Learning for OCR)

Uses Group Relative Policy Optimization to improve OCR accuracy using
character-level reward functions. Instead of standard SFT, GRPO generates
multiple transcription attempts per image and uses CER (Character Error Rate)
as the reward signal to reinforce more accurate transcriptions.

This is a novel approach: applying RL to OCR to optimize for character accuracy.

Model: Qwen3.5-0.8B (general VLM)
Dataset: unsloth/LaTeX_OCR
Reward: Combined CER + Exact Match reward

Usage:
    python examples/36_ocr_grpo_training.py
"""

# ===========================================================================
# MLX-Tune OCR imports
# ===========================================================================
from mlx_tune import (
    FastOCRModel,
    OCR_MODELS,
    cer_reward,
    exact_match_reward,
    combined_ocr_reward,
    compute_ocr_metrics,
)
from mlx_tune.ocr import OCRGRPOTrainer, OCRGRPOConfig

# ===========================================================================
# Step 1: Load the model
# ===========================================================================
print("=" * 70)
print("Step 1: Loading Qwen3.5-0.8B for OCR GRPO Training")
print("=" * 70)

model, processor = FastOCRModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-bf16",
    # Other models for OCR GRPO:
    # "mlx-community/GLM-OCR-4bit"          # 0.9B, dedicated OCR
    # "mlx-community/DeepSeek-OCR-8bit"     # 0.9B, dedicated OCR
    # "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"  # 7B, strong vision
)

# ===========================================================================
# Step 2: Add LoRA
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters")
print("=" * 70)

model = FastOCRModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16,
    lora_alpha=16,
)

# ===========================================================================
# Step 3: Prepare GRPO dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Loading LaTeX OCR Dataset for GRPO")
print("=" * 70)

from datasets import load_dataset

raw_dataset = load_dataset("unsloth/LaTeX_OCR", split="train[:20]")
print(f"Loaded {len(raw_dataset)} samples for GRPO training")

# GRPO dataset format: each sample has prompt (image + instruction) + answer
instruction = "Write the LaTeX representation for this image."

grpo_dataset = []
for sample in raw_dataset:
    grpo_dataset.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
        ],
        "answer": sample["text"],  # Ground truth for reward computation
    })
print(f"Prepared {len(grpo_dataset)} GRPO samples")

# ===========================================================================
# Step 4: Define reward function
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Defining OCR Reward Functions")
print("=" * 70)

# Demonstrate individual rewards
sample_pred = r"\frac{1}{2}"
sample_ref = r"\frac{1}{2}"
print(f"Perfect match:")
print(f"  CER reward:     {cer_reward(sample_pred, sample_ref):.3f}")
print(f"  EM reward:      {exact_match_reward(sample_pred, sample_ref):.3f}")
print(f"  Combined:       {combined_ocr_reward(sample_pred, sample_ref):.3f}")

sample_pred_bad = r"\frac{1}{3}"
print(f"\nOne char error:")
print(f"  CER reward:     {cer_reward(sample_pred_bad, sample_ref):.3f}")
print(f"  EM reward:      {exact_match_reward(sample_pred_bad, sample_ref):.3f}")
print(f"  Combined:       {combined_ocr_reward(sample_pred_bad, sample_ref):.3f}")

# Custom reward function that weights format correctness
def latex_ocr_reward(response: str, ground_truth: str) -> float:
    """Custom reward: CER accuracy + bonus for matching LaTeX structure."""
    base = cer_reward(response, ground_truth)

    # Bonus for having matching LaTeX delimiters
    structure_bonus = 0.0
    for pair in [("{", "}"), (r"\frac", ""), (r"\sum", ""), (r"\int", "")]:
        if pair[0] in ground_truth and pair[0] in response:
            structure_bonus += 0.05

    return min(1.0, base * 0.8 + structure_bonus + exact_match_reward(response, ground_truth) * 0.15)

# ===========================================================================
# Step 5: GRPO Training
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: GRPO Training with CER Reward")
print("=" * 70)

FastOCRModel.for_training(model)

trainer = OCRGRPOTrainer(
    model=model,
    train_dataset=grpo_dataset,
    processor=processor,
    reward_fn=latex_ocr_reward,
    args=OCRGRPOConfig(
        num_generations=2,
        temperature=0.7,
        max_completion_length=256,
        learning_rate=1e-6,
        max_steps=10,
        logging_steps=1,
        output_dir="ocr_grpo_outputs",
    ),
)

trainer.train()

# ===========================================================================
# Step 6: Post-training evaluation
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Evaluation")
print("=" * 70)

FastOCRModel.for_inference(model)

test_images = [raw_dataset[i]["image"] for i in range(min(5, len(raw_dataset)))]
test_refs = [raw_dataset[i]["text"] for i in range(min(5, len(raw_dataset)))]

metrics = model.evaluate(
    images=test_images,
    references=test_refs,
    prompt=instruction,
    max_tokens=256,
)

# ===========================================================================
# Step 7: Save
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving")
print("=" * 70)

model.save_pretrained("ocr_grpo_lora")
print("GRPO-trained LoRA adapters saved to ocr_grpo_lora/")

print("\n" + "=" * 70)
print("Done! OCR GRPO training complete.")
print("=" * 70)
