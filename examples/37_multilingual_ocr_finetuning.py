"""
Multilingual OCR Fine-Tuning

Fine-tunes GLM-OCR on real multilingual receipt images from the CORD-v2
dataset (Consolidated Receipt Dataset). CORD-v2 contains receipts in
various languages with ground truth text annotations.

This demonstrates fine-tuning for multilingual document understanding:
receipts with mixed-language text, numbers, and document layout.

Model: GLM-OCR (0.9B, supports 8+ languages)
Dataset: naver-clova-ix/cord-v2 (real receipt images)

Usage:
    python examples/37_multilingual_ocr_finetuning.py
"""

# ===========================================================================
# MLX-Tune OCR imports
# ===========================================================================
from mlx_tune import (
    FastOCRModel,
    OCRSFTTrainer,
    OCRSFTConfig,
    compute_ocr_metrics,
    convert_ocr_pairs_to_messages,
)

# ===========================================================================
# Step 1: Load GLM-OCR
# ===========================================================================
print("=" * 70)
print("Step 1: Loading GLM-OCR (Multilingual OCR Model)")
print("=" * 70)

model, processor = FastOCRModel.from_pretrained(
    "mlx-community/GLM-OCR-4bit",
    # Other models for multilingual OCR:
    # "mlx-community/GLM-OCR-bf16"           # 0.9B, full precision
    # "mlx-community/DeepSeek-OCR-8bit"      # 0.9B, 100+ languages
    # "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"  # 7B, strong multilingual
    # "mlx-community/LightOnOCR-1B-1025-bf16"  # 1B, 109 languages
)

# ===========================================================================
# Step 2: Add LoRA
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters (vision frozen)")
print("=" * 70)

model = FastOCRModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# ===========================================================================
# Step 3: Load CORD-v2 receipt dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Loading CORD-v2 Receipt Dataset")
print("=" * 70)

from datasets import load_dataset
import json

raw_dataset = load_dataset("naver-clova-ix/cord-v2", split="train[:50]")
print(f"Dataset loaded: {len(raw_dataset)} receipt images")
print(f"Columns: {raw_dataset.column_names}")

# CORD-v2 has 'image' and 'ground_truth' columns
# ground_truth is a JSON string with parsed receipt fields
instruction = "Extract all text from this receipt image."

converted_dataset = []
for sample in raw_dataset:
    # Parse the ground truth JSON to get the text content
    gt = sample.get("ground_truth", "{}")
    if isinstance(gt, str):
        try:
            gt_parsed = json.loads(gt)
        except json.JSONDecodeError:
            gt_parsed = {}
    else:
        gt_parsed = gt

    # Extract text from ground_truth — CORD-v2 stores it in gt_parse.text_sequence
    gt_parse = gt_parsed.get("gt_parse", gt_parsed)

    # Build a flat text representation of the receipt
    text_parts = []
    if isinstance(gt_parse, dict):
        for key, value in gt_parse.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, dict):
                                for inner_k, inner_v in v.items():
                                    if inner_v:
                                        text_parts.append(f"{inner_v}")
                            elif v:
                                text_parts.append(f"{v}")
                    elif item:
                        text_parts.append(str(item))
            elif isinstance(value, str) and value:
                text_parts.append(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if v:
                        text_parts.append(f"{v}")

    text = "\n".join(text_parts) if text_parts else str(gt_parse)

    if text.strip():
        converted_dataset.append(convert_ocr_pairs_to_messages(
            image=sample["image"],
            text=text,
            instruction=instruction,
        ))

print(f"Converted {len(converted_dataset)} samples with valid text")
if converted_dataset:
    # Show a sample
    sample_text = converted_dataset[0]["messages"][1]["content"][0]["text"]
    print(f"Sample receipt text:\n{sample_text[:200]}...")

# ===========================================================================
# Step 4: Pre-training inference
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference")
print("=" * 70)

FastOCRModel.for_inference(model)

print("Testing receipt OCR before training...")
try:
    response = model.transcribe(
        raw_dataset[0]["image"],
        prompt=instruction,
        max_tokens=512,
    )
    print(f"Transcription:\n{response[:200]}...")
except Exception as e:
    print(f"Pre-training error: {e}")

# ===========================================================================
# Step 5: Train
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: Training on Receipt Data")
print("=" * 70)

FastOCRModel.for_training(model)

trainer = OCRSFTTrainer(
    model=model,
    processor=processor,
    train_dataset=converted_dataset,
    args=OCRSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=5e-5,
        logging_steps=1,
        output_dir="ocr_multilingual_outputs",
        max_length=4096,
    ),
)

trainer_stats = trainer.train()

# ===========================================================================
# Step 6: Post-training evaluation
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Receipt OCR Evaluation")
print("=" * 70)

FastOCRModel.for_inference(model)

# Evaluate on a few samples
n_eval = min(5, len(converted_dataset))
eval_images = []
eval_refs = []
for i in range(n_eval):
    msgs = converted_dataset[i]["messages"]
    for msg in msgs:
        if msg["role"] == "user":
            for c in msg["content"]:
                if c.get("type") == "image":
                    eval_images.append(c["image"])
        if msg["role"] == "assistant":
            eval_refs.append(msg["content"][0]["text"])

if eval_images and eval_refs:
    metrics = model.evaluate(
        images=eval_images,
        references=eval_refs,
        prompt=instruction,
        max_tokens=512,
    )

# ===========================================================================
# Step 7: Save
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Model")
print("=" * 70)

model.save_pretrained("ocr_multilingual_lora")
print("LoRA adapters saved to ocr_multilingual_lora/")

print("\n" + "=" * 70)
print("Done! Multilingual OCR fine-tuning complete.")
print("=" * 70)
