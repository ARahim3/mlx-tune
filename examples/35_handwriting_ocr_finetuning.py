"""
Handwriting OCR Fine-Tuning

Fine-tunes Qwen3.5-0.8B on real handwriting images from the Handwriting-OCR
dataset on HuggingFace. This is a practical use case: adapt a VLM to
recognize handwritten text which is typically harder than printed/digital text.

Uses vision+language fine-tuning since the VLM needs to learn handwriting features.

Dataset: prithivMLmods/Handwriting-OCR (real handwriting images)

Usage:
    python examples/35_handwriting_ocr_finetuning.py
"""

# ===========================================================================
# MLX-Tune OCR imports
# ===========================================================================
from mlx_tune import (
    FastOCRModel,
    OCRSFTTrainer,
    OCRSFTConfig,
    compute_ocr_metrics,
    load_ocr_dataset,
)

# ===========================================================================
# Step 1: Load DeepSeek-OCR-2
# ===========================================================================
print("=" * 70)
print("Step 1: Loading Qwen3.5-0.8B for Handwriting OCR")
print("=" * 70)

model, processor = FastOCRModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-bf16",
    # Other models for handwriting OCR:
    # "mlx-community/DeepSeek-OCR-2-8bit"   # 1B, improved OCR
    # "mlx-community/GLM-OCR-4bit"          # 0.9B, lightweight
    # "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"  # 7B, strong vision
)

# ===========================================================================
# Step 2: Add LoRA (vision frozen — optimized encoder)
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters (vision + language)")
print("=" * 70)

model = FastOCRModel.get_peft_model(
    model,
    finetune_vision_layers=True,   # Need vision features for handwriting
    finetune_language_layers=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# ===========================================================================
# Step 3: Load handwriting dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Loading Handwriting OCR Dataset")
print("=" * 70)

instruction = "Transcribe the handwritten text in this image."

converted_dataset = load_ocr_dataset(
    "prithivMLmods/Handwriting-OCR",
    image_column="image",
    text_column="text",
    instruction=instruction,
    split="train[:50]",
)
print(f"Loaded {len(converted_dataset)} handwriting samples")

# ===========================================================================
# Step 4: Pre-training inference
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference")
print("=" * 70)

FastOCRModel.for_inference(model)

from datasets import load_dataset
raw_dataset = load_dataset("prithivMLmods/Handwriting-OCR", split="train[:10]")

print("Testing handwriting transcription before training...")
try:
    response = model.transcribe(
        raw_dataset[0]["image"],
        prompt=instruction,
        max_tokens=256,
    )
    print(f"Ground truth: {raw_dataset[0]['text'][:80]}...")
    print(f"Prediction:   {response[:80]}...")
except Exception as e:
    print(f"Pre-training error: {e}")

# ===========================================================================
# Step 5: Train
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: Training on Handwriting Data")
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
        output_dir="ocr_handwriting_outputs",
        max_length=2048,
    ),
)

trainer_stats = trainer.train()

# ===========================================================================
# Step 6: Post-training evaluation
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Handwriting Evaluation")
print("=" * 70)

FastOCRModel.for_inference(model)

test_images = [raw_dataset[i]["image"] for i in range(min(5, len(raw_dataset)))]
test_refs = [raw_dataset[i]["text"] for i in range(min(5, len(raw_dataset)))]

print("Evaluating handwriting OCR performance...")
metrics = model.evaluate(
    images=test_images,
    references=test_refs,
    prompt=instruction,
    max_tokens=256,
)

# Show per-sample comparison
print("\nPer-sample results:")
predictions = model.batch_transcribe(test_images, prompt=instruction, max_tokens=256, verbose=False)
for i, (pred, ref) in enumerate(zip(predictions, test_refs)):
    from mlx_tune import compute_cer
    cer = compute_cer(pred, ref)
    print(f"  [{i}] CER={cer:.3f}  ref={ref[:40]}...  pred={pred[:40]}...")

# ===========================================================================
# Step 7: Save
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Model")
print("=" * 70)

model.save_pretrained("ocr_handwriting_lora")
print("LoRA adapters saved to ocr_handwriting_lora/")

print("\n" + "=" * 70)
print("Done! Handwriting OCR fine-tuning complete.")
print("=" * 70)
