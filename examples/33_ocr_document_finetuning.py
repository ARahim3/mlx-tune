"""
OCR Document Fine-Tuning with a Dedicated OCR Model

Fine-tunes GLM-OCR (0.9B, ranked #1 on OmniDocBench) on LaTeX equation images
using the FastOCRModel API. This demonstrates the "dedicated OCR model" track:
take a model pre-trained for OCR and adapt it to your specific domain.

GLM-OCR has a pre-optimized vision encoder, so we freeze vision layers and
only fine-tune the language decoder (the default for FastOCRModel).

Usage:
    python examples/33_ocr_document_finetuning.py
"""

# ===========================================================================
# MLX-Tune OCR imports
# ===========================================================================
from mlx_tune import (
    FastOCRModel,
    OCRSFTTrainer,
    OCRSFTConfig,
    UnslothVisionDataCollator,
    compute_ocr_metrics,
)

# ===========================================================================
# Step 1: Load the dedicated OCR model
# ===========================================================================
print("=" * 70)
print("Step 1: Loading GLM-OCR-4bit (Dedicated OCR Model)")
print("=" * 70)

model, processor = FastOCRModel.from_pretrained(
    "mlx-community/GLM-OCR-4bit",  # 0.9B, #1 on OmniDocBench
    # Other dedicated OCR models you can try:
    # "mlx-community/DeepSeek-OCR-8bit"      # 0.9B, 32x vision compression
    # "mlx-community/DeepSeek-OCR-2-8bit"    # 1B, improved accuracy
    # "mlx-community/GLM-OCR-bf16"           # 0.9B, full precision
    # "mlx-community/dots.ocr-4bit"          # DOTS OCR specialized
    # "mlx-community/olmOCR-2-7B-1025-5bit"  # 7B, trained on 270K PDF pages
    # "mlx-community/LightOnOCR-1B-1025-bf16"  # 1B, multilingual
)

# ===========================================================================
# Step 2: Add LoRA adapters (vision frozen by default for OCR models)
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters (vision layers frozen)")
print("=" * 70)

model = FastOCRModel.get_peft_model(
    model,
    # finetune_vision_layers=False,  # Default! OCR models have pre-optimized encoders
    # finetune_language_layers=True,  # Default! Adapt language decoder to domain
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ===========================================================================
# Step 3: Prepare the dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Loading LaTeX OCR Dataset")
print("=" * 70)

from datasets import load_dataset

dataset = load_dataset("unsloth/LaTeX_OCR", split="train[:50]")
print(f"Dataset loaded: {len(dataset)} samples")
print(f"Columns: {dataset.column_names}")
print(f"Sample text: {dataset[0]['text'][:80]}...")

# Convert to OCR conversation format
from mlx_tune import convert_ocr_pairs_to_messages

instruction = "Write the LaTeX representation for this image."
converted_dataset = [
    convert_ocr_pairs_to_messages(
        image=sample["image"],
        text=sample["text"],
        instruction=instruction,
    )
    for sample in dataset
]
print(f"Converted {len(converted_dataset)} samples to OCR format")

# ===========================================================================
# Step 4: Pre-training inference test
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference")
print("=" * 70)

FastOCRModel.for_inference(model)

image = dataset[0]["image"]
print("Testing transcribe()...")
try:
    response = model.transcribe(image, prompt=instruction, max_tokens=128)
    print(f"Transcription: {response[:100]}...")
except Exception as e:
    print(f"Pre-training inference error (expected): {e}")

# ===========================================================================
# Step 5: Train the model
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: Training")
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
        output_dir="ocr_glm_outputs",
        max_length=2048,
    ),
)

trainer_stats = trainer.train()

# ===========================================================================
# Step 6: Post-training inference + evaluation
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Evaluation")
print("=" * 70)

FastOCRModel.for_inference(model)

# Transcribe a few test samples
test_images = [dataset[i]["image"] for i in range(min(5, len(dataset)))]
test_refs = [dataset[i]["text"] for i in range(min(5, len(dataset)))]

print("Transcribing test samples...")
predictions = model.batch_transcribe(test_images, prompt=instruction, max_tokens=256)

# Compute metrics
metrics = compute_ocr_metrics(predictions, test_refs)
print(f"\nOCR Metrics:")
print(f"  CER:         {metrics['cer']:.4f}")
print(f"  WER:         {metrics['wer']:.4f}")
print(f"  Exact Match: {metrics['exact_match']:.4f}")

# ===========================================================================
# Step 7: Save the model
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Model")
print("=" * 70)

model.save_pretrained("ocr_glm_lora")
print("LoRA adapters saved to ocr_glm_lora/")

print("\n" + "=" * 70)
print("Done! OCR document fine-tuning complete.")
print("=" * 70)
