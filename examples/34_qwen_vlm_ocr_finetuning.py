"""
Fine-Tuning a General VLM for OCR Tasks

Takes Qwen3.5-0.8B (a general-purpose VLM) and fine-tunes it specifically
for OCR tasks using the FastOCRModel API. This demonstrates the "VLM to OCR"
track: adapt a general vision model to excel at text recognition.

Unlike dedicated OCR models, we enable vision layer fine-tuning here since
the VLM wasn't pre-trained specifically for document understanding.

Usage:
    python examples/34_qwen_vlm_ocr_finetuning.py
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
    load_ocr_dataset,
)

# ===========================================================================
# Step 1: Load a general VLM
# ===========================================================================
print("=" * 70)
print("Step 1: Loading Qwen3.5-0.8B (General VLM for OCR)")
print("=" * 70)

model, processor = FastOCRModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-bf16",
    # Other general VLMs you can fine-tune for OCR:
    # "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"  # 7B, strong vision
    # "mlx-community/pixtral-12b-4bit"               # 12B, Mistral's VLM
    # "mlx-community/Qwen2-VL-2B-Instruct-4bit"     # 2B, lightweight
)

# ===========================================================================
# Step 2: Add LoRA (enable BOTH vision + language fine-tuning)
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters (vision + language layers)")
print("=" * 70)

model = FastOCRModel.get_peft_model(
    model,
    finetune_vision_layers=True,   # Train vision for OCR-specific features
    finetune_language_layers=True,  # Train language for OCR output
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ===========================================================================
# Step 3: Load dataset using OCR helper
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Loading LaTeX OCR Dataset")
print("=" * 70)

# Use the convenient load_ocr_dataset helper
converted_dataset = load_ocr_dataset(
    "unsloth/LaTeX_OCR",
    image_column="image",
    text_column="text",
    instruction="Write the LaTeX representation for this image.",
    split="train[:50]",
)
print(f"Loaded {len(converted_dataset)} samples in OCR format")

# ===========================================================================
# Step 4: Pre-training inference
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference")
print("=" * 70)

FastOCRModel.for_inference(model)

from datasets import load_dataset
raw_dataset = load_dataset("unsloth/LaTeX_OCR", split="train[:5]")

print("Testing transcribe()...")
try:
    response = model.transcribe(
        raw_dataset[0]["image"],
        prompt="Write the LaTeX representation for this image.",
        max_tokens=128,
    )
    print(f"Transcription: {response[:100]}...")
except Exception as e:
    print(f"Pre-training error (expected for general VLM): {e}")

# ===========================================================================
# Step 5: Train
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
        learning_rate=2e-4,  # Higher LR since we're training from a general VLM
        logging_steps=1,
        output_dir="ocr_qwen_outputs",
        max_length=2048,
    ),
)

trainer_stats = trainer.train()

# ===========================================================================
# Step 6: Post-training evaluation
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Evaluation")
print("=" * 70)

FastOCRModel.for_inference(model)

test_images = [raw_dataset[i]["image"] for i in range(min(5, len(raw_dataset)))]
test_refs = [raw_dataset[i]["text"] for i in range(min(5, len(raw_dataset)))]

print("Evaluating OCR performance...")
metrics = model.evaluate(
    images=test_images,
    references=test_refs,
    prompt="Write the LaTeX representation for this image.",
    max_tokens=256,
)

# ===========================================================================
# Step 7: Save
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Model")
print("=" * 70)

model.save_pretrained("ocr_qwen_lora")
print("LoRA adapters saved to ocr_qwen_lora/")

# Uncomment for merged model:
# model.save_pretrained_merged("ocr_qwen_merged", processor)

print("\n" + "=" * 70)
print("Done! VLM-to-OCR fine-tuning complete.")
print("=" * 70)
