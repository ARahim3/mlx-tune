"""
Qwen3.5 Text-Only Fine-Tuning with MLX-Tune

Qwen3.5 is natively multimodal (text + vision), but you can fine-tune it
on text-only datasets too. This is useful when you want the Qwen3.5
architecture for text tasks without needing vision capabilities.

Since Qwen3.5 is loaded via mlx-vlm (not mlx-lm), we use FastVisionModel
even for text-only fine-tuning — just skip images in the dataset.

Usage:
    python examples/11_qwen35_text_finetuning.py
"""

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig

# ===========================================================================
# Step 1: Load Qwen3.5
# ===========================================================================
print("=" * 70)
print("Step 1: Loading Qwen3.5-0.8B (natively multimodal)")
print("=" * 70)

model, processor = FastVisionModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-8bit",  # or bf16 for better quality
)

# ===========================================================================
# Step 2: Add LoRA (language layers only — no vision fine-tuning needed)
# ===========================================================================
print("\n" + "=" * 70)
print("Step 2: Adding LoRA Adapters (language only)")
print("=" * 70)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Skip vision — text only
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ===========================================================================
# Step 3: Prepare text-only dataset
# ===========================================================================
print("\n" + "=" * 70)
print("Step 3: Preparing Text-Only Dataset")
print("=" * 70)

# Format: same as vision, just without {"type": "image"} entries
# You can also load from HuggingFace:
#   from datasets import load_dataset
#   dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

dataset = [
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Explain Python in one sentence."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Python is a high-level, interpreted programming language known for its readability and versatility."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "How does gravity work?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to objects and causes them to fall toward the ground."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is an API?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of Japan?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The capital of Japan is Tokyo."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is a neural network?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A neural network is a computing system inspired by the brain, consisting of interconnected nodes (neurons) organized in layers that process information and learn patterns from data."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is the speed of light?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 186,000 miles per second."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is Docker?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Docker is a platform that packages applications and their dependencies into lightweight, portable containers that run consistently across different environments."}]},
    ]},
    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Who wrote Romeo and Juliet?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Romeo and Juliet was written by William Shakespeare, first published in 1597."}]},
    ]},
]

print(f"Dataset: {len(dataset)} text-only samples (no images)")

# ===========================================================================
# Step 4: Test inference BEFORE training
# ===========================================================================
print("\n" + "=" * 70)
print("Step 4: Pre-Training Inference Test")
print("=" * 70)

FastVisionModel.for_inference(model)

questions = [
    "What is machine learning?",
    "What is Python?",
    "What is the capital of Japan?",
]

for q in questions:
    response = model.generate(prompt=q, max_tokens=64, temperature=0.0)
    print(f"\nQ: {q}")
    print(f"A: {response}")

# ===========================================================================
# Step 5: Train
# ===========================================================================
print("\n" + "=" * 70)
print("Step 5: Training (text-only)")
print("=" * 70)

FastVisionModel.for_training(model)

trainer = VLMSFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=dataset,
    args=VLMSFTConfig(
        per_device_train_batch_size=1,
        max_steps=20,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir="outputs_text",
    ),
)

trainer_stats = trainer.train()
print(f"\nTraining metrics: {trainer_stats.metrics}")

# ===========================================================================
# Step 6: Test inference AFTER training
# ===========================================================================
print("\n" + "=" * 70)
print("Step 6: Post-Training Inference Test")
print("NOTE: With only 20 steps on 10 samples, the model overfits to the")
print("training data. Answers will be short and match the training format.")
print("For real results, use a larger dataset and more training steps.")
print("=" * 70)

FastVisionModel.for_inference(model)

questions = [
    "What is machine learning?",
    "What is Python?",
    "What is the capital of Japan?",
]

for q in questions:
    response = model.generate(prompt=q, max_tokens=64, temperature=0.0)
    print(f"\nQ: {q}")
    print(f"A: {response}")

# ===========================================================================
# Step 7: Save adapters
# ===========================================================================
print("\n" + "=" * 70)
print("Step 7: Saving Adapters")
print("=" * 70)

model.save_pretrained("qwen35_text_lora")

# Verify saved files
import os
for f in sorted(os.listdir("qwen35_text_lora")):
    size = os.path.getsize(f"qwen35_text_lora/{f}")
    print(f"  {f} ({size:,} bytes)")

# ===========================================================================
# Step 8: Load saved adapters into a fresh model
# ===========================================================================
print("\n" + "=" * 70)
print("Step 8: Loading Saved Adapters (verify save/load roundtrip)")
print("=" * 70)

model2, processor2 = FastVisionModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-8bit",
)
model2.load_adapter("qwen35_text_lora")

FastVisionModel.for_inference(model2)

for q in questions:
    response = model2.generate(prompt=q, max_tokens=64, temperature=0.0)
    print(f"\nQ: {q}")
    print(f"A: {response}")

print("\n" + "=" * 70)
print("Done! Text-only fine-tuning of Qwen3.5 complete.")
print("=" * 70)
