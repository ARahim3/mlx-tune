"""
Example 15: Spark-TTS Fine-Tuning with MLX-Tune

Fine-tune Spark-TTS (a Qwen2-based TTS model) using LoRA on Apple Silicon.

Spark-TTS converts text to speech by:
1. Tokenizing the text prompt with <|tts|> and <|start_content|> markers
2. Predicting discrete audio tokens via BiCodec (global + semantic)
3. Using text-based tokens like <|bicodec_global_X|>, <|bicodec_semantic_X|>
4. Decoding audio tokens back to a waveform at 16kHz

Key features:
- Ultralight: only 0.5B parameters (great for Apple Silicon)
- Qwen2-based backbone
- BiCodec: separate global (speaker) and semantic (content) tokens
- 16kHz sample rate

Requirements:
    uv pip install 'mlx-tune[audio]'
    # Also needs: datasets, soundfile

Usage:
    python examples/15_spark_tts_finetuning.py
"""

from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator


def main():
    # =========================================================================
    # 1. Load Model + BiCodec
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Spark-TTS Model")
    print("=" * 70)

    model, tokenizer = FastTTSModel.from_pretrained(
        # Spark-TTS 0.5B (bf16) from mlx-community
        model_name="mlx-community/Spark-TTS-0.5B-bf16",
        max_seq_length=2048,
        # BiCodec is bundled with the model (no separate codec needed)
    )

    # =========================================================================
    # 2. Add LoRA Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Adding LoRA Adapters")
    print("=" * 70)

    model = FastTTSModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        # Target all attention + MLP layers (Spark is Qwen2-based, same LoRA targets)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # =========================================================================
    # 3. Load & Prepare Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Loading Dataset")
    print("=" * 70)

    from datasets import load_dataset, Audio

    # Example: Elise voice dataset
    # Spark uses 16kHz sample rate (not 24kHz like Orpheus/OuteTTS)
    dataset = load_dataset("MrDragonFox/Elise", split="train[:10]")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Dataset: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample text: {dataset[0]['text'][:80]}...")

    # =========================================================================
    # 4. Create Data Collator
    # =========================================================================
    collator = TTSDataCollator(
        model=model,
        tokenizer=tokenizer,
        text_column="text",
        audio_column="audio",
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Fine-Tuning")
    print("=" * 70)

    train_data = [dataset[i] for i in range(len(dataset))]

    trainer = TTSSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_data,
        args=TTSSFTConfig(
            output_dir="./spark_tts_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=60,
            warmup_steps=5,
            logging_steps=1,
            weight_decay=0.01,
            sample_rate=16000,  # Spark uses 16kHz
            train_on_completions=True,
        ),
    )

    result = trainer.train()
    print(f"\nFinal loss: {result.metrics['train_loss']:.4f}")

    # =========================================================================
    # 6. Save Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Saving Adapters")
    print("=" * 70)

    model.save_pretrained("./spark_tts_output/final_adapter")
    print("Done! Adapters saved to ./spark_tts_output/final_adapter")


if __name__ == "__main__":
    main()
