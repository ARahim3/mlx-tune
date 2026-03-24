"""
Example 14: OuteTTS Fine-Tuning with MLX-Tune

Fine-tune OuteTTS (a Llama-based TTS model) to clone or adapt a voice
using LoRA on Apple Silicon.

OuteTTS converts text to speech by:
1. Tokenizing the text prompt with special markers (<|text_start|>, <|text_end|>)
2. Predicting discrete audio tokens via DAC (Descript Audio Codec)
3. Using text-based audio tokens like <|c1_X|>, <|c2_X|> (not numeric offsets)
4. Decoding audio tokens back to a waveform at 24kHz

Key differences from Orpheus (example 12):
- Smaller model: 1B params (vs Orpheus 3B)
- DAC codec (vs SNAC)
- Text-based audio tokens (vs numeric token IDs with offset)
- Loaded via mlx-audio's TTS loader internally

The API is identical -- same FastTTSModel, TTSSFTTrainer, TTSDataCollator.

Requirements:
    uv pip install 'mlx-tune[audio]'
    # Also needs: datasets, soundfile

Usage:
    python examples/14_outetts_finetuning.py
"""

from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator


def main():
    # =========================================================================
    # 1. Load Model + DAC Audio Codec
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading OuteTTS Model")
    print("=" * 70)

    model, tokenizer = FastTTSModel.from_pretrained(
        # OuteTTS 1B (8-bit quantized) from mlx-community
        model_name="mlx-community/Llama-OuteTTS-1.0-1B-8bit",
        max_seq_length=2048,
        # DAC codec is auto-detected from the "outetts" profile
        # (default: mlx-community/dac-speech-24khz-1.5kbps)
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
        # Target all attention + MLP layers (OuteTTS is Llama-based)
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

    # Example: Elise voice dataset (~1200 samples, ~3 hours)
    # Each sample has 'audio' (waveform) and 'text' (transcript)
    dataset = load_dataset("MrDragonFox/Elise", split="train[:10]")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

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

    # Convert to list for trainer
    train_data = [dataset[i] for i in range(len(dataset))]

    trainer = TTSSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_data,
        args=TTSSFTConfig(
            output_dir="./outetts_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=60,
            warmup_steps=5,
            logging_steps=1,
            weight_decay=0.01,
            train_on_completions=True,  # Only train on audio tokens
        ),
    )

    result = trainer.train()
    print(f"\nFinal loss: {result.metrics['train_loss']:.4f}")

    # =========================================================================
    # 6. Generate Speech
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Generating Speech")
    print("=" * 70)

    FastTTSModel.for_inference(model)

    audio = model.generate(
        "Hello, this is a test of voice cloning with OuteTTS and MLX-Tune.",
        max_tokens=1250,  # ~10 seconds of audio
        temperature=0.6,
        top_p=0.95,
    )

    if len(audio) > 0:
        import soundfile as sf
        sf.write("outetts_output.wav", audio, model.sample_rate)
        print(f"Audio saved: outetts_output.wav ({len(audio) / model.sample_rate:.1f}s)")
    else:
        print("No audio generated (try increasing max_tokens)")

    # =========================================================================
    # 7. Save Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Adapters")
    print("=" * 70)

    model.save_pretrained("./outetts_output/final_adapter")
    print("Done! Adapters saved to ./outetts_output/final_adapter")


if __name__ == "__main__":
    main()
