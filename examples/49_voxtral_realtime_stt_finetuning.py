"""
Example 49: Voxtral Realtime STT Fine-Tuning with mlx-tune

Fine-tune Mistral's Voxtral Realtime — a *streaming* ASR model with a
fundamentally different architecture from regular Voxtral:

  - Causal audio encoder (32 layers, conv stem, sliding-window attention)
  - Adapter MLP (4x temporal downsampling)
  - LLM decoder (26 layers, GQA, **AdaRMSNorm** time-conditioning, tied embeds)
  - Additive embedding fusion: at each position, the input is
        adapter_out[pos] + decoder.embed_token(token)
    (no audio_tower / language_model split, no audio placeholder tokens)
  - Custom Tekken tokenizer (Mistral's own, not HF)

Because of this, Voxtral Realtime is a *third* architecture type in mlx-tune
(alongside encoder_decoder and audio_llm). LoRA targets are the
Mistral-internal names: decoder.layers[i].attention.{wq,wk,wv,wo} and
.feed_forward_w{1,2,3}. The encoder is frozen by default.

New language adaptation:
    The Tekken tokenizer is byte-level BPE (131K vocab) — it encodes ANY
    UTF-8 text with lossless roundtrip, and the audio encoder processes
    mel spectrograms language-agnostically. So fine-tuning on a new language
    requires ZERO tokenizer changes. Just swap the dataset:

        # Turkish from Common Voice:
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "tr",
                          split="train[:200]", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        collator = STTDataCollator(model, processor,
                                   audio_column="audio", text_column="sentence")

        # Japanese from FLEURS:
        ds = load_dataset("google/fleurs", "ja_jp", split="test")
        collator = STTDataCollator(model, processor,
                                   audio_column="audio", text_column="transcription")

    Any HF dataset with an audio column + text column works. The text column
    is auto-detected from common names (text, transcription, sentence, transcript).

Requirements:
    uv pip install 'mlx-tune[audio]'
    brew install ffmpeg

Usage:
    python examples/49_voxtral_realtime_stt_finetuning.py
"""

from pathlib import Path

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


# Default to the 4-bit MLX-community variant for accessibility (~2.5GB).
# The same code path works with any quantization — just swap the model name:
#
#     "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"   (~2.5GB, fastest)
#     "mlx-community/Voxtral-Mini-4B-Realtime-6bit"        (~3.5GB)
#     "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16"   (~8GB, highest fidelity)
#
# LoRA training works on all three thanks to mlx-lm's LoRALinear.from_base()
# which handles both nn.Linear and nn.QuantizedLinear bases.
MODEL_NAME = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
OUTPUT_DIR = "./out_voxtral_realtime"
N_TRAIN_SAMPLES = 20
MAX_STEPS = 20


def main():
    # =========================================================================
    # 1. Load Voxtral Realtime
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Voxtral Realtime")
    print("=" * 70)

    model, processor = FastSTTModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
    )

    # =========================================================================
    # 2. Pre-train baseline transcription
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Loading Dataset (LibriSpeech dummy)")
    print("=" * 70)

    from datasets import load_dataset
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
    )
    ds = ds.select(range(min(N_TRAIN_SAMPLES, len(ds))))
    print(f"  Loaded {len(ds)} samples")

    # Show one sample's columns for debug
    sample0 = ds[0]
    print(f"  Sample columns: {list(sample0.keys())}")
    print(f"  Sample text: {sample0['text'][:80]}...")

    # Pre-training transcription on the first sample
    print("\n  Pre-train transcription on sample 0:")
    audio_array = sample0["audio"]["array"]
    pre_text = model.transcribe(audio_array)
    print(f"    Reference: {sample0['text']!r}")
    print(f"    Predicted: {pre_text!r}")

    # =========================================================================
    # 3. Add LoRA Adapters (decoder only — encoder stays frozen)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Adding LoRA Adapters (decoder only)")
    print("=" * 70)

    # Conservative hyperparameters for the demo: small rank, attention-only,
    # encoder frozen. The base model is already strong, so a gentle nudge is
    # plenty for an E2E sanity check. Real fine-tuning runs would crank
    # rank/steps/dataset size for the target domain.
    model = FastSTTModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        # Mistral-internal naming — DO NOT use q_proj/k_proj here
        target_modules=["wq", "wk", "wv", "wo"],
        finetune_encoder=False,  # Keep causal encoder frozen
        finetune_decoder=True,
    )

    # =========================================================================
    # 4. Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Training")
    print("=" * 70)

    collator = STTDataCollator(
        model=model,
        processor=processor,
        language="en",
        task="transcribe",
        audio_column="audio",
        text_column="text",
        max_audio_length=30.0,
    )

    trainer = STTSFTTrainer(
        model=model,
        processor=processor,
        data_collator=collator,
        train_dataset=ds,
        args=STTSFTConfig(
            per_device_train_batch_size=1,  # Voxtral RT requires bs=1
            gradient_accumulation_steps=1,
            max_steps=MAX_STEPS,
            # Low LR is critical for Voxtral RT: the base model is already
            # well-trained, AdaRMSNorm time conditioning is sensitive, and
            # starting from a strong point means a high LR drives the decoder
            # off-distribution within a handful of steps. 2e-5 is a sane demo
            # default; for serious fine-tuning use 1e-5 → 5e-5 with linear
            # warmup over the first 5-10% of steps.
            learning_rate=2e-5,
            warmup_steps=2,
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            sample_rate=16000,
            language="en",
            task="transcribe",
        ),
    )
    stats = trainer.train()
    print(f"\nFinal average loss: {stats.metrics['train_loss']:.4f}")

    # =========================================================================
    # 5. Post-train transcription
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Post-Train Transcription")
    print("=" * 70)
    post_text = model.transcribe(audio_array)
    print(f"  Reference: {sample0['text']!r}")
    print(f"  Pre-train: {pre_text!r}")
    print(f"  Post-train: {post_text!r}")

    # =========================================================================
    # 6. Save adapter and reload
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Save adapter, reload, verify")
    print("=" * 70)
    adapter_dir = Path(OUTPUT_DIR) / "adapters"
    model.save_pretrained(str(adapter_dir))
    print(f"  Adapter saved to {adapter_dir}")

    # Reload on a fresh model and verify identical output
    print("\n  Loading fresh model + adapter...")
    fresh_model, _ = FastSTTModel.from_pretrained(model_name=MODEL_NAME)
    fresh_model.load_adapter(str(adapter_dir))
    reload_text = fresh_model.transcribe(audio_array)
    print(f"    Reload text: {reload_text!r}")
    assert reload_text == post_text, (
        f"Reload mismatch: {reload_text!r} != {post_text!r}"
    )
    print("  ✓ Save / reload produced identical output")

    # =========================================================================
    # 7. Save merged model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 7: Save merged model")
    print("=" * 70)
    merged_dir = Path(OUTPUT_DIR) / "merged"
    model.save_pretrained_merged(str(merged_dir))
    print(f"  Merged model saved to {merged_dir}")

    print("\n" + "=" * 70)
    print("✓ Voxtral Realtime fine-tuning E2E complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
