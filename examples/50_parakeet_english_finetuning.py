"""
Example 50: Parakeet TDT English fine-tuning on LibriSpeech (baseline)
======================================================================

This is the canonical Parakeet fine-tuning example. It demonstrates the core
workflow on the original training language (English), proving that the
pipeline works end-to-end before you take it to new languages:

1. Load `mlx-community/parakeet-tdt-0.6b-v3` (FastConformer + TDT transducer)
2. Apply LoRA on the Conformer encoder's attention projections
3. Train with CTC loss on a new CTC head mounted on the encoder
4. Verify loss decreases, save the adapters, reload them, and transcribe
5. Verify the native TDT decoding path still works post-training

LoRA target preset (defaults to "Standard"):
  - Minimal  : linear_q, linear_k, linear_v, linear_out   (~4.7M params)
  - Standard : + linear_pos                               (~5.9M params, default)
  - Aggressive: + feed_forward1.linear{1,2}, feed_forward2.linear{1,2} (~30M)

Loss type options (all implemented in pure MLX):
  - "ctc"    : frame-level CTC on the new head (fastest, simplest)
  - "rnnt"   : full RNN-T forward-backward on the existing joint network
  - "tdt"    : Token-and-Duration Transducer (NVIDIA's native loss)
  - "hybrid" : weighted sum of CTC + TDT (matches NVIDIA's Stage 2 recipe)

This example uses `loss_type="ctc"` for fast iteration. Example 51 (Welsh)
uses `hybrid` to demonstrate the joint network training path.
"""

import os
import sys
import numpy as np

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
OUTPUT_DIR = "./parakeet_english_finetuned"


def main():
    # 1. Load the pretrained Parakeet TDT model
    print("=" * 70)
    print("Example 50: Parakeet TDT English fine-tuning")
    print("=" * 70)
    print()
    model, processor = FastSTTModel.from_pretrained(MODEL_NAME)

    # 2. Apply LoRA on the Conformer encoder
    model = FastSTTModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        # Default preset includes linear_pos; uncomment to customize:
        # target_modules=["linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"],
    )
    print()

    # 3. Load a small LibriSpeech subset and filter it for CTC feasibility.
    # CTC requires the encoder to produce at least (2*U + 1) frames, where U
    # is the number of BPE tokens. With Parakeet's 8x subsampling and
    # profile.max_audio_samples=160000 (10s), the max encoder length is ~125
    # frames, which fits ~60 BPE tokens. We filter out samples where the
    # transcript is too long for the (possibly clamped) audio.
    from datasets import load_dataset
    print("Loading LibriSpeech dummy dataset...")
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    # Convert and filter. Rough heuristic: require tokens*2+1 <= audio_s*12.5
    # (12.5Hz is Parakeet's encoder frame rate after 8x subsampling from
    # 100Hz mel frame rate).
    sp = model.sp_tokenizer
    samples = []
    for i in range(len(ds)):
        item = ds[i]
        audio = item["audio"]
        audio_arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        # Cap to the profile's audio length (10s by default)
        max_samples = model.profile.max_audio_samples
        if max_samples > 0 and len(audio_arr) > max_samples:
            audio_arr = audio_arr[:max_samples]
        audio_s = len(audio_arr) / sr
        text = item["text"].lower()
        n_tokens = len(sp.encode(text, out_type=int))
        max_tokens_for_audio = int((audio_s * 12.5 - 1) / 2)
        if n_tokens > max_tokens_for_audio:
            continue
        samples.append({
            "audio": {"array": audio_arr, "sampling_rate": sr},
            "text": text,
        })
        if len(samples) >= 20:
            break

    print(f"  Using {len(samples)} filtered samples")
    reference_text = samples[0]["text"]
    print(f"  Reference (sample 0): {reference_text}")
    print()

    class SimpleDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, key):
            if isinstance(key, slice):
                return self.samples[key]
            return self.samples[key]

    train_ds = SimpleDataset(samples)

    # 4. Data collator and training config
    collator = STTDataCollator(model=model, processor=processor)
    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=150,
        learning_rate=3e-4,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        loss_type="ctc",
    )

    # 5. Pre-training CTC transcription (should be garbage — head is random)
    print("Pre-training CTC transcription (random head):")
    pre_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"  {pre_ctc[:120]}...")
    print()

    # 6. Verify the native TDT inference path works before training
    print("Pre-training native TDT transcription (pretrained):")
    pre_tdt = model.transcribe_tdt(samples[0]["audio"]["array"])
    print(f"  {pre_tdt}")
    print()

    # 7. Train
    trainer = STTSFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_ds,
        data_collator=collator,
        args=config,
    )
    trainer.train()
    print()

    # 8. Post-training CTC transcription
    print("Post-training CTC transcription:")
    post_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"  {post_ctc[:200]}")
    print()

    # 9. Native TDT path still works (joint/decoder/LSTM weights weren't
    # touched — only the encoder's LoRA adapters modified the acoustic
    # features). The TDT decoding path may produce slightly different
    # text because the encoder output drifted, but it should still run
    # without crashes and produce recognizable English.
    print("Post-training native TDT transcription (encoder adapted):")
    post_tdt = model.transcribe_tdt(samples[0]["audio"]["array"])
    print(f"  {post_tdt}")
    print()
    assert len(post_tdt) > 0, "Native TDT path produced empty output after training"
    print("Backwards compatibility: OK — native TDT path still functional")
    print()

    # 10. Save / load round trip
    adapter_path = os.path.join(OUTPUT_DIR, "adapters")
    print(f"Reloading adapters from {adapter_path}...")
    model2, _ = FastSTTModel.from_pretrained(MODEL_NAME)
    model2.load_adapter(adapter_path)
    reloaded_ctc = model2.transcribe_ctc(samples[0]["audio"]["array"])
    assert reloaded_ctc == post_ctc, (
        f"Reloaded model produced different output:\n"
        f"  original: {post_ctc}\n"
        f"  reloaded: {reloaded_ctc}"
    )
    print("Adapter reload: OK — reloaded model matches trained model")
    print()
    print("Example 50 completed successfully.")


if __name__ == "__main__":
    main()
