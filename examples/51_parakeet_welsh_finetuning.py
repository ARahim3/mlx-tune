"""
Example 51: Parakeet TDT Welsh fine-tuning (new language, no vocab extension)
=============================================================================

Welsh is a Celtic language with Latin-script writing that is NOT one of
Parakeet v3's 25 training languages. Every character in typical Welsh text
(including diacritics `â ê î ô û ŵ ŷ`) is covered by Parakeet's multilingual
SentencePiece BPE vocabulary, so we can fine-tune without any vocab
extension — this is the "new language, easy path" demonstration.

Dataset: Google FLEURS `cy_gb` split (Welsh, public, no auth required).

Training mode:
  - Loss: CTC (simplest path)
  - LoRA on the Conformer encoder's self-attention projections
  - CTC head warm-started from the pretrained joint network, full-weight
  - Joint network kept frozen

This example demonstrates the fine-tuning pipeline end-to-end: loading,
collation, loss computation, gradient updates, save, reload. It uses a
small training budget (~100 steps, ~25 samples) that is enough to show
the loss decreasing and the pipeline working correctly. For production-
quality Welsh ASR you'd want hundreds of hours of data and thousands of
training steps — this demo is about *correctness*, not quality.

If you want to try a non-Latin language (Arabic, Bengali, Hindi, Thai,
Japanese, Korean, Chinese...), see Example 52, which uses the
`extend_vocabulary()` method to automatically detect missing characters
from your dataset and resize the CTC head.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"
OUTPUT_DIR = "./parakeet_welsh_finetuned"


def main():
    print("=" * 70)
    print("Example 51: Parakeet TDT Welsh fine-tuning (FLEURS cy_gb)")
    print("=" * 70)
    print()

    # 1. Load Parakeet. Warm-starting the CTC head from the pretrained joint
    # network gives a useful starting point even for languages the model
    # wasn't trained on — the encoder's acoustic features are transferable
    # across languages, so the composed projection is a reasonable initial
    # mapping that fine-tuning can refine.
    model, processor = FastSTTModel.from_pretrained(MODEL_NAME)
    model = FastSTTModel.get_peft_model(model, r=16, lora_alpha=32)
    print()

    # 2. Load FLEURS Welsh (public Google multilingual ASR dataset)
    from datasets import load_dataset
    print("Loading FLEURS Welsh (cy_gb)...")
    ds = load_dataset(
        "google/fleurs",
        "cy_gb",
        split="train[:80]",
        trust_remote_code=True,
    )

    # 3. Filter samples that can fit within the audio length cap
    sp = model.sp_tokenizer
    max_audio_samples = model.profile.max_audio_samples  # 10s at 16kHz
    samples = []
    rejected = 0
    for i in range(len(ds)):
        item = ds[i]
        audio = item["audio"]
        audio_arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        if max_audio_samples > 0 and len(audio_arr) > max_audio_samples:
            audio_arr = audio_arr[:max_audio_samples]
        audio_s = len(audio_arr) / sr
        # FLEURS has "transcription" (normalized) and "raw_transcription"
        text = item["transcription"].strip()
        n_tokens = len(sp.encode(text, out_type=int))
        max_tokens_for_audio = int((audio_s * 12.5 - 1) / 2)
        if n_tokens > max_tokens_for_audio:
            rejected += 1
            continue
        samples.append({
            "audio": {"array": audio_arr, "sampling_rate": sr},
            "text": text,
        })
        if len(samples) >= 40:
            break

    print(f"  {len(samples)} samples accepted, {rejected} rejected (too long for audio cap)")
    print(f"  First sentence: {samples[0]['text']}")
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

    # 4. Data collator and training config. Small training budget —
    # enough to demonstrate the pipeline end-to-end with loss decrease
    # and save/reload, not enough to reach production-quality WER.
    # For real Welsh ASR, scale up: 100+ hours of audio, 2000+ steps,
    # and the aggressive LoRA preset (see docstring at top).
    collator = STTDataCollator(model=model, processor=processor)
    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=100,
        learning_rate=1e-4,
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        loss_type="ctc",
    )

    # 5. Pre-training Welsh transcription (CTC head warm-started from joint)
    print("Pre-training CTC Welsh transcription (warm-start, pre-finetune):")
    pre_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"  {pre_ctc[:150]}")
    print()

    # 6. Native TDT on Welsh BEFORE training (not a trained language, so
    # the output will be garbled but functional)
    print("Pre-training native TDT on Welsh (untrained language):")
    pre_tdt = model.transcribe_tdt(samples[0]["audio"]["array"])
    print(f"  {pre_tdt[:150]}")
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

    # 8. Post-training Welsh transcription
    print("Post-training CTC Welsh transcription:")
    post_ctc = model.transcribe_ctc(samples[0]["audio"]["array"])
    print(f"  {post_ctc[:300]}")
    print(f"Reference:")
    print(f"  {samples[0]['text'][:300]}")
    print()

    # 9. Save / reload round trip
    adapter_path = os.path.join(OUTPUT_DIR, "adapters")
    print(f"Reloading adapters from {adapter_path}...")
    model2, _ = FastSTTModel.from_pretrained(MODEL_NAME)
    model2.load_adapter(adapter_path)
    reloaded_ctc = model2.transcribe_ctc(samples[0]["audio"]["array"])
    assert reloaded_ctc == post_ctc, (
        f"Reload mismatch:\n  trained:  {post_ctc}\n  reloaded: {reloaded_ctc}"
    )
    print("Adapter reload: OK")
    print()
    print("Example 51 completed successfully — Welsh language adapted via CTC.")


if __name__ == "__main__":
    main()
