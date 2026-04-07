"""
Example 43: Continual Pretraining - Language Adaptation

Teach a model new language capabilities by training on raw text.
This example adapts a base model to Turkish using CPT.

CPT Key Differences from SFT:
- Uses BASE models (not instruction-tuned)
- Raw text datasets (no chat template)
- Loss on ALL tokens (no response masking)
- Includes embed_tokens + lm_head for vocabulary adaptation
- Lower learning rate for embeddings (prevents catastrophic forgetting)

NOTE: Requires downloading a model. Uses a small inline dataset for demo.
For real use, train on large corpora (Wikipedia, Common Crawl, etc.)
"""

from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("Continual Pretraining - Language Adaptation (Turkish)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load BASE Model (not instruction-tuned!)
    # ========================================================================
    print("\n[Step 1] Loading base model (NOT instruct)...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/SmolLM2-135M-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA with Embedding Layers
    # ========================================================================
    print("\n[Step 2] Applying LoRA (CPT will auto-add embed_tokens + lm_head)...")

    # Note: embed_tokens and lm_head will be auto-added by CPTTrainer
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )

    # ========================================================================
    # Step 3: Prepare Turkish Text Dataset
    # ========================================================================
    print("\n[Step 3] Preparing Turkish text corpus...")

    # Sample Turkish text (in real use, load a large corpus)
    turkish_texts = [
        {"text": "Turkiye, Avrupa ve Asya kitalari arasinda yer alan bir ulkedir. Baskenti Ankara'dir ve en buyuk sehri Istanbul'dur. Turkiye, zengin tarihi ve kulturel mirasıyla taninir. Osmanli Imparatorlugu'nun mirascisi olarak, binlerce yillik medeniyet tarihine ev sahipligi yapar."},
        {"text": "Istanbul, dunyanin en buyuk sehirlerinden biridir ve iki kitayi birbirine baglayan benzersiz konumuyla taninir. Bogazici Koprusu, Avrupa ve Asya'yi birbirine baglar. Sehir, Ayasofya, Topkapi Sarayi ve Sultan Ahmet Camii gibi tarihi yapilarla doludur."},
        {"text": "Turk mutfagi, dunyanin en zengin mutfaklarindan biridir. Kebap, baklava, lokum ve Turk kahvesi en bilinen lezzetler arasindadir. Her bolgenin kendine ozgu yemekleri vardir. Karadeniz bolgesi muhlama ve hamsi ile, Guneydogu ise lahmacun ve ciğ kofte ile unludur."},
        {"text": "Kapadokya, Turkiye'nin en populer turizm merkezlerinden biridir. Peri bacalari olarak bilinen dogal kaya olusumlari, yer alti sehirleri ve sicak hava balonlariyla unludur. UNESCO Dunya Mirasi Listesi'nde yer alan bolge, her yil milyonlarca turisti agirlar."},
        {"text": "Yapay zeka, bilgisayar biliminin hizla gelisen bir dalıdir. Makine ogrenimi, derin ogrenme ve dogal dil isleme gibi alt alanlari vardir. Turkiye'de de yapay zeka arastirmalari hiz kazanmistir. Universitelerde ve teknoloji sirketlerinde onemli calismalar yurutulmektedir."},
        {"text": "Egitim, bir toplumun gelecegini sekillendiren en onemli faktorlerden biridir. Turkiye'de zorunlu egitim suresi 12 yildir. Ilkokul, ortaokul ve lise asamalarindan olusur. Universiteler, ogrencilere uzmanlik alanlarina gore cesitli programlar sunmaktadir."},
        {"text": "Turkiye'nin cografi yapisi oldukca cesitlidir. Kuzeyinde Karadeniz, batisinda Ege Denizi, guneyinde Akdeniz yer alir. Ic Anadolu'da genis stepter, dogu Anadolu'da yuksek daglar bulunur. Bu cografi cesitlilik, zengin bir biyocesitliligi de beraberinde getirir."},
        {"text": "Bilim ve teknoloji, modern dunyanin temel yapitaslarindan biridir. Turkiye, uzay teknolojileri, savunma sanayi ve yazilim gelistirme alanlarinda onemli adimlar atmaktadir. TUBITAK, ulkenin bilimsel arastirma ve teknoloji gelistirme faaliyetlerini koordine eden kurulustur."},
    ]

    dataset = Dataset.from_list(turkish_texts)
    print(f"Dataset: {len(dataset)} Turkish text documents")

    # ========================================================================
    # Step 4: Configure CPT Training
    # ========================================================================
    print("\n[Step 4] Configuring CPT...")

    config = CPTConfig(
        # Main learning rate
        learning_rate=5e-5,
        # Embedding LR: 10x smaller to prevent catastrophic forgetting
        embedding_learning_rate=5e-6,
        # Include embed_tokens + lm_head (default True)
        include_embeddings=True,
        # Training params
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        logging_steps=5,
        output_dir="outputs_cpt_turkish",
        max_seq_length=max_seq_length,
        lr_scheduler_type="cosine",
    )

    # ========================================================================
    # Step 5: Train with CPT
    # ========================================================================
    print("\n[Step 5] Starting continual pretraining...")

    trainer = CPTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    trainer.train()

    # ========================================================================
    # Step 6: Test Generation in Turkish
    # ========================================================================
    print("\n[Step 6] Testing Turkish generation...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "Turkiye'nin baskenti"
    response = generate(
        model.model, tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {response}")

    print("\nCPT Language Adaptation complete!")
    print("\nNext steps for production:")
    print("  1. Train on large corpus (Wikipedia, news, books)")
    print("  2. Use 1000+ steps with larger batch sizes")
    print("  3. Follow up with instruction fine-tuning (SFT)")
    print("  4. Pipeline: CPT -> SFT -> RLHF for best results")


if __name__ == "__main__":
    main()
