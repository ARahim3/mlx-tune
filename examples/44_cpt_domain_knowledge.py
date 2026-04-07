"""
Example 44: Continual Pretraining - Domain Knowledge Injection

Inject specialized medical/scientific knowledge into a base model.
CPT on domain-specific text teaches the model new vocabulary and concepts
without requiring labeled instruction-response pairs.

Use Cases:
- Medical/clinical text adaptation
- Legal document understanding
- Financial analysis specialization
- Scientific literature comprehension

NOTE: Uses small inline dataset for demo. For real use, train on large corpora.
"""

from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("Continual Pretraining - Medical Domain Knowledge")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Base Model
    # ========================================================================
    print("\n[Step 1] Loading base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/SmolLM2-360M-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Model loaded: {model.model_name}")

    # ========================================================================
    # Step 2: Apply LoRA
    # ========================================================================
    print("\n[Step 2] Applying LoRA...")

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
    # Step 3: Prepare Medical Text Corpus
    # ========================================================================
    print("\n[Step 3] Preparing medical text corpus...")

    medical_corpus = [
        {"text": "Cardiovascular disease (CVD) remains the leading cause of mortality worldwide. Atherosclerosis, characterized by the accumulation of lipids and fibrous elements in arterial walls, is the primary pathological process underlying most cardiovascular events. Risk factors include hypertension, dyslipidemia, diabetes mellitus, smoking, and family history. The pathogenesis involves endothelial dysfunction, inflammatory cell recruitment, foam cell formation, and eventual plaque rupture."},
        {"text": "Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency. The condition affects glucose homeostasis through impaired insulin signaling in peripheral tissues. Beta-cell dysfunction progresses over time, leading to inadequate insulin secretion. Management includes lifestyle modifications, metformin as first-line therapy, and potentially GLP-1 receptor agonists or SGLT2 inhibitors for additional glycemic control."},
        {"text": "Immunotherapy has revolutionized cancer treatment by harnessing the body's immune system to target malignant cells. Checkpoint inhibitors, such as anti-PD-1 and anti-CTLA-4 antibodies, block inhibitory signals that tumors exploit to evade immune detection. CAR-T cell therapy engineers patient T cells to express chimeric antigen receptors specific to tumor antigens. These approaches have shown remarkable efficacy in melanoma, lung cancer, and hematological malignancies."},
        {"text": "Alzheimer's disease is a progressive neurodegenerative disorder characterized by cognitive decline and memory loss. The neuropathological hallmarks include extracellular amyloid-beta plaques and intracellular neurofibrillary tangles composed of hyperphosphorylated tau protein. Neuroinflammation, mediated by activated microglia and astrocytes, contributes to disease progression. Current therapeutic approaches target amyloid clearance, tau aggregation, and neuroinflammatory pathways."},
        {"text": "The human microbiome encompasses trillions of microorganisms residing in the gastrointestinal tract, skin, respiratory system, and other body sites. The gut microbiome plays crucial roles in nutrient metabolism, immune system development, and pathogen resistance. Dysbiosis, or microbial imbalance, has been associated with inflammatory bowel disease, obesity, type 2 diabetes, and mental health disorders through the gut-brain axis."},
        {"text": "Pharmacogenomics studies how genetic variation affects drug response, enabling personalized medicine approaches. Cytochrome P450 enzymes, particularly CYP2D6 and CYP2C19, exhibit significant polymorphism affecting drug metabolism. Poor metabolizers may experience adverse effects at standard doses, while ultrarapid metabolizers may require higher doses for therapeutic efficacy. Integration of pharmacogenomic data into clinical decision-making improves treatment outcomes and reduces adverse drug reactions."},
        {"text": "Epigenetic modifications, including DNA methylation, histone modification, and non-coding RNA regulation, control gene expression without altering the DNA sequence. These modifications are heritable through cell division and can be influenced by environmental factors such as diet, stress, and toxin exposure. Aberrant epigenetic patterns are implicated in cancer, autoimmune diseases, and neurological disorders, making epigenetic therapy a promising therapeutic frontier."},
        {"text": "CRISPR-Cas9 gene editing technology has transformed biomedical research and holds promise for treating genetic diseases. The system uses guide RNA to direct the Cas9 nuclease to specific genomic loci, creating double-strand breaks that can be repaired by non-homologous end joining or homology-directed repair. Clinical trials are underway for sickle cell disease, beta-thalassemia, and certain cancers. Delivery challenges and off-target effects remain key areas of ongoing research."},
    ]

    dataset = Dataset.from_list(medical_corpus)
    print(f"Dataset: {len(dataset)} medical text documents")

    # ========================================================================
    # Step 4: Train with CPT
    # ========================================================================
    print("\n[Step 4] Starting CPT on medical corpus...")

    config = CPTConfig(
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,  # 5x smaller for embeddings
        include_embeddings=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        logging_steps=5,
        output_dir="outputs_cpt_medical",
        max_seq_length=2048,
    )

    trainer = CPTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    trainer.train()

    # ========================================================================
    # Step 5: Test Domain Knowledge
    # ========================================================================
    print("\n[Step 5] Testing medical knowledge...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompts = [
        "Atherosclerosis is characterized by",
        "Type 2 diabetes mellitus involves",
        "CRISPR-Cas9 gene editing works by",
    ]

    for prompt in prompts:
        response = generate(
            model.model, tokenizer,
            prompt=prompt,
            max_tokens=80,
            verbose=False,
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {response[:150]}")

    print("\nMedical domain CPT complete!")
    print("\nRecommended pipeline for production:")
    print("  1. CPT on large medical corpus (PubMed, clinical notes)")
    print("  2. SFT on medical Q&A pairs")
    print("  3. Evaluate on medical benchmarks (MedQA, PubMedQA)")


if __name__ == "__main__":
    main()
