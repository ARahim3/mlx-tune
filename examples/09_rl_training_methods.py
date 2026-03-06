"""
Example 9: Reinforcement Learning Training Methods

This example demonstrates MLX-Tune's RL training capabilities:
- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- GRPO (Group Relative Policy Optimization) - DeepSeek R1 style
- KTO, SimPO

These trainers match the Unsloth/TRL API!
"""

from mlx_tune import (
    FastLanguageModel,
    # RL Trainers
    DPOTrainer, DPOConfig,
    ORPOTrainer, ORPOConfig,
    GRPOTrainer, GRPOConfig,
    # Utilities
    prepare_preference_dataset,
    create_reward_function,
)


def demo_dpo_training():
    """
    DPO (Direct Preference Optimization) Training Demo

    DPO trains models on preference data (chosen vs rejected responses)
    without requiring a separate reward model.

    Dataset format:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The less preferred response
    """

    print("=" * 70)
    print("DPO Training Demo")
    print("=" * 70)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
    )
    model = FastLanguageModel.get_peft_model(model, r=16)

    # Preference dataset
    preference_data = [
        {
            "prompt": "Explain what machine learning is.",
            "chosen": "Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data and make decisions without explicit programming. It includes techniques like supervised learning, unsupervised learning, and reinforcement learning.",
            "rejected": "idk its like computers doing stuff automatically i guess"
        },
        {
            "prompt": "What is Python?",
            "chosen": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It's widely used in web development, data science, AI, and automation.",
            "rejected": "python is a snake"
        },
    ]

    # Configure DPO
    config = DPOConfig(
        beta=0.1,  # KL penalty coefficient
        loss_type="sigmoid",  # sigmoid, hinge, ipo, kto_pair
        learning_rate=5e-7,
        max_steps=10,  # Small for demo
        output_dir="./dpo_output",
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        train_dataset=preference_data,
        tokenizer=tokenizer,
        args=config,
    )

    print("\nDPO Config:")
    print(f"  Beta: {config.beta}")
    print(f"  Loss type: {config.loss_type}")
    print(f"  This is the SAME API as TRL's DPOTrainer!")

    # Would train with: trainer.train()
    print("\nTo train: trainer.train()")


def demo_grpo_training():
    """
    GRPO (Group Relative Policy Optimization) Training Demo

    GRPO is used by DeepSeek to train their R1 reasoning models.
    Key features:
    - No value model needed
    - Uses custom reward functions
    - Multiple generations per prompt for group statistics

    Great for: Math reasoning, code generation, structured outputs
    """

    print("\n" + "=" * 70)
    print("GRPO Training Demo (DeepSeek R1 Style)")
    print("=" * 70)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
    )
    model = FastLanguageModel.get_peft_model(model, r=16)

    # Math reasoning dataset
    reasoning_data = [
        {
            "prompt": "What is 15 + 27?",
            "answer": "42"
        },
        {
            "prompt": "If a train travels at 60 mph for 2 hours, how far does it go?",
            "answer": "120 miles"
        },
    ]

    # Create a math reward function
    math_reward = create_reward_function("math")

    # Configure GRPO
    config = GRPOConfig(
        loss_type="grpo",  # Phase 1 accepts grpo/dr_grpo/dapo/bnpo via one shared objective
        beta=0.04,
        num_generations=4,  # Multiple generations per prompt
        temperature=0.7,
        learning_rate=1e-6,
        max_steps=10,
        output_dir="./grpo_output",
    )

    # Create trainer with custom reward function
    trainer = GRPOTrainer(
        model=model,
        train_dataset=reasoning_data,
        tokenizer=tokenizer,
        reward_fn=math_reward,  # Custom reward!
        args=config,
    )

    print("\nGRPO Config:")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Num generations: {config.num_generations}")
    print(f"  Custom reward function: math_reward")
    print(f"  This is how DeepSeek trains reasoning models!")

    # Would train with: trainer.train()
    print("\nTo train: trainer.train()")


def demo_orpo_training():
    """
    ORPO (Odds Ratio Preference Optimization) Training Demo

    ORPO combines SFT and preference learning in one step,
    making it simpler and more memory-efficient than DPO.
    """

    print("\n" + "=" * 70)
    print("ORPO Training Demo")
    print("=" * 70)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    model = FastLanguageModel.get_peft_model(model, r=16)

    # Same format as DPO
    preference_data = [
        {
            "prompt": "Write a haiku about coding",
            "chosen": "Lines of code flow free\nBugs emerge from morning mist\nDebug, compile, run",
            "rejected": "coding is fun i like it a lot yes"
        },
    ]

    config = ORPOConfig(
        beta=0.1,
        learning_rate=8e-6,
        max_steps=10,
    )

    trainer = ORPOTrainer(
        model=model,
        train_dataset=preference_data,
        tokenizer=tokenizer,
        args=config,
    )

    print("\nORPO: Combines SFT + preference alignment in one step!")
    print("More memory efficient than DPO")


def show_available_trainers():
    """Show all available training methods."""

    print("\n" + "=" * 70)
    print("Available RL Training Methods in MLX-Tune")
    print("=" * 70)

    trainers = [
        ("SFTTrainer", "Supervised Fine-Tuning", "Basic instruction tuning"),
        ("DPOTrainer", "Direct Preference Optimization", "Learn from human preferences"),
        ("ORPOTrainer", "Odds Ratio Preference Optimization", "SFT + preference in one step"),
        ("GRPOTrainer", "Group Relative Policy Optimization", "DeepSeek R1 reasoning training"),
        ("KTOTrainer", "Kahneman-Tversky Optimization", "Prospect theory-based"),
        ("SimPOTrainer", "Simple Preference Optimization", "No reference model needed"),
    ]

    print("\n| Trainer | Method | Use Case |")
    print("|---------|--------|----------|")
    for name, method, use_case in trainers:
        print(f"| {name} | {method} | {use_case} |")

    print("\n" + "=" * 70)
    print("GRPO Loss Types (accepted in Phase 1)")
    print("=" * 70)

    grpo_types = [
        ("grpo", "Standard GRPO", "Primary Phase 1 name"),
        ("dr_grpo", "Dr. GRPO", "Accepted alias; shared Phase 1 objective"),
        ("dapo", "DAPO", "Accepted alias; shared Phase 1 objective"),
        ("bnpo", "BNPO", "Accepted alias; shared Phase 1 objective"),
    ]

    print("\n| Loss Type | Name | Description |")
    print("|-----------|------|-------------|")
    for loss_type, name, desc in grpo_types:
        print(f"| {loss_type} | {name} | {desc} |")


def main():
    print("=" * 70)
    print("MLX-Tune: Reinforcement Learning Training Methods")
    print("=" * 70)
    print()
    print("This example shows how to use RL training methods that match")
    print("Unsloth's API. Just change your imports!")
    print()
    print("Unsloth (CUDA):")
    print("  from unsloth import FastLanguageModel")
    print("  from trl import DPOTrainer, DPOConfig")
    print()
    print("MLX-Tune (Apple Silicon):")
    print("  from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig")
    print()

    show_available_trainers()

    demo_dpo_training()
    demo_grpo_training()
    demo_orpo_training()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
What you can do with MLX-Tune:

1. SFT Training: Basic instruction fine-tuning
   trainer = SFTTrainer(model, train_dataset, args=SFTConfig(...))

2. DPO Training: Learn from preference pairs
   trainer = DPOTrainer(model, train_dataset, args=DPOConfig(beta=0.1))

3. ORPO Training: Combined SFT + preference
   trainer = ORPOTrainer(model, train_dataset, args=ORPOConfig(beta=0.1))

4. GRPO Training: Reasoning model training (like DeepSeek R1)
   trainer = GRPOTrainer(model, train_dataset, reward_fn=my_reward,
                         args=GRPOConfig(loss_type='grpo'))

All with the SAME API as Unsloth!
""")


if __name__ == "__main__":
    main()
