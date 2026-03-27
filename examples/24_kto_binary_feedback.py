"""
Example 24: KTO (Kahneman-Tversky Optimization) Training

End-to-end KTO training with binary feedback. Unlike DPO which needs paired
preferences (chosen vs rejected), KTO works with independent examples labeled
as desirable (good) or undesirable (bad). This makes data collection easier.

Usage:
    python examples/24_kto_binary_feedback.py
"""

from mlx_tune import FastLanguageModel, KTOTrainer, KTOConfig


def main():
    print("=" * 70)
    print("KTO Binary Feedback Training — End-to-End")
    print("=" * 70)

    # 1. Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        max_seq_length=1024,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
    )

    # 2. Binary feedback dataset — each sample is independently labeled
    #    label=True means desirable, label=False means undesirable
    kto_data = [
        # Good responses (desirable)
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a subset of artificial intelligence where systems learn patterns from data to make predictions, without being explicitly programmed for each task.",
            "label": True,
        },
        {
            "prompt": "Explain recursion.",
            "completion": "Recursion is when a function calls itself to solve smaller instances of the same problem. It requires a base case to stop and a recursive case that breaks down the problem. Example: factorial(n) = n * factorial(n-1).",
            "label": True,
        },
        {
            "prompt": "What is a REST API?",
            "completion": "A REST API is an interface that follows REST principles: using HTTP methods (GET, POST, PUT, DELETE) for CRUD operations on resources identified by URLs, with stateless communication and JSON responses.",
            "label": True,
        },
        {
            "prompt": "How does Git work?",
            "completion": "Git is a distributed version control system that tracks file changes through snapshots (commits). It supports branching for parallel development, merging to combine work, and maintains a full history for rollbacks.",
            "label": True,
        },
        {
            "prompt": "What is Big O notation?",
            "completion": "Big O describes algorithmic complexity — how runtime or space scales with input size. Common classes: O(1) constant, O(log n) logarithmic, O(n) linear, O(n log n) linearithmic, O(n^2) quadratic.",
            "label": True,
        },
        {
            "prompt": "Explain Docker containers.",
            "completion": "Docker containers package applications with their dependencies into isolated, portable units. Unlike VMs, containers share the host OS kernel, making them lightweight and fast to start.",
            "label": True,
        },
        {
            "prompt": "What is SQL injection?",
            "completion": "SQL injection is an attack where malicious SQL is inserted into application queries through user input. Prevention: use parameterized queries, prepared statements, and input validation.",
            "label": True,
        },
        {
            "prompt": "Explain the difference between TCP and UDP.",
            "completion": "TCP provides reliable, ordered delivery with connection management and error checking. UDP is connectionless and faster but unreliable — no guaranteed delivery or ordering. TCP for web/email, UDP for streaming/gaming.",
            "label": True,
        },
        # Bad responses (undesirable)
        {
            "prompt": "What is machine learning?",
            "completion": "idk computers learning stuff",
            "label": False,
        },
        {
            "prompt": "Explain recursion.",
            "completion": "recursion is recursion",
            "label": False,
        },
        {
            "prompt": "What is a REST API?",
            "completion": "its an api that rests",
            "label": False,
        },
        {
            "prompt": "How does Git work?",
            "completion": "you just save files",
            "label": False,
        },
        {
            "prompt": "What is Big O notation?",
            "completion": "math stuff",
            "label": False,
        },
        {
            "prompt": "Explain Docker containers.",
            "completion": "docker is like a box for code or something",
            "label": False,
        },
        {
            "prompt": "What is SQL injection?",
            "completion": "hacking",
            "label": False,
        },
        {
            "prompt": "Explain the difference between TCP and UDP.",
            "completion": "they are both internet protocols",
            "label": False,
        },
    ]

    # 3. Configure KTO
    config = KTOConfig(
        beta=0.1,                    # Temperature coefficient
        desirable_weight=1.0,        # Weight for positive examples
        undesirable_weight=1.0,      # Weight for negative examples
        learning_rate=5e-7,
        max_steps=30,                # 30 training steps
        logging_steps=5,             # Log every 5 steps
        output_dir="./kto_output",
    )

    # 4. Create trainer
    trainer = KTOTrainer(
        model=model,
        train_dataset=kto_data,
        tokenizer=tokenizer,
        args=config,
    )

    print("\nKTO advantages over DPO:")
    print("  - No need for paired preferences (easier data collection)")
    print("  - Works with independent binary feedback")
    print("  - Based on Kahneman-Tversky prospect theory")

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
