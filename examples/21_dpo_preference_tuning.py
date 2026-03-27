"""
Example 21: DPO (Direct Preference Optimization) Training

End-to-end DPO training that loads a real model, trains on preference pairs,
and saves LoRA adapters. DPO trains models to prefer chosen responses over
rejected ones without needing a separate reward model.

Usage:
    python examples/21_dpo_preference_tuning.py
"""

from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig


def main():
    print("=" * 70)
    print("DPO Preference Tuning — End-to-End")
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

    # 2. Preference dataset — chosen vs rejected responses
    preference_data = [
        {
            "prompt": "Explain what machine learning is in simple terms.",
            "chosen": "Machine learning is a type of AI where computers learn patterns from data to make predictions or decisions, rather than following explicit rules. For example, a spam filter learns from labeled emails to classify new ones.",
            "rejected": "idk its like computers doing stuff i guess",
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.",
            "rejected": "france",
        },
        {
            "prompt": "How does photosynthesis work?",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. Light energy is captured by chlorophyll in the leaves, driving chemical reactions that produce food for the plant.",
            "rejected": "plants eat sunlight or something",
        },
        {
            "prompt": "Write a Python function to reverse a string.",
            "chosen": "Here's a clean solution:\n\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```\n\nThis uses Python's slice notation with a step of -1 to reverse the string efficiently.",
            "rejected": "def f(x): return x",
        },
        {
            "prompt": "What is the difference between a list and a tuple in Python?",
            "chosen": "Lists are mutable (can be modified after creation) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses (). Tuples are slightly faster and can be used as dictionary keys since they're hashable.",
            "rejected": "they are basically the same thing",
        },
        {
            "prompt": "Explain the concept of recursion.",
            "chosen": "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. Each recursive call works on a simpler version of the problem until reaching a base case. For example, factorial(n) = n * factorial(n-1) with base case factorial(0) = 1.",
            "rejected": "recursion is recursion",
        },
        {
            "prompt": "What are the benefits of regular exercise?",
            "chosen": "Regular exercise improves cardiovascular health, strengthens muscles and bones, boosts mood through endorphin release, enhances sleep quality, helps maintain a healthy weight, and reduces the risk of chronic diseases like diabetes and heart disease.",
            "rejected": "its good for you",
        },
        {
            "prompt": "How do you make a good cup of coffee?",
            "chosen": "For a great cup: use freshly ground beans (about 2 tbsp per 6 oz water), heat water to 195-205F (just off boiling), brew for 4-5 minutes for drip/pour-over, and use filtered water. The grind size should match your brewing method — coarser for French press, finer for espresso.",
            "rejected": "put water and coffee in a cup",
        },
        {
            "prompt": "What causes seasons on Earth?",
            "chosen": "Seasons are caused by Earth's 23.5-degree axial tilt relative to its orbital plane around the Sun. When the Northern Hemisphere tilts toward the Sun, it receives more direct sunlight, creating summer. When tilted away, less direct light creates winter. It is not caused by distance from the Sun.",
            "rejected": "the earth is closer to the sun in summer",
        },
        {
            "prompt": "Explain what an API is.",
            "chosen": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. Think of it like a waiter in a restaurant — you tell the waiter your order (request), and the waiter brings your food (response) from the kitchen (server).",
            "rejected": "its a computer thing",
        },
        {
            "prompt": "What is the Pythagorean theorem?",
            "chosen": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2. For example, a 3-4-5 triangle: 9 + 16 = 25.",
            "rejected": "a+b=c",
        },
        {
            "prompt": "How does a neural network learn?",
            "chosen": "A neural network learns through a process called backpropagation. During training, it makes predictions, compares them to correct answers using a loss function, and then adjusts its weights to reduce the error. This process repeats over many examples until the network's predictions improve.",
            "rejected": "magic",
        },
    ]

    # 3. Configure DPO training
    config = DPOConfig(
        beta=0.1,                    # KL penalty coefficient
        loss_type="sigmoid",         # Standard DPO loss
        learning_rate=5e-7,          # Low LR typical for preference optimization
        max_steps=30,                # 30 training steps
        logging_steps=5,             # Log every 5 steps
        output_dir="./dpo_output",
    )

    # 4. Create trainer
    trainer = DPOTrainer(
        model=model,
        train_dataset=preference_data,
        tokenizer=tokenizer,
        args=config,
    )

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
