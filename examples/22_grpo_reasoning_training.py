"""
Example 22: GRPO (Group Relative Policy Optimization) Reasoning Training

End-to-end GRPO training for building reasoning models (DeepSeek R1 style).
GRPO generates multiple completions per prompt, scores them with reward
functions, and uses group-normalized advantages for policy gradient updates.

Usage:
    python examples/22_grpo_reasoning_training.py
"""

import re
from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig


# System prompt that encourages structured reasoning
SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.
Always show your reasoning before giving the final answer.
Format your response as:
<reasoning>
[Your step-by-step work here]
</reasoning>
<answer>
[Your final answer here]
</answer>"""


def correctness_reward(response: str, ground_truth: str) -> float:
    """
    Reward for getting the correct answer.
    Extracts the answer from <answer> tags and compares to ground truth.
    """
    # Try to extract from <answer> tags
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        # Fall back to extracting last number
        extracted = response.strip()

    # Compare numbers
    response_nums = re.findall(r'-?\d+\.?\d*', extracted)
    truth_nums = re.findall(r'-?\d+\.?\d*', ground_truth)

    if response_nums and truth_nums:
        try:
            if float(response_nums[-1]) == float(truth_nums[-1]):
                return 1.0
        except ValueError:
            pass

    # Partial credit for containing the answer
    if ground_truth.strip().lower() in response.lower():
        return 0.5

    return 0.0


def format_reward(response: str, ground_truth: str) -> float:
    """
    Reward for following the expected output format.
    Checks for <reasoning> and <answer> tags.
    """
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

    if has_reasoning and has_answer:
        return 1.0
    elif has_reasoning or has_answer:
        return 0.5
    return 0.0


def combined_reward(response: str, ground_truth: str) -> float:
    """
    Combined reward: 70% correctness + 30% format.
    This encourages both getting the right answer AND showing reasoning.
    """
    return 0.7 * correctness_reward(response, ground_truth) + \
           0.3 * format_reward(response, ground_truth)


def main():
    print("=" * 70)
    print("GRPO Reasoning Training — End-to-End (DeepSeek R1 Style)")
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

    # 2. Math reasoning dataset (GSM8K style)
    reasoning_data = [
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 15 + 27?", "answer": "42"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 8 * 7?", "answer": "56"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 100 - 37?", "answer": "63"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 144 / 12?", "answer": "12"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nIf a train travels at 60 mph for 2 hours, how far does it go?", "answer": "120"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 25% of 80?", "answer": "20"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nA store has 45 apples. If 18 are sold, how many remain?", "answer": "27"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is 2^5?", "answer": "32"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nIf x + 7 = 15, what is x?", "answer": "8"},
        {"prompt": f"{SYSTEM_PROMPT}\n\nWhat is the sum of 11, 22, and 33?", "answer": "66"},
    ]

    # 3. Configure GRPO
    config = GRPOConfig(
        loss_type="grpo",
        beta=0.04,                    # KL coefficient
        num_generations=2,            # Generate 2 completions per prompt
        temperature=0.7,              # Sampling temperature
        max_completion_length=128,    # Max tokens per completion
        learning_rate=1e-6,
        max_steps=10,                 # 10 training steps
        logging_steps=1,              # Log every step
        output_dir="./grpo_output",
    )

    # 4. Create trainer with custom reward function
    trainer = GRPOTrainer(
        model=model,
        train_dataset=reasoning_data,
        tokenizer=tokenizer,
        reward_fn=combined_reward,
        args=config,
    )

    print(f"\nReward function: 70% correctness + 30% format")
    print(f"Expected format: <reasoning>...</reasoning><answer>...</answer>")

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
