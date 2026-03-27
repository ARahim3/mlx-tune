"""
Example 23: ORPO (Odds Ratio Preference Optimization) Training

End-to-end ORPO training. ORPO combines SFT and preference learning into
a single training step, making it simpler and more memory-efficient than DPO.
No reference model is needed.

Usage:
    python examples/23_orpo_preference_tuning.py
"""

from mlx_tune import FastLanguageModel, ORPOTrainer, ORPOConfig


def main():
    print("=" * 70)
    print("ORPO Preference Tuning — End-to-End")
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

    # 2. Preference dataset — same format as DPO
    preference_data = [
        {
            "prompt": "Summarize the benefits of version control.",
            "chosen": "Version control tracks changes to code over time, enables collaboration through branching and merging, provides a complete history for rollbacks, and makes code review possible through pull requests. Git is the most widely used system.",
            "rejected": "it saves your code",
        },
        {
            "prompt": "What is Docker and why is it useful?",
            "chosen": "Docker is a platform for containerizing applications. Containers package your app with all its dependencies into a portable unit that runs consistently across environments. This solves the 'works on my machine' problem and simplifies deployment, scaling, and CI/CD workflows.",
            "rejected": "its like a virtual machine but different",
        },
        {
            "prompt": "Explain the difference between HTTP and HTTPS.",
            "chosen": "HTTP (HyperText Transfer Protocol) sends data in plain text, while HTTPS adds TLS/SSL encryption to protect data in transit. HTTPS prevents eavesdropping and tampering, which is critical for login pages, payments, and any sensitive data. Modern browsers flag HTTP sites as 'Not Secure'.",
            "rejected": "https has an s",
        },
        {
            "prompt": "What is a database index?",
            "chosen": "A database index is a data structure (usually a B-tree) that speeds up data retrieval by providing quick lookup paths to rows. Like a book's index, it maps values to their locations. Indexes speed up SELECT queries but slow down INSERT/UPDATE operations since the index must be maintained.",
            "rejected": "it makes things faster",
        },
        {
            "prompt": "Explain REST API design principles.",
            "chosen": "REST APIs follow key principles: use HTTP methods (GET, POST, PUT, DELETE) for CRUD operations, organize resources with clear URL paths (/users/123), return appropriate status codes (200, 404, 500), support stateless requests, and use JSON for data exchange. Versioning (v1/v2) ensures backward compatibility.",
            "rejected": "rest means the api rests between calls",
        },
        {
            "prompt": "What is the purpose of unit testing?",
            "chosen": "Unit testing verifies individual components work correctly in isolation. Benefits include catching bugs early, enabling safe refactoring, serving as documentation, and providing confidence in code changes. Good unit tests are fast, independent, and cover both happy paths and edge cases.",
            "rejected": "to test units",
        },
        {
            "prompt": "Explain Big O notation.",
            "chosen": "Big O notation describes how an algorithm's runtime or space grows relative to input size. O(1) is constant time, O(n) is linear, O(n^2) is quadratic, O(log n) is logarithmic. It helps compare algorithms — for example, binary search O(log n) is much faster than linear search O(n) for large inputs.",
            "rejected": "its a math thing for algorithms",
        },
        {
            "prompt": "What is CSS Flexbox?",
            "chosen": "CSS Flexbox is a layout model for arranging items in one dimension (row or column). Key properties include display:flex on the container, justify-content for main axis alignment, align-items for cross axis, flex-wrap for wrapping, and flex-grow/shrink on children for sizing. It replaced many float-based hacks.",
            "rejected": "it makes things flexible",
        },
        {
            "prompt": "How does garbage collection work?",
            "chosen": "Garbage collection automatically frees memory that is no longer referenced by the program. Common approaches include reference counting (track how many pointers reference an object) and mark-and-sweep (traverse from roots, mark reachable objects, sweep unreachable ones). Languages like Python, Java, and Go use GC; C/C++ require manual memory management.",
            "rejected": "the computer cleans up after itself",
        },
        {
            "prompt": "What is a hash table?",
            "chosen": "A hash table maps keys to values using a hash function that converts keys to array indices. Lookups, insertions, and deletions average O(1) time. Collisions (two keys mapping to the same index) are handled via chaining (linked lists) or open addressing. Python's dict and JavaScript's objects are hash tables.",
            "rejected": "its a table with hashes",
        },
        {
            "prompt": "Explain the CAP theorem.",
            "chosen": "The CAP theorem states that a distributed system can guarantee at most two of three properties: Consistency (all nodes see the same data), Availability (every request gets a response), and Partition Tolerance (system works despite network failures). Since network partitions are inevitable, real systems choose between CP (e.g., MongoDB) or AP (e.g., Cassandra).",
            "rejected": "cap stands for something in databases",
        },
        {
            "prompt": "What is WebSocket?",
            "chosen": "WebSocket is a protocol that provides full-duplex communication over a single TCP connection. Unlike HTTP (request-response), WebSocket allows the server to push data to clients in real time. It starts with an HTTP upgrade handshake, then maintains a persistent connection. Used for chat apps, live feeds, and gaming.",
            "rejected": "its a socket on the web",
        },
    ]

    # 3. Configure ORPO
    config = ORPOConfig(
        beta=0.1,                    # Odds ratio coefficient
        learning_rate=8e-6,          # Higher LR than DPO (ORPO can use higher)
        max_steps=30,                # 30 training steps
        logging_steps=5,             # Log every 5 steps
        output_dir="./orpo_output",
    )

    # 4. Create trainer
    trainer = ORPOTrainer(
        model=model,
        train_dataset=preference_data,
        tokenizer=tokenizer,
        args=config,
    )

    print("\nORPO combines SFT + preference alignment in one step")
    print("More memory efficient than DPO (no reference model)")

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
