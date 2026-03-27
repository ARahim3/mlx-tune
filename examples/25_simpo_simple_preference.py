"""
Example 25: SimPO (Simple Preference Optimization) Training

End-to-end SimPO training. SimPO simplifies DPO by removing the need for a
reference model. It uses length-normalized log probabilities as implicit
rewards, making training simpler and more memory-efficient.

Usage:
    python examples/25_simpo_simple_preference.py
"""

from mlx_tune import FastLanguageModel, SimPOTrainer, SimPOConfig


def main():
    print("=" * 70)
    print("SimPO Simple Preference Training — End-to-End")
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
            "prompt": "What is a linked list?",
            "chosen": "A linked list is a data structure where each element (node) contains data and a pointer to the next node. Unlike arrays, elements aren't stored contiguously in memory. This makes insertions and deletions O(1) at known positions, but random access is O(n) since you must traverse from the head.",
            "rejected": "its a list that is linked",
        },
        {
            "prompt": "Explain the concept of inheritance in OOP.",
            "chosen": "Inheritance allows a class (child/subclass) to inherit properties and methods from another class (parent/superclass). This promotes code reuse — for example, a Dog class can inherit from Animal and add its own bark() method while getting walk() and eat() from Animal.",
            "rejected": "when one class gets stuff from another class",
        },
        {
            "prompt": "What is a deadlock in concurrent programming?",
            "chosen": "A deadlock occurs when two or more threads are each waiting for the other to release a resource, creating a circular dependency. Example: Thread A holds Lock 1 and waits for Lock 2, while Thread B holds Lock 2 and waits for Lock 1. Prevention strategies include lock ordering and timeout-based acquisition.",
            "rejected": "when things get stuck",
        },
        {
            "prompt": "Explain what a compiler does.",
            "chosen": "A compiler translates source code written in a high-level language (like C or Rust) into machine code that a CPU can execute. It performs lexical analysis, parsing, semantic analysis, optimization, and code generation. Unlike interpreters, compilers process the entire program before execution.",
            "rejected": "it compiles code",
        },
        {
            "prompt": "What is DNS?",
            "chosen": "DNS (Domain Name System) translates human-readable domain names like google.com into IP addresses like 142.250.80.46. It uses a hierarchical system of nameservers — your request goes from a recursive resolver to root servers, TLD servers, and finally authoritative servers for the domain.",
            "rejected": "internet address lookup",
        },
        {
            "prompt": "Explain what a load balancer does.",
            "chosen": "A load balancer distributes incoming network traffic across multiple servers to ensure no single server is overwhelmed. Common algorithms include round-robin, least connections, and weighted distribution. This improves availability, reliability, and response times for web applications.",
            "rejected": "it balances the load",
        },
        {
            "prompt": "What is a binary search tree?",
            "chosen": "A binary search tree (BST) is a tree where each node's left children are smaller and right children are larger. This ordering enables O(log n) search, insertion, and deletion on balanced trees. However, worst case (when unbalanced/degenerate) is O(n). Self-balancing variants like AVL and Red-Black trees maintain O(log n).",
            "rejected": "a tree with two children",
        },
        {
            "prompt": "Explain microservices architecture.",
            "chosen": "Microservices decomposes an application into small, independently deployable services that communicate via APIs. Each service owns its data and can use different technologies. Benefits: independent scaling and deployment, team autonomy. Challenges: distributed system complexity, network latency, data consistency.",
            "rejected": "small services",
        },
        {
            "prompt": "What is OAuth 2.0?",
            "chosen": "OAuth 2.0 is an authorization framework that lets third-party apps access user resources without sharing passwords. It uses access tokens and supports flows like Authorization Code (web apps), Client Credentials (server-to-server), and PKCE (mobile/SPA). The user grants permissions through a consent screen.",
            "rejected": "login protocol",
        },
        {
            "prompt": "Explain event-driven architecture.",
            "chosen": "Event-driven architecture is a design pattern where components communicate through events (state changes). Producers emit events to a message broker (like Kafka or RabbitMQ), and consumers subscribe to process them asynchronously. This decouples services, improves scalability, and enables real-time processing.",
            "rejected": "things happen when events occur",
        },
        {
            "prompt": "What is memoization?",
            "chosen": "Memoization is an optimization technique that caches function results for previously seen inputs. When called with the same arguments again, it returns the cached result instead of recomputing. This is especially useful for recursive algorithms like Fibonacci, reducing time from O(2^n) to O(n).",
            "rejected": "caching stuff",
        },
        {
            "prompt": "Explain ACID properties in databases.",
            "chosen": "ACID ensures database transaction reliability: Atomicity (all-or-nothing execution), Consistency (data remains valid after transactions), Isolation (concurrent transactions don't interfere), and Durability (committed data survives crashes). These guarantees are critical for financial systems and data integrity.",
            "rejected": "database rules",
        },
    ]

    # 3. Configure SimPO
    config = SimPOConfig(
        beta=2.0,                    # Temperature coefficient (higher than DPO)
        gamma=0.5,                   # Target reward margin
        learning_rate=5e-7,
        max_steps=30,                # 30 training steps
        logging_steps=5,             # Log every 5 steps
        output_dir="./simpo_output",
    )

    # 4. Create trainer
    trainer = SimPOTrainer(
        model=model,
        train_dataset=preference_data,
        tokenizer=tokenizer,
        args=config,
    )

    print("\nSimPO advantages:")
    print("  - No reference model needed (simpler than DPO)")
    print("  - Uses length-normalized log probs as implicit rewards")
    print("  - Memory efficient — only one model in memory")

    # 5. Train!
    result = trainer.train()

    print(f"\nResult: {result['status']}")
    print(f"Adapters saved to: {result['adapter_path']}")


if __name__ == "__main__":
    main()
