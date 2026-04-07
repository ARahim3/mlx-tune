"""
Example 45: Continual Pretraining - Code Capabilities

Extend a model's programming knowledge by training on code and documentation.
CPT on code teaches the model new APIs, frameworks, and coding patterns.

Use Cases:
- Teaching a model a specific framework (FastAPI, SwiftUI, MLX)
- Adapting to internal codebase patterns
- Specializing in a programming language
- Learning domain-specific DSLs

NOTE: Uses small inline dataset for demo.
"""

from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
from datasets import Dataset


def main():
    print("=" * 70)
    print("Continual Pretraining - Code Capabilities (MLX Framework)")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Base Model
    # ========================================================================
    print("\n[Step 1] Loading base model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/SmolLM2-135M-Instruct",
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
    # Step 3: Prepare Code Corpus
    # ========================================================================
    print("\n[Step 3] Preparing code corpus (MLX framework)...")

    code_corpus = [
        {"text": """# MLX Array Operations
import mlx.core as mx

# Creating arrays
x = mx.array([1, 2, 3, 4, 5])
y = mx.zeros((3, 4))
z = mx.ones((2, 3), dtype=mx.float16)
r = mx.random.normal((5, 5))

# Array operations are lazy - computed only when evaluated
result = mx.matmul(x.reshape(1, -1), mx.ones((5, 3)))
mx.eval(result)  # Forces computation

# MLX uses unified memory on Apple Silicon
# Arrays live in shared CPU/GPU memory - no transfers needed
"""},
        {"text": """# MLX Neural Network Module
import mlx.core as mx
import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dims, num_heads)
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.ffn = nn.Sequential(
            nn.Linear(dims, 4 * dims),
            nn.GELU(),
            nn.Linear(4 * dims, dims),
        )

    def __call__(self, x, mask=None):
        h = self.norm1(x)
        h = self.attention(h, h, h, mask=mask)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h
"""},
        {"text": """# MLX Training Loop Pattern
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def train_step(model, optimizer, x, y):
    def loss_fn(model, x, y):
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction='mean')
        return loss

    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss

# Create model and optimizer
model = MyModel()
optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, optimizer, batch['input'], batch['target'])
        print(f"Loss: {loss.item():.4f}")
"""},
        {"text": """# MLX LoRA Fine-Tuning
from mlx_lm import load, generate
from mlx_lm.tuner.utils import linear_to_lora_layers

# Load a pretrained model
model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")

# Freeze base model, then apply LoRA
model.freeze()
lora_config = {
    "rank": 16,
    "scale": 1.0,  # alpha / rank
    "dropout": 0.0,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"],
}
linear_to_lora_layers(model, num_layers=32, config=lora_config)

# Only LoRA parameters are trainable now
trainable = [k for k, v in model.trainable_parameters()]
print(f"Trainable: {len(trainable)} LoRA parameter tensors")
"""},
        {"text": """# MLX Image Processing for Vision Models
import mlx.core as mx

def preprocess_image(image_path, target_size=(224, 224)):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    # Convert to MLX array and normalize
    pixels = mx.array(np.array(img, dtype=np.float32)) / 255.0

    # ImageNet normalization
    mean = mx.array([0.485, 0.456, 0.406])
    std = mx.array([0.229, 0.224, 0.225])
    pixels = (pixels - mean) / std

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    return pixels[None]
"""},
        {"text": """# MLX Optimizers and Learning Rate Schedules
import mlx.optimizers as optim

# Cosine decay schedule
lr_schedule = optim.cosine_decay(
    init=1e-4,      # Initial learning rate
    decay_steps=1000  # Steps for full cosine cycle
)

# Linear warmup + cosine decay
warmup = optim.linear_schedule(init=0, end=1e-4, steps=100)
decay = optim.cosine_decay(init=1e-4, decay_steps=900)
schedule = optim.join_schedules([warmup, decay], [100])

# Create optimizer with schedule
optimizer = optim.AdamW(
    learning_rate=schedule,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)
"""},
    ]

    dataset = Dataset.from_list(code_corpus)
    print(f"Dataset: {len(dataset)} code documents")

    # ========================================================================
    # Step 4: Train with CPT
    # ========================================================================
    print("\n[Step 4] Starting CPT on code corpus...")

    config = CPTConfig(
        learning_rate=5e-5,
        embedding_learning_rate=1e-5,
        include_embeddings=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        logging_steps=5,
        output_dir="outputs_cpt_code",
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
    # Step 5: Test Code Generation
    # ========================================================================
    print("\n[Step 5] Testing code completion...")

    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompts = [
        "import mlx.core as mx\n\n# Create a random matrix and compute",
        "# MLX training loop\ndef train_step(model, optimizer",
    ]

    for prompt in prompts:
        response = generate(
            model.model, tokenizer,
            prompt=prompt,
            max_tokens=80,
            verbose=False,
        )
        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Generated: {response[:120]}")

    print("\nCode CPT complete!")
    print("\nFor production, train on:")
    print("  - Full framework source code and documentation")
    print("  - GitHub repos with high-quality code")
    print("  - Stack Overflow Q&A for the target domain")


if __name__ == "__main__":
    main()
