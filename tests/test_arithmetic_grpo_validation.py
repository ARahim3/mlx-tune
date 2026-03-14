from pathlib import Path

from mlx_tune.arithmetic_grpo_validation import (
    ArithmeticSolutionReward,
    generate_benchmark_splits,
    parse_solution_response,
    run_baseline,
    run_compare,
    run_training,
    score_solution_output,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        prompt = "\n".join(message["content"] for message in messages)
        return prompt + ("\nassistant:" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 256 for char in text]


class FakeModel:
    def __init__(self):
        self.trained = False
        self.loaded_adapter = None

    def load_adapter(self, path):
        self.loaded_adapter = path
        self.trained = True

    def generate(self, prompt, max_tokens, sampler=None, verbose=False):
        del max_tokens, sampler, verbose
        expression = prompt.split("Compute exactly:\n", 1)[1].split("\nassistant:", 1)[0].strip()
        answer = str(int(eval(expression, {"__builtins__": {}}, {})))
        if self.trained:
            return f"<think>\nchecking\n</think>\n<solution>{answer}</solution>"
        return answer


class FakeTrainer:
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, reward_fn, args):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.args = args

    def train(self):
        self.model.trained = True
        policy_dir = Path(self.args.output_dir) / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "adapters.safetensors").write_text("fake")
        (policy_dir / "adapter_config.json").write_text("{}")
        return {"status": "success", "global_step": self.args.max_steps}


def _fake_load_model_bundle(*args, **kwargs):
    del args, kwargs
    return FakeModel(), FakeTokenizer()


def test_generate_benchmark_splits_is_deterministic_and_disjoint(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    generate_benchmark_splits(first_dir, train_size=12, val_size=4, test_size=4, seed=7)
    generate_benchmark_splits(second_dir, train_size=12, val_size=4, test_size=4, seed=7)

    first_paths = {split: (first_dir / "datasets" / f"{split}.jsonl").read_text() for split in ("train", "val", "test")}
    second_paths = {split: (second_dir / "datasets" / f"{split}.jsonl").read_text() for split in ("train", "val", "test")}

    assert first_paths == second_paths

    expressions = {}
    for split in ("train", "val", "test"):
        rows = [line for line in first_paths[split].strip().splitlines() if line]
        expressions[split] = {__import__("json").loads(line)["expression"] for line in rows}

    assert expressions["train"].isdisjoint(expressions["val"])
    assert expressions["train"].isdisjoint(expressions["test"])
    assert expressions["val"].isdisjoint(expressions["test"])


def test_parse_and_score_solution_output():
    parsed = parse_solution_response("<think>x</think><solution>\n42\n</solution>")
    assert parsed["single_solution_tag"] is True
    assert parsed["parseable_solution"] is True
    assert parsed["parsed_answer"] == 42

    exact = score_solution_output("<solution>42</solution>", "42")
    assert exact["exact_match"] is True
    assert exact["reward"] == 1.1

    wrong = score_solution_output("<solution>41</solution>", "42")
    assert wrong["exact_match"] is False
    assert wrong["reward"] == 0.1

    missing = score_solution_output("42", "42")
    assert missing["single_solution_tag"] is False
    assert missing["reward"] == 0.0

    multiple = score_solution_output("<solution>41</solution><solution>42</solution>", "42")
    assert multiple["multiple_solution_tags"] is True
    assert multiple["reward"] == 0.0


def test_reward_evaluator_returns_components():
    evaluator = ArithmeticSolutionReward()
    result = evaluator.evaluate({"completion_text": "<solution>7</solution>", "reward_context": "7"})
    assert result["reward"] == 1.1
    assert result["components"]["solution_tag"] == 0.1
    assert result["components"]["correctness"] == 1.0


def test_baseline_train_and_compare_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr("mlx_tune.arithmetic_grpo_validation.load_model_bundle", _fake_load_model_bundle)
    monkeypatch.setattr("mlx_tune.arithmetic_grpo_validation.GRPOTrainer", FakeTrainer)
    monkeypatch.setattr("mlx_tune.arithmetic_grpo_validation.FastLanguageModel.for_inference", lambda model: model)

    generate_benchmark_splits(tmp_path, train_size=8, val_size=3, test_size=3, seed=0)

    baseline = run_baseline(
        tmp_path,
        model_name="fake-qwen",
        seed=0,
        max_seq_length=64,
        max_completion_length=32,
    )
    assert baseline["aggregate"]["exact_match"] == 0.0
    assert (tmp_path / "baseline_outputs.jsonl").exists()
    assert (tmp_path / "baseline_metrics.json").exists()

    trained = run_training(
        tmp_path,
        model_name="fake-qwen",
        seed=0,
        max_seq_length=64,
        max_completion_length=32,
        max_steps=3,
        learning_rate=1e-6,
        per_device_train_batch_size=2,
        rollout_batch_size=2,
        num_generations=2,
        rl_temperature=0.9,
        lora_rank=4,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
    )
    assert trained["training"]["status"] == "success"
    assert trained["post_eval"]["aggregate"]["exact_match"] == 1.0
    assert (tmp_path / "post_rl_outputs.jsonl").exists()
    assert (tmp_path / "post_rl_metrics.json").exists()

    comparison = run_compare(tmp_path)
    assert comparison["aggregate_delta"]["exact_match"] > 0.0
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
