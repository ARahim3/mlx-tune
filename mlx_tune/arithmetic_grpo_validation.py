"""
Deterministic arithmetic benchmark and GRPO validation for native-thinking Qwen 3 models.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mlx_lm.sample_utils import make_sampler

from mlx_tune import (
    FastLanguageModel,
    GRPOConfig,
    GRPOTrainer,
    get_chat_template,
)


DEFAULT_MODEL_NAME = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful arithmetic solver. Think freely if useful. "
    "Put the final integer answer inside <solution>...</solution>."
)
DEFAULT_OUTPUT_DIR = Path("./artifacts/qwen3_arithmetic_grpo_validation")
DEFAULT_RL_SUBDIR = "rl_run"
DEFAULT_TRAIN_SIZE = 3000
DEFAULT_VAL_SIZE = 300
DEFAULT_TEST_SIZE = 300
DEFAULT_SEED = 0
DEFAULT_MAX_COMPLETION_LENGTH = 512
DEFAULT_MAX_SEQ_LENGTH = 768
DEFAULT_LORA_RANK = 16
DEFAULT_RL_TEMPERATURE = 0.9
DEFAULT_BASELINE_TEMPERATURE = 0.0

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dataset_dir(output_dir: Path) -> Path:
    return output_dir / "datasets"


def _dataset_path(output_dir: Path, split: str) -> Path:
    return _dataset_dir(output_dir) / f"{split}.jsonl"


def _rl_output_dir(output_dir: Path) -> Path:
    return output_dir / DEFAULT_RL_SUBDIR


def _rl_adapter_dir(output_dir: Path) -> Path:
    return _rl_output_dir(output_dir) / "policy"


def _baseline_outputs_path(output_dir: Path) -> Path:
    return output_dir / "baseline_outputs.jsonl"


def _baseline_metrics_path(output_dir: Path) -> Path:
    return output_dir / "baseline_metrics.json"


def _post_outputs_path(output_dir: Path) -> Path:
    return output_dir / "post_rl_outputs.jsonl"


def _post_metrics_path(output_dir: Path) -> Path:
    return output_dir / "post_rl_metrics.json"


def _comparison_json_path(output_dir: Path) -> Path:
    return output_dir / "comparison.json"


def _comparison_md_path(output_dir: Path) -> Path:
    return output_dir / "comparison.md"


def _training_summary_path(output_dir: Path) -> Path:
    return output_dir / "rl_training_summary.json"


def _safe_eval(expression: str) -> int:
    return int(eval(expression, {"__builtins__": {}}, {}))


def _make_easy_expression(rng: random.Random) -> Tuple[str, int]:
    left = rng.randint(0, 50)
    right = rng.randint(0, 50)
    operator = rng.choice(["+", "-"])
    expression = f"{left} {operator} {right}"
    return expression, _safe_eval(expression)


def _make_medium_expression(rng: random.Random) -> Tuple[str, int]:
    values = [rng.randint(0, 20) for _ in range(3)]
    operators = [rng.choice(["+", "-", "*"]) for _ in range(2)]
    mode = rng.choice(["plain", "left_paren", "right_paren"])
    if mode == "left_paren":
        expression = f"({values[0]} {operators[0]} {values[1]}) {operators[1]} {values[2]}"
    elif mode == "right_paren":
        expression = f"{values[0]} {operators[0]} ({values[1]} {operators[1]} {values[2]})"
    else:
        expression = f"{values[0]} {operators[0]} {values[1]} {operators[1]} {values[2]}"
    return expression, _safe_eval(expression)


def build_sample(task_id: str, expression: str, answer: int, difficulty: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Compute exactly:\n{expression}"},
        ],
        "answer": str(answer),
        "expression": expression,
        "difficulty": difficulty,
    }


def generate_benchmark_splits(
    output_dir: Path,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = {
        "train": _dataset_path(output_dir, "train"),
        "val": _dataset_path(output_dir, "val"),
        "test": _dataset_path(output_dir, "test"),
    }
    if not force and all(path.exists() for path in dataset_paths.values()):
        return dataset_paths

    rng = random.Random(seed)
    existing_expressions = set()
    split_sizes = {"train": train_size, "val": val_size, "test": test_size}

    for split_name, split_size in split_sizes.items():
        rows: List[Dict[str, Any]] = []
        while len(rows) < split_size:
            difficulty = "easy" if (len(rows) % 2 == 0) else "medium"
            if difficulty == "easy":
                expression, answer = _make_easy_expression(rng)
            else:
                expression, answer = _make_medium_expression(rng)
            if expression in existing_expressions:
                continue
            if abs(answer) > 999:
                continue
            existing_expressions.add(expression)
            task_id = f"{split_name}-{len(rows) + 1:06d}"
            rows.append(build_sample(task_id, expression, answer, difficulty))
        _write_jsonl(dataset_paths[split_name], rows)

    return dataset_paths


def parse_solution_response(response_text: str) -> Dict[str, Any]:
    matches = _SOLUTION_RE.findall(response_text)
    single_solution_tag = len(matches) == 1
    solution_text = matches[0].strip() if single_solution_tag else None
    parseable_solution = bool(solution_text is not None and _INTEGER_RE.fullmatch(solution_text))
    parsed_answer = int(solution_text) if parseable_solution else None
    return {
        "single_solution_tag": single_solution_tag,
        "multiple_solution_tags": len(matches) > 1,
        "solution_text": solution_text,
        "parseable_solution": parseable_solution,
        "parsed_answer": parsed_answer,
    }


def score_solution_output(response_text: str, gold_answer: str) -> Dict[str, Any]:
    parsed = parse_solution_response(response_text)
    exact_match = parsed["parseable_solution"] and str(parsed["parsed_answer"]) == str(gold_answer).strip()
    tag_reward = 0.1 if parsed["single_solution_tag"] else 0.0
    correctness_reward = 1.0 if exact_match else 0.0
    reward = tag_reward + correctness_reward
    return {
        **parsed,
        "exact_match": exact_match,
        "tag_reward": tag_reward,
        "correctness_reward": correctness_reward,
        "reward": reward,
    }


class ArithmeticSolutionReward:
    def evaluate(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        scored = score_solution_output(
            str(payload.get("completion_text", "")),
            str(payload.get("reward_context", "")),
        )
        return {
            "reward": scored["reward"],
            "components": {
                "solution_tag": scored["tag_reward"],
                "correctness": scored["correctness_reward"],
            },
            "diagnostics": {
                "solution_text": scored["solution_text"],
                "parseable_solution": scored["parseable_solution"],
                "multiple_solution_tags": scored["multiple_solution_tags"],
                "exact_match": scored["exact_match"],
            },
        }


def _token_length(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def load_model_bundle(
    model_name: str,
    *,
    max_seq_length: int,
    load_adapter_path: Optional[Path] = None,
    for_training: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
) -> Tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            max_seq_length=max_seq_length,
        )
    if load_adapter_path is not None and load_adapter_path.exists():
        model.load_adapter(str(load_adapter_path))
    if not for_training:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_completion(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=False,
    )
    return model.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
    )


def _load_split_records(output_dir: Path, split: str) -> List[Dict[str, Any]]:
    path = _dataset_path(output_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return _read_jsonl(path)


def _metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "split": split,
            "model_name": model_name,
            "seed": seed,
            "num_samples": 0,
            "exact_match": 0.0,
            "solution_tag_rate": 0.0,
            "parseable_solution_rate": 0.0,
            "multiple_solution_tag_rate": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_reward": 0.0,
        }
    return {
        "split": split,
        "model_name": model_name,
        "seed": seed,
        "num_samples": count,
        "exact_match": sum(1.0 for row in rows if row["exact_match"]) / count,
        "solution_tag_rate": sum(1.0 for row in rows if row["single_solution_tag"]) / count,
        "parseable_solution_rate": sum(1.0 for row in rows if row["parseable_solution"]) / count,
        "multiple_solution_tag_rate": sum(1.0 for row in rows if row["multiple_solution_tags"]) / count,
        "avg_completion_tokens": sum(float(row["completion_tokens"]) for row in rows) / count,
        "avg_reward": sum(float(row["reward"]) for row in rows) / count,
    }


def _aggregate_metrics(
    per_split: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    all_rows = sum(int(metrics["num_samples"]) for metrics in per_split.values())
    if all_rows == 0:
        return _metrics_from_rows([], split="aggregate", model_name=model_name, seed=seed)
    weighted = {}
    for field in (
        "exact_match",
        "solution_tag_rate",
        "parseable_solution_rate",
        "multiple_solution_tag_rate",
        "avg_completion_tokens",
        "avg_reward",
    ):
        weighted[field] = sum(
            float(metrics[field]) * int(metrics["num_samples"]) for metrics in per_split.values()
        ) / all_rows
    return {
        "split": "aggregate",
        "model_name": model_name,
        "seed": seed,
        "num_samples": all_rows,
        **weighted,
    }


def evaluate_records(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    model_name: str,
    seed: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []
    for row in rows:
        response_text = generate_completion(
            model,
            tokenizer,
            row["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        scored = score_solution_output(response_text, str(row["answer"]))
        evaluated.append(
            {
                "task_id": row["task_id"],
                "split": split,
                "expression": row["expression"],
                "difficulty": row["difficulty"],
                "answer": row["answer"],
                "completion_text": response_text,
                "completion_tokens": _token_length(tokenizer, response_text),
                **scored,
            }
        )
    return evaluated, _metrics_from_rows(evaluated, split=split, model_name=model_name, seed=seed)


def _run_eval(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    model_name: str,
    seed: int,
    output_path: Path,
    metrics_path: Path,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    all_outputs: List[Dict[str, Any]] = []
    per_split: Dict[str, Dict[str, Any]] = {}
    for split in ("val", "test"):
        rows = _load_split_records(output_dir, split)
        split_outputs, split_metrics = evaluate_records(
            model,
            tokenizer,
            rows,
            split=split,
            model_name=model_name,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        all_outputs.extend(split_outputs)
        per_split[split] = split_metrics
    _write_jsonl(output_path, all_outputs)
    payload = {
        "model_name": model_name,
        "seed": seed,
        "splits": per_split,
        "aggregate": _aggregate_metrics(per_split, model_name=model_name, seed=seed),
    }
    _write_json(metrics_path, payload)
    return payload


def run_baseline(
    output_dir: Path,
    *,
    model_name: str,
    seed: int,
    max_seq_length: int,
    max_completion_length: int,
) -> Dict[str, Any]:
    model, tokenizer = load_model_bundle(
        model_name,
        max_seq_length=max_seq_length,
    )
    return _run_eval(
        model,
        tokenizer,
        output_dir,
        model_name=model_name,
        seed=seed,
        output_path=_baseline_outputs_path(output_dir),
        metrics_path=_baseline_metrics_path(output_dir),
        max_tokens=max_completion_length,
        temperature=DEFAULT_BASELINE_TEMPERATURE,
    )


def run_training(
    output_dir: Path,
    *,
    model_name: str,
    seed: int,
    max_seq_length: int,
    max_completion_length: int,
    max_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    rollout_batch_size: int,
    num_generations: int,
    rl_temperature: float,
    lora_rank: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
) -> Dict[str, Any]:
    train_rows = _load_split_records(output_dir, "train")
    val_rows = _load_split_records(output_dir, "val")
    model, tokenizer = load_model_bundle(
        model_name,
        max_seq_length=max_seq_length,
        for_training=True,
        lora_rank=lora_rank,
    )
    reward = ArithmeticSolutionReward()
    config = GRPOConfig(
        loss_type="grpo",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        rollout_batch_size=rollout_batch_size,
        num_generations=num_generations,
        temperature=rl_temperature,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        max_completion_length=max_completion_length,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        reward_source="online",
        reward_normalization="none",
        mask_truncated_completions=True,
        output_dir=str(_rl_output_dir(output_dir)),
        seed=seed,
    )
    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        tokenizer=tokenizer,
        reward_fn=reward,
        args=config,
    )
    result = trainer.train()
    _write_json(_training_summary_path(output_dir), result)
    FastLanguageModel.for_inference(model)
    post_metrics = _run_eval(
        model,
        tokenizer,
        output_dir,
        model_name=model_name,
        seed=seed,
        output_path=_post_outputs_path(output_dir),
        metrics_path=_post_metrics_path(output_dir),
        max_tokens=max_completion_length,
        temperature=DEFAULT_BASELINE_TEMPERATURE,
    )
    return {"training": result, "post_eval": post_metrics}


def _metric_delta(baseline: Mapping[str, Any], post: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(post[key]) - float(baseline[key])
        for key in (
            "exact_match",
            "solution_tag_rate",
            "parseable_solution_rate",
            "multiple_solution_tag_rate",
            "avg_completion_tokens",
            "avg_reward",
        )
    }


def _load_eval_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (row["split"], row["task_id"]): row
        for row in _read_jsonl(path)
    }


def _pair_examples(
    baseline_rows: Mapping[Tuple[str, str], Mapping[str, Any]],
    post_rows: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    improvements: List[Dict[str, Any]] = []
    regressions: List[Dict[str, Any]] = []
    unchanged_failures: List[Dict[str, Any]] = []
    for key in sorted(set(baseline_rows) & set(post_rows)):
        baseline = baseline_rows[key]
        post = post_rows[key]
        pair = {
            "split": key[0],
            "task_id": key[1],
            "expression": baseline["expression"],
            "answer": baseline["answer"],
            "baseline_completion_text": baseline["completion_text"],
            "post_completion_text": post["completion_text"],
            "baseline_reward": baseline["reward"],
            "post_reward": post["reward"],
            "baseline_exact_match": baseline["exact_match"],
            "post_exact_match": post["exact_match"],
            "baseline_single_solution_tag": baseline["single_solution_tag"],
            "post_single_solution_tag": post["single_solution_tag"],
        }
        if float(post["reward"]) > float(baseline["reward"]):
            improvements.append(pair)
        elif float(post["reward"]) < float(baseline["reward"]):
            regressions.append(pair)
        elif not baseline["exact_match"] and not post["exact_match"]:
            unchanged_failures.append(pair)
    return improvements, regressions, unchanged_failures


def _render_examples_md(title: str, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [f"### {title}", ""]
    if not rows:
        lines.append("_None_")
        lines.append("")
        return lines
    for row in rows:
        lines.extend(
            [
                f"- `{row['split']}/{row['task_id']}` `{row['expression']}` -> `{row['answer']}`",
                f"  baseline: {row['baseline_completion_text']}",
                f"  post_rl: {row['post_completion_text']}",
                "",
            ]
        )
    return lines


def run_compare(output_dir: Path) -> Dict[str, Any]:
    baseline_metrics = _read_json(_baseline_metrics_path(output_dir))
    post_metrics = _read_json(_post_metrics_path(output_dir))
    comparison = {
        "baseline": baseline_metrics,
        "post_rl": post_metrics,
        "delta": {
            split: _metric_delta(baseline_metrics["splits"][split], post_metrics["splits"][split])
            for split in ("val", "test")
        },
        "aggregate_delta": _metric_delta(
            baseline_metrics["aggregate"],
            post_metrics["aggregate"],
        ),
    }
    baseline_rows = _load_eval_rows(_baseline_outputs_path(output_dir))
    post_rows = _load_eval_rows(_post_outputs_path(output_dir))
    improvements, regressions, unchanged_failures = _pair_examples(baseline_rows, post_rows)
    comparison["sample_counts"] = {
        "improvements": len(improvements),
        "regressions": len(regressions),
        "unchanged_failures": len(unchanged_failures),
    }
    _write_json(_comparison_json_path(output_dir), comparison)

    lines = [
        "# Qwen3 Arithmetic GRPO Comparison",
        "",
        "| Split | Baseline Exact | Post Exact | Delta Exact | Baseline Reward | Post Reward | Delta Reward | Baseline Tag | Post Tag | Delta Tag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("val", "test"):
        baseline = baseline_metrics["splits"][split]
        post = post_metrics["splits"][split]
        delta = comparison["delta"][split]
        lines.append(
            "| {split} | {b_exact:.4f} | {p_exact:.4f} | {d_exact:+.4f} | "
            "{b_reward:.4f} | {p_reward:.4f} | {d_reward:+.4f} | "
            "{b_tag:.4f} | {p_tag:.4f} | {d_tag:+.4f} |".format(
                split=split,
                b_exact=baseline["exact_match"],
                p_exact=post["exact_match"],
                d_exact=delta["exact_match"],
                b_reward=baseline["avg_reward"],
                p_reward=post["avg_reward"],
                d_reward=delta["avg_reward"],
                b_tag=baseline["solution_tag_rate"],
                p_tag=post["solution_tag_rate"],
                d_tag=delta["solution_tag_rate"],
            )
        )
    aggregate_delta = comparison["aggregate_delta"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "GRPO appears to work on this benchmark."
                if aggregate_delta["avg_reward"] > 0 or aggregate_delta["solution_tag_rate"] > 0
                else "GRPO did not show a positive held-out signal on this run."
            ),
            "",
        ]
    )
    lines.extend(_render_examples_md("Improvements", improvements[:10]))
    lines.extend(_render_examples_md("Regressions", regressions[:5]))
    lines.extend(_render_examples_md("Unchanged Failures", unchanged_failures[:5]))
    _comparison_md_path(output_dir).write_text("\n".join(lines))
    return comparison


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "baseline", "train", "compare", "all"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--max-completion-length", type=int, default=DEFAULT_MAX_COMPLETION_LENGTH)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--rl-temperature", type=float, default=DEFAULT_RL_TEMPERATURE)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    generate_benchmark_splits(
        args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        force=args.force_generate,
    )
    if args.mode == "generate":
        return 0
    if args.mode == "baseline":
        run_baseline(
            args.output_dir,
            model_name=args.model_name,
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            max_completion_length=args.max_completion_length,
        )
        return 0
    if args.mode == "train":
        run_training(
            args.output_dir,
            model_name=args.model_name,
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            max_completion_length=args.max_completion_length,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            rollout_batch_size=args.rollout_batch_size,
            num_generations=args.num_generations,
            rl_temperature=args.rl_temperature,
            lora_rank=args.lora_rank,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
        )
        return 0
    if args.mode == "compare":
        run_compare(args.output_dir)
        return 0
    run_baseline(
        args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
    )
    run_training(
        args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        num_generations=args.num_generations,
        rl_temperature=args.rl_temperature,
        lora_rank=args.lora_rank,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
    )
    run_compare(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
