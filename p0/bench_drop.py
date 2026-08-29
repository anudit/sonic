#!/usr/bin/env python3
"""P0-6: Downstream benchmark drop harness.

Measures the accuracy drop across multiple-choice reasoning and knowledge tasks
between the BF16 baseline and the fake-quantized chip recipe.

Gated at:
    bench_drop_max: 1.5% accuracy drop

Usage:
    python3 p0/bench_drop.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic import quant
from p0 import gates

LETTERS = ["A", "B", "C", "D", "E"]


def _format_task(name: str, question: str, choice_texts: list[str], correct_idx: int) -> dict:
    lines = [f"Question: {question}", "Choices:"]
    for i, c in enumerate(choice_texts):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer:")
    return {
        "name": name,
        "prompt": "\n".join(lines),
        "choices": [f" {LETTERS[i]}" for i in range(len(choice_texts))],
        "correct": correct_idx,
    }


def load_real_tasks(n: int, seed: int = 0) -> list[dict]:
    """n real multiple-choice items from ARC-Challenge (test) + MMLU (test),
    split evenly, deterministically sampled. Replaces the 25 hand-written
    questions HANDOFF.md flagged as statistically unusable (BF16 inside the
    chance band at n=25) -- see T1.2."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    n_arc = n // 2
    n_mmlu = n - n_arc
    tasks: list[dict] = []

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc = arc.filter(lambda r: len(r["choices"]["text"]) == 4)
    idx = rng.choice(len(arc), size=min(n_arc, len(arc)), replace=False)
    for i in idx:
        r = arc[int(i)]
        try:
            correct = r["choices"]["label"].index(r["answerKey"])
        except ValueError:
            continue
        tasks.append(_format_task(f"arc_{r['id']}", r["question"], r["choices"]["text"], correct))

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = rng.choice(len(mmlu), size=min(n_mmlu, len(mmlu)), replace=False)
    for i in idx:
        r = mmlu[int(i)]
        tasks.append(_format_task(f"mmlu_{r['subject']}_{int(i)}", r["question"], r["choices"], r["answer"]))

    return tasks


@torch.no_grad()
def evaluate_task(model, tok, task: dict, device: str,
                  chat: bool = False) -> int:
    """Evaluate one multiple choice question, returning 1 if correct else 0."""
    prompt = task["prompt"]
    choices = task["choices"]
    correct_idx = task["correct"]

    # LFM2.5-8B-A1B is instruction-tuned. Scored as a raw completion it answers
    # at chance on this suite; scored through its own chat template it does not.
    # The template is what the model was trained to see, so --chat is the honest
    # configuration and plain completion is the ablation.
    if chat and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(text, return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)
    else:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    out = model(prompt_ids)
    last_logits = out.logits[0, -1, :]  # [V]

    choice_scores = []
    for c in choices:
        ids = tok(c, add_special_tokens=False).input_ids
        choice_scores.append(last_logits[ids[0]].item())

    pred = int(np.argmax(choice_scores))
    return 1 if pred == correct_idx else 0


def wilson95(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval on a proportion, in percent.

    Reported because n is 25. A drop of 12 points is four questions, and four
    questions out of 25 is well inside the noise of the measurement -- quoting
    it as a measured degradation without the interval overstates it by a wide
    margin.
    """
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def run_benchmark_suite(model, tok, device: str, tasks: list[dict],
                        chat: bool = False) -> tuple[float, int, int]:
    """Returns (accuracy %, correct, n) over the benchmark tasks."""
    correct = 0
    for task in tasks:
        correct += evaluate_task(model, tok, task, device, chat)
    n = len(tasks)
    return (correct / n) * 100.0, correct, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=gates.MODEL)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--pack", choices=["rtn", "clip", "awq", "gptq"], default="rtn")
    ap.add_argument("--out", type=Path, default=Path("p0/out/bench_drop.json"))
    ap.add_argument("--chat", action="store_true",
                    help="score through the model's chat template (see "
                         "evaluate_task); the instruct model answers at chance "
                         "without it, which makes any drop unmeasurable")
    ap.add_argument("--n-tasks", type=int, default=200,
                    help="real ARC-Challenge + MMLU items, split evenly (T1.2: "
                         "n=25 hand-written questions was statistically unusable)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Sampling {a.n_tasks} real items from ARC-Challenge + MMLU (seed={a.seed})...", flush=True)
    tasks = load_real_tasks(a.n_tasks, seed=a.seed)
    print(f"  {len(tasks)} tasks loaded")

    print(f"Loading {a.model} on {a.device} for downstream benchmark suite...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device).eval()

    print("\n--- Evaluating BF16 baseline accuracy ---")
    t0 = time.time()
    bf16_acc, bf16_k, n = run_benchmark_suite(model, tok, a.device, tasks, a.chat)
    bf16_ci = wilson95(bf16_k, n)
    print(f"  BF16 accuracy: {bf16_acc:.1f}%  95% CI [{bf16_ci[0]:.1f}, "
          f"{bf16_ci[1]:.1f}]  ({bf16_k}/{n}, {time.time()-t0:.1f}s)")

    print(f"\n--- Applying recipe via pack={a.pack} ---")
    table = dict(quant.BLOCK_FMT)
    gates.quantize_(model, table, mode=a.pack)

    print("\n--- Evaluating quantized model accuracy ---")
    t1 = time.time()
    quant_acc, quant_k, _ = run_benchmark_suite(model, tok, a.device, tasks, a.chat)
    quant_ci = wilson95(quant_k, n)
    print(f"  Quantized accuracy: {quant_acc:.1f}%  95% CI [{quant_ci[0]:.1f}, "
          f"{quant_ci[1]:.1f}]  ({quant_k}/{n}, {time.time()-t1:.1f}s)")

    drop = max(0.0, bf16_acc - quant_acc)
    gate = quant.GATES["bench_drop_max"]
    passed = drop <= gate

    print("\n" + "=" * 60)
    print(f"{'Metric':25s} {'Measured':>10s} {'Gate':>10s}   Status")
    print("-" * 60)
    print(f"{'bench_drop (%)':25s} {drop:10.2f}% {gate:9.2f}%   {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    # A drop is only meaningful if the BF16 anchor is meaningful. At n=25 with
    # four choices, chance is 25% and its upper 95% bound is ~44%: a baseline
    # inside that band cannot support ANY statement about degradation, because
    # the reference itself is indistinguishable from guessing.
    chance_hi = wilson95(n // 4, n)[1]
    usable = bf16_acc > chance_hi
    if not usable:
        print(f"\n  WARNING: BF16 baseline {bf16_acc:.1f}% is within the 95% "
              f"band of chance for 4-way choice at n={n} (upper bound "
              f"{chance_hi:.1f}%).\n  The drop below is NOT a measurement of "
              f"quantization: it is noise around a reference that is itself\n"
              f"  guessing. Fix the prompting (try --chat) or enlarge the suite "
              f"before quoting it.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "bf16_acc": bf16_acc,
        "bf16_ci95": bf16_ci,
        "quant_acc": quant_acc,
        "quant_ci95": quant_ci,
        "n_tasks": n,
        "chat_template": bool(a.chat),
        "bench_drop": drop,
        "gate": gate,
        "passed": passed,
        "baseline_usable": bool(usable),
    }, indent=2))
    print(f"\nWrote results to {a.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
