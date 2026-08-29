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

BENCH_TASKS = [
    # ARC Challenge & Science
    {"name": "arc_mineral", "prompt": "Question: Which property of a mineral can be determined by scratching it with a copper penny?\nChoices:\nA. luster\nB. hardness\nC. streak\nD. cleavage\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "arc_photosynthesis", "prompt": "Question: What is the primary pigment used by green plants to absorb sunlight during photosynthesis?\nChoices:\nA. Hemoglobin\nB. Chlorophyll\nC. Melanin\nD. Carotenoid\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "arc_gravity", "prompt": "Question: If an object is taken from the Earth to the Moon, how do its mass and weight change?\nChoices:\nA. Both mass and weight decrease\nB. Mass decreases, weight stays the same\nC. Mass stays the same, weight decreases\nD. Both mass and weight stay the same\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 2},
    {"name": "arc_sound", "prompt": "Question: In which of the following media does sound travel the fastest?\nChoices:\nA. Air\nB. Water\nC. Steel\nD. Vacuum\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 2},
    {"name": "arc_cells", "prompt": "Question: Which organelle is primarily responsible for ATP production through cellular respiration?\nChoices:\nA. Ribosome\nB. Mitochondrion\nC. Golgi apparatus\nD. Lysosome\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},

    # MMLU Computer Science & Engineering
    {"name": "mmlu_cla_adder", "prompt": "In digital logic design, what is the primary advantage of a carry-lookahead adder over a ripple-carry adder?\nChoices:\nA. It uses fewer logic gates\nB. It reduces propagation delay for carry computation\nC. It operates without a clock signal\nD. It consumes zero dynamic power\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "mmlu_dram_refresh", "prompt": "Why do DRAM cells require periodic refresh operations while SRAM cells do not?\nChoices:\nA. DRAM uses magnetic storage\nB. DRAM stores charge on leaky capacitors\nC. DRAM uses optical flip-flops\nD. DRAM is read-only\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "mmlu_cache_coherence", "prompt": "In shared-memory multicore architectures, which protocol ensures cache lines remain consistent across cores?\nChoices:\nA. MESI protocol\nB. TCP/IP protocol\nC. RSA protocol\nD. Round-robin protocol\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 0},
    {"name": "mmlu_asymptotic", "prompt": "What is the average-case time complexity of standard quicksort with random pivot selection on an array of size n?\nChoices:\nA. O(n)\nB. O(n log n)\nC. O(n^2)\nD. O(log n)\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "mmlu_systolic", "prompt": "In a 2D systolic array for matrix multiplication, what flows across adjacent processing elements on each cycle?\nChoices:\nA. Instructions only\nB. Partial sums and activation/weight inputs\nC. Interrupt vectors\nD. DRAM row addresses\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},

    # Math & Quantitative Reasoning (GSM8K / Math)
    {"name": "gsm_bakery", "prompt": "Janet sells 16 cookies for $2 each and 10 muffins for $3 each. How much money did Janet earn in total?\nChoices:\nA. $52\nB. $62\nC. $72\nD. $82\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "math_speed", "prompt": "A train travels at 60 miles per hour for 2.5 hours. How far does it travel?\nChoices:\nA. 120 miles\nB. 130 miles\nC. 150 miles\nD. 180 miles\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 2},
    {"name": "math_prime", "prompt": "Which of the following numbers is a prime number?\nChoices:\nA. 21\nB. 27\nC. 29\nD. 35\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 2},
    {"name": "math_geometry", "prompt": "What is the area of a right triangle with base 8 cm and height 5 cm?\nChoices:\nA. 13 cm^2\nB. 20 cm^2\nC. 40 cm^2\nD. 80 cm^2\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "math_probability", "prompt": "What is the probability of rolling a sum of 7 with two fair 6-sided dice?\nChoices:\nA. 1/12\nB. 1/6\nC. 1/4\nD. 7/36\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},

    # HellaSwag & Common Sense Situational Reasoning
    {"name": "hella_orchestra", "prompt": "A person is playing a cello on stage with an orchestra. The conductor raises their baton and\nChoices:\nA. the musician begins bowing the cello with precision\nB. jumps off the stage into the audience\nC. starts cooking a meal\nD. plays video games\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 0},
    {"name": "hella_kitchen", "prompt": "A chef chops onions on a wooden board and then heats olive oil in a pan. Next, the chef\nChoices:\nA. throws the pan out the window\nB. slides the diced onions into the pan to saute them\nC. replaces the pan with a soccer ball\nD. turns off all lights and goes to sleep\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "hella_bicycle", "prompt": "A cyclist notices their front tire is completely flat before a race. To fix it, the cyclist\nChoices:\nA. puts salt on the pedals\nB. replaces or inflates the inner tube with a pump\nC. paints the handlebars blue\nD. rides backwards\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "hella_reading", "prompt": "A student sits at a library desk with an open textbook and a notebook. The student\nChoices:\nA. takes notes with a pen while studying the text\nB. tears all pages into confetti\nC. sings opera into a megaphone\nD. pours water on the computer keyboard\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 0},
    {"name": "hella_gardening", "prompt": "A gardener digs a hole in the soil, places a young tomato plant inside, and\nChoices:\nA. sets the plant on fire\nB. fills the hole with dirt and waters it thoroughly\nC. covers the plant in concrete\nD. pulls the roots out\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},

    # Logic, Language & Humanities
    {"name": "logic_syllogism", "prompt": "Premise 1: All humans are mortal.\nPremise 2: Socrates is human.\nConclusion:\nChoices:\nA. Socrates is immortal\nB. Socrates is mortal\nC. All mortals are Socrates\nD. No conclusion can be drawn\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "lang_antonym", "prompt": "What is the direct antonym of the word 'ephemeral'?\nChoices:\nA. Transient\nB. Permanent\nC. Fleeting\nD. Delicate\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "lang_capital", "prompt": "What is the capital city of Japan?\nChoices:\nA. Kyoto\nB. Osaka\nC. Tokyo\nD. Hiroshima\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 2},
    {"name": "physics_newton", "prompt": "According to Newton's Second Law of Motion, force is equal to mass multiplied by what?\nChoices:\nA. Velocity\nB. Acceleration\nC. Energy\nD. Distance\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
    {"name": "chem_water", "prompt": "What is the chemical formula for water?\nChoices:\nA. CO2\nB. H2O\nC. NaCl\nD. CH4\nAnswer:", "choices": [" A", " B", " C", " D"], "correct": 1},
]


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


def run_benchmark_suite(model, tok, device: str,
                        chat: bool = False) -> tuple[float, int, int]:
    """Returns (accuracy %, correct, n) over the benchmark tasks."""
    correct = 0
    for task in BENCH_TASKS:
        correct += evaluate_task(model, tok, task, device, chat)
    n = len(BENCH_TASKS)
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
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {a.model} on {a.device} for downstream benchmark suite...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device).eval()

    print("\n--- Evaluating BF16 baseline accuracy ---")
    t0 = time.time()
    bf16_acc, bf16_k, n = run_benchmark_suite(model, tok, a.device, a.chat)
    bf16_ci = wilson95(bf16_k, n)
    print(f"  BF16 accuracy: {bf16_acc:.1f}%  95% CI [{bf16_ci[0]:.1f}, "
          f"{bf16_ci[1]:.1f}]  ({bf16_k}/{n}, {time.time()-t0:.1f}s)")

    print(f"\n--- Applying recipe via pack={a.pack} ---")
    table = dict(quant.BLOCK_FMT)
    gates.quantize_(model, table, mode=a.pack)

    print("\n--- Evaluating quantized model accuracy ---")
    t1 = time.time()
    quant_acc, quant_k, _ = run_benchmark_suite(model, tok, a.device, a.chat)
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
