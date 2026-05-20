#!/usr/bin/env python3
"""
run_ablation.py
───────────────────────────────────────────────────────────────────────────
Runs the 4-way ablation study from the proposal (RQ1, RQ2, RQ3).

Ablation conditions:
  1. baseline      — WorldVLA single-pass (no GoT)
  2. bon           — Best-of-N over full trajectory (CoT-SC equivalent)
  3. got_no_wm     — GoT temporal decomposition, NO World Model scoring
  4. got           — GoT temporal decomposition + World Model scoring (proposed)

After all runs, prints a comparison table matching the proposal's
experiment design.

Usage:
    python run_ablation.py \\
        --resume_path /path/to/checkpoint \\
        --tokenizer_path /path/to/tokenizer \\
        --task_suite_name libero_spatial \\
        --output_base ./results \\
        --device 0 \\
        --num_trials 10   # use 50 for final results, 10 for quick test
"""

import argparse
import os
import subprocess
import sys
import csv
from pathlib import Path


EVAL_SCRIPT = str(Path(__file__).parent / "eval_solver_libero_got_v2.py")

ABLATION_CONDITIONS = [
    # (name, extra_args)
    ("baseline",    ["--mode", "baseline"]),
    ("bon",         ["--mode", "bon",  "--k_candidates", "3"]),
    ("got_no_wm",   ["--mode", "got",  "--k_candidates", "3", "--no_world_model_scoring"]),
    ("got",         ["--mode", "got",  "--k_candidates", "3"]),
]


def run_condition(name: str, extra_args: list, base_args: list, output_base: str):
    output_dir = str(Path(output_base) / name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--output_dir", output_dir,
    ] + base_args + extra_args

    print(f"\n{'='*60}")
    print(f"[Ablation] Running: {name}")
    print(f"[Ablation] Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode, output_dir


def parse_csv_results(csv_path: str) -> dict:
    """Read results.csv and compute overall success rate."""
    if not os.path.exists(csv_path):
        return {"success_rate": None, "total_episodes": 0, "successes": 0}

    total, successes = 0, 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            successes += int(row.get("success", 0))

    sr = successes / total if total > 0 else 0.0
    return {"success_rate": sr, "total_episodes": total, "successes": successes}


def print_comparison_table(results: dict):
    """Print a formatted comparison table (matches the proposal's Table 3 style)."""
    print("\n" + "=" * 65)
    print("GoT-VLA Ablation Results")
    print("=" * 65)
    print(f"{'Condition':<20} {'SR (%)':<12} {'Episodes':<12} {'Successes'}")
    print("-" * 65)

    for name, stats in results.items():
        sr = stats["success_rate"]
        sr_str = f"{sr*100:.1f}" if sr is not None else "N/A"
        print(
            f"{name:<20} {sr_str:<12} "
            f"{stats['total_episodes']:<12} {stats['successes']}"
        )

    print("=" * 65)

    # Compute improvement over baseline
    baseline_sr = results.get("baseline", {}).get("success_rate")
    if baseline_sr and baseline_sr > 0:
        print("\nImprovement over baseline:")
        for name, stats in results.items():
            if name == "baseline":
                continue
            sr = stats["success_rate"]
            if sr is not None:
                delta = (sr - baseline_sr) / baseline_sr * 100
                print(f"  {name:<18}: {'+' if delta >= 0 else ''}{delta:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="GoT-VLA ablation runner")
    # Pass-through args for the eval script
    parser.add_argument("--resume_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output_base", default="./results/ablation")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--num_trials", default=50, type=int,
                        help="num_trials_per_task (use 10 for quick test)")
    parser.add_argument("--n_segments", default=3, type=int)
    parser.add_argument("--segment_len", default=4, type=int)
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "bon", "got_no_wm", "got"],
                        help="Which conditions to run")
    args = parser.parse_args()

    base_args = [
        "--resume_path", args.resume_path,
        "--tokenizer_path", args.tokenizer_path,
        "--task_suite_name", args.task_suite_name,
        "--device", str(args.device),
        "--resolution", str(args.resolution),
        "--num_trials_per_task", str(args.num_trials),
        "--n_segments", str(args.n_segments),
        "--segment_len", str(args.segment_len),
    ]

    results = {}

    for name, extra_args in ABLATION_CONDITIONS:
        if name not in args.conditions:
            print(f"[Ablation] Skipping: {name}")
            continue

        rc, output_dir = run_condition(name, extra_args, base_args, args.output_base)

        csv_path = str(Path(output_dir) / "results.csv")
        stats = parse_csv_results(csv_path)
        results[name] = stats

        print(f"\n[Ablation] {name}: SR={stats['success_rate']}")

    print_comparison_table(results)

    # Save summary
    summary_path = str(Path(args.output_base) / "ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "success_rate", "total_episodes", "successes"])
        for name, stats in results.items():
            writer.writerow([name, stats["success_rate"],
                             stats["total_episodes"], stats["successes"]])
    print(f"[Ablation] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
