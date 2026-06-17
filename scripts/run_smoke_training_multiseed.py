#!/usr/bin/env python3
"""Run multi-seed small-data smoke training stability checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STRICT_PYTHON = Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=ROOT, env=merged, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2025, 42])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--output-root", default="experiments/smoke_training/output_tail/multiseed")
    parser.add_argument("--max-seeds", type=int, default=0, help="Optional safety limit for very slow machines; 0 means all seeds.")
    args = parser.parse_args()

    out_root = ROOT / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds[: args.max_seeds] if args.max_seeds and args.max_seeds > 0 else args.seeds
    rows: list[dict[str, object]] = []

    py = str(STRICT_PYTHON if STRICT_PYTHON.exists() else sys.executable)
    for seed in seeds:
        rel_dir = f"{args.output_root}/seed_{seed}"
        seed_dir = ROOT / rel_dir
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable,
            "tools/export_tadsr_smoke_training_data.py",
            "--num-samples", str(args.num_samples),
            "--train-samples", str(args.train_samples),
            "--val-samples", str(args.val_samples),
            "--seed", str(seed),
            "--output-dir", rel_dir,
        ])
        run([
            py,
            "tools/train_smoke_pytorch_output_tail.py",
            "--data-dir", rel_dir,
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--device", "cpu",
            "--eval-interval", str(args.eval_interval),
        ])
        run([
            sys.executable,
            "scripts/train_smoke_jittor_output_tail.py",
            "--data-dir", rel_dir,
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--use-cuda", "0",
            "--eval-interval", str(args.eval_interval),
        ], env={"USE_CUDA": "0", "nvcc_path": ""})
        run([sys.executable, "scripts/plot_smoke_training_curves.py", "--pytorch-log", f"{rel_dir}/pytorch/loss.csv", "--jittor-log", f"{rel_dir}/jittor/loss.csv", "--output-dir", rel_dir])
        run([sys.executable, "scripts/analyze_smoke_training_alignment.py", "--data-dir", rel_dir])

        p = load_json(seed_dir / "pytorch/final_metrics.json")
        j = load_json(seed_dir / "jittor/final_metrics.json")
        align = load_json(seed_dir / "smoke_training_alignment_metrics.json")
        val_gap = float(align.get("loss_curve_alignment", {}).get("final_val_loss_relative_gap", 999.0))
        pred_l2 = float(align.get("pytorch_vs_jittor_prediction", {}).get("relative_l2_diff", 999.0))
        seed_status = "PASS" if p.get("status") == "PASS" and j.get("status") == "PASS" and align.get("status") == "PASS" else (
            "PARTIAL" if p.get("status") == "PASS" and j.get("status") == "PASS" and align.get("status") in {"PASS", "PARTIAL"} else "FAIL"
        )
        rows.append({
            "seed": seed,
            "pytorch_train_final_loss": p.get("train_final_loss", ""),
            "jittor_train_final_loss": j.get("train_final_loss", ""),
            "pytorch_val_final_loss": p.get("val_final_loss", ""),
            "jittor_val_final_loss": j.get("val_final_loss", ""),
            "pytorch_loss_drop_ratio": p.get("train_loss_drop_ratio", ""),
            "jittor_loss_drop_ratio": j.get("train_loss_drop_ratio", ""),
            "pytorch_val_loss_drop_ratio": p.get("val_loss_drop_ratio", ""),
            "jittor_val_loss_drop_ratio": j.get("val_loss_drop_ratio", ""),
            "val_loss_relative_gap": val_gap,
            "prediction_relative_l2_diff": pred_l2,
            "status": seed_status,
        })

    write_csv(out_root / "multiseed_summary.csv", rows)
    val_losses = [float(r["jittor_val_final_loss"]) for r in rows if r["jittor_val_final_loss"] != ""]
    gaps = [float(r["val_loss_relative_gap"]) for r in rows if r["val_loss_relative_gap"] != ""]
    num_pass = sum(1 for r in rows if r["status"] == "PASS")
    num_partial = sum(1 for r in rows if r["status"] == "PARTIAL")
    num_fail = sum(1 for r in rows if r["status"] == "FAIL")
    status = "PASS" if num_pass == len(rows) and len(rows) == len(args.seeds) else (
        "PARTIAL" if num_fail == 0 and rows else "FAIL"
    )
    summary = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seeds": args.seeds,
        "completed_seeds": seeds,
        "num_pass": num_pass,
        "num_partial": num_partial,
        "num_fail": num_fail,
        "mean_jittor_final_val_loss": float(np.mean(val_losses)) if val_losses else None,
        "std_jittor_final_val_loss": float(np.std(val_losses)) if val_losses else None,
        "mean_val_loss_relative_gap": float(np.mean(gaps)) if gaps else None,
        "std_val_loss_relative_gap": float(np.std(gaps)) if gaps else None,
        "full_tadsr_training": False,
        "full_inference_executed": False,
        "image_saved": False,
        "video_saved": False,
        "rows": rows,
    }
    (out_root / "multiseed_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    for seed in seeds:
        seed_dir = out_root / f"seed_{seed}"
        try:
            pr = list(csv.DictReader((seed_dir / "pytorch/loss.csv").open(newline="", encoding="utf-8")))
            jr = list(csv.DictReader((seed_dir / "jittor/loss.csv").open(newline="", encoding="utf-8")))
            ax.plot([int(r["step"]) for r in pr], [float(r["train_loss"]) for r in pr], alpha=0.65, label=f"PyTorch seed {seed}")
            ax.plot([int(r["step"]) for r in jr], [float(r["train_loss"]) for r in jr], alpha=0.65, linestyle="--", label=f"Jittor seed {seed}")
        except Exception:
            continue
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("Multi-Seed Smoke Training Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "multiseed_loss_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [float(r["pytorch_val_final_loss"]) for r in rows], width=0.36, label="PyTorch val")
    ax.bar(x + 0.18, [float(r["jittor_val_final_loss"]) for r in rows], width=0.36, label="Jittor val")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["seed"]) for r in rows])
    ax.set_ylabel("Final validation loss")
    ax.set_title("Multi-Seed Final Validation Loss")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "multiseed_final_metrics_bar.png")
    plt.close(fig)

    md = [
        "# Multi-Seed Smoke Training Stability",
        "",
        f"Status: **{status}**",
        "",
        "| Seed | PyTorch val final | Jittor val final | Val relative gap | Prediction relative L2 | Status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['seed']} | {float(r['pytorch_val_final_loss']):.8f} | {float(r['jittor_val_final_loss']):.8f} | {float(r['val_loss_relative_gap']):.8e} | {float(r['prediction_relative_l2_diff']):.8e} | {r['status']} |"
        )
    md += [
        "",
        f"- mean_jittor_final_val_loss: {summary['mean_jittor_final_val_loss']}",
        f"- std_jittor_final_val_loss: {summary['std_jittor_final_val_loss']}",
        f"- mean_val_loss_relative_gap: {summary['mean_val_loss_relative_gap']}",
        "",
        "This is a small-data output-tail training-loop stability check, not full TADSR training.",
    ]
    (out_root / "multiseed_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"TADSR_SMOKE_TRAINING_MULTI_SEED_STABILITY: {status}")
    print(f"completed_seeds: {seeds}")
    print(f"num_pass: {num_pass}")
    print(f"num_partial: {num_partial}")
    print(f"num_fail: {num_fail}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
