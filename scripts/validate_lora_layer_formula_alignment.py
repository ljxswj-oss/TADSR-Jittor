#!/usr/bin/env python3
"""Validate fixed-adapter LoRA runtime formula vs static effective weight.

This is a synthetic NumPy-only unit-level check. It does not import torch or
jittor, does not load TADSR weights, and does not implement dynamic adapter
switching. A PASS only means that, for a fixed adapter and scale, the runtime
LoRA formula is numerically equivalent to a static effective weight.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments/lora_layer_formula_alignment.json"
OUT_MD = ROOT / "experiments/lora_layer_formula_alignment.md"


def main() -> int:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment fallback
        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status_marker": "TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT",
            "status": "PARTIAL_ACCEPTABLE",
            "reason": f"NumPy unavailable: {exc!r}",
            "dynamic_runtime_adapter_switching_implemented": False,
            "full_tadsr_runtime_lora_integrated": False,
            "tadsr_dynamic_runtime_lora_status": "NOT_IMPLEMENTED_BY_DESIGN",
        }
        write_outputs(result)
        print("TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT: PARTIAL_ACCEPTABLE")
        return 0

    rng = np.random.default_rng(20260609)
    batch, in_dim, out_dim, rank = 7, 11, 13, 5
    scale = 0.75
    x = rng.normal(size=(batch, in_dim)).astype(np.float64)
    w = rng.normal(size=(out_dim, in_dim)).astype(np.float64)
    a = rng.normal(size=(rank, in_dim)).astype(np.float64)
    b = rng.normal(size=(out_dim, rank)).astype(np.float64)

    runtime_y = x @ w.T + scale * ((x @ a.T) @ b.T)
    w_eff = w + scale * (b @ a)
    static_y = x @ w_eff.T
    diff = runtime_y - static_y
    max_abs_error = float(np.max(np.abs(diff)))
    mean_abs_error = float(np.mean(np.abs(diff)))
    relative_l2_error = float(np.linalg.norm(diff) / max(np.linalg.norm(runtime_y), 1e-30))
    denom = max(float(np.linalg.norm(runtime_y) * np.linalg.norm(static_y)), 1e-30)
    cosine_similarity = float(np.sum(runtime_y * static_y) / denom)
    passed = (
        max_abs_error < 1e-10
        and relative_l2_error < 1e-10
        and cosine_similarity > 0.999999999
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_marker": "TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT",
        "status": "PASS" if passed else "FAIL",
        "dtype": "float64",
        "seed": 20260609,
        "shape": {
            "batch": batch,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "rank": rank,
        },
        "scale": scale,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
        "dynamic_runtime_adapter_switching_implemented": False,
        "full_tadsr_runtime_lora_integrated": False,
        "tadsr_dynamic_runtime_lora_status": "NOT_IMPLEMENTED_BY_DESIGN",
        "conclusion": (
            "Fixed adapter / fixed scale runtime LoRA formula matches static "
            "effective weight formula. This does not mean dynamic runtime LoRA "
            "adapter switching is implemented."
        ),
    }
    write_outputs(result)
    print(f"TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT: {result['status']}")
    print(f"max_abs_error: {max_abs_error:.6e}")
    print(f"relative_l2_error: {relative_l2_error:.6e}")
    print(f"cosine_similarity: {cosine_similarity:.12f}")
    return 0 if passed else 1


def write_outputs(result: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# LoRA Layer Formula Alignment",
        "",
        f"Status marker: `{result['status_marker']}`",
        "",
        f"Status: **{result['status']}**",
        "",
        "本检查只验证固定 adapter / 固定 scale 下 runtime LoRA 公式与 static effective weight 公式等价。",
        "",
        "它不表示 dynamic runtime LoRA adapter switching 已实现，也不接入 full TADSR inference。",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| max_abs_error | {result.get('max_abs_error', '')} |",
        f"| mean_abs_error | {result.get('mean_abs_error', '')} |",
        f"| relative_l2_error | {result.get('relative_l2_error', '')} |",
        f"| cosine_similarity | {result.get('cosine_similarity', '')} |",
        "",
        "## Scope",
        "",
        f"- dynamic_runtime_adapter_switching_implemented: `{result['dynamic_runtime_adapter_switching_implemented']}`",
        f"- full_tadsr_runtime_lora_integrated: `{result['full_tadsr_runtime_lora_integrated']}`",
        f"- tadsr_dynamic_runtime_lora_status: `{result['tadsr_dynamic_runtime_lora_status']}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
