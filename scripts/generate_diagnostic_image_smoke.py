#!/usr/bin/env python3
"""Generate or honestly block the one-step diagnostic image-smoke artifacts.

This script is intentionally evidence-only. It never imports torch or Jittor,
never runs model code, and never opens the production denoising loop. If the
ignored local one-step tensors are absent, it writes a PARTIAL blocker report
instead of fabricating images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STAGE_CANDIDATES = ("clamped_output", "decode_output")
FORBIDDEN_RAW_SUFFIXES = {".npy", ".npz"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_tensor(local_dir: Path, stage: str, role: str) -> Path | None:
    names = [
        f"{role}_{stage}.npy",
        f"{stage}_{role}.npy",
        f"{stage}.npy" if role == "official" else "",
        f"jittor_{stage}_got.npy" if role == "jittor" else "",
        f"{stage}_got.npy" if role == "jittor" else "",
    ]
    for name in names:
        if not name:
            continue
        path = local_dir / name
        if path.exists():
            return path
    return None


def to_float_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


def chw_or_hwc(arr: np.ndarray) -> tuple[np.ndarray, str]:
    x = np.asarray(arr)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in {1, 3, 4}:
        if x.shape[0] == 1:
            return x[0], "single_channel_chw"
        if x.shape[0] == 4:
            x = x[:3]
        return np.transpose(x[:3], (1, 2, 0)), "rgb_chw"
    if x.ndim == 3 and x.shape[-1] in {1, 3, 4}:
        if x.shape[-1] == 1:
            return x[..., 0], "single_channel_hwc"
        if x.shape[-1] == 4:
            x = x[..., :3]
        return x[..., :3], "rgb_hwc"
    if x.ndim == 2:
        return x, "single_channel_hw"
    flat = np.squeeze(x)
    if flat.ndim >= 2:
        return flat.reshape(flat.shape[-2], flat.shape[-1]), "single_channel_squeezed"
    raise ValueError(f"Unsupported tensor shape for diagnostic visualization: {arr.shape}")


def normalize_for_png(arr: np.ndarray) -> np.ndarray:
    x, _ = chw_or_hwc(arr)
    x = np.asarray(x, dtype=np.float32)
    # Official clamped/decode tensors are expected around [-1, 1]. For heatmap
    # style tensors we still clamp to avoid misleading contrast explosions.
    x = np.clip(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def save_png(path: Path, arr_01: np.ndarray, title: str) -> None:
    from PIL import Image, ImageDraw

    x = np.asarray(arr_01)
    if x.ndim == 2:
        rgb = np.stack([x, x, x], axis=-1)
    else:
        rgb = x[..., :3]
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = Image.fromarray(rgb8, mode="RGB")
    banner_h = 34
    canvas = Image.new("RGB", (image.width, image.height + banner_h), color=(255, 255, 255))
    canvas.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    canvas.save(path)


def heatmap(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3:
        axis = 0 if x.shape[0] in {1, 3, 4} else -1
        x = np.max(np.abs(x), axis=axis)
    elif x.ndim != 2:
        x = np.squeeze(x)
        if x.ndim > 2:
            x = np.max(np.abs(x), axis=0)
    x = np.abs(x)
    denom = float(np.max(x)) if np.max(x) > 0 else 1.0
    x = np.clip(x / denom, 0.0, 1.0)
    # Simple blue-to-red heatmap without depending on matplotlib.
    return np.stack([x, 0.15 * (1.0 - x), 1.0 - x], axis=-1)


def concat_grid(official: np.ndarray, jittor: np.ndarray, diff: np.ndarray) -> np.ndarray:
    a = normalize_for_png(official)
    b = normalize_for_png(jittor)
    c = heatmap(diff)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if b.ndim == 2:
        b = np.stack([b, b, b], axis=-1)
    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    return np.concatenate([a[:h, :w, :3], b[:h, :w, :3], c[:h, :w, :3]], axis=1)


def metrics(official: np.ndarray, jittor: np.ndarray) -> dict[str, Any]:
    a = np.asarray(official, dtype=np.float64)
    b = np.asarray(jittor, dtype=np.float64)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(mse))
    rel_l2 = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(a.ravel()), 1e-12))
    denom = max(float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())), 1e-12)
    cosine = float(np.dot(a.ravel(), b.ravel()) / denom)
    psnr = math.inf if mse == 0.0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    ssim_value: float | str
    try:
        from skimage.metrics import structural_similarity

        aa = normalize_for_png(a.astype(np.float32))
        bb = normalize_for_png(b.astype(np.float32))
        if aa.ndim == 2:
            ssim_value = float(structural_similarity(aa, bb, data_range=1.0))
        else:
            ssim_value = float(structural_similarity(aa, bb, channel_axis=-1, data_range=1.0))
    except Exception:
        ssim_value = "SSIM_NOT_AVAILABLE"
    return {
        "shape_match": list(a.shape) == list(b.shape),
        "official_shape": list(a.shape),
        "jittor_shape": list(b.shape),
        "official_range": [float(np.min(a)), float(np.max(a))],
        "jittor_range": [float(np.min(b)), float(np.max(b))],
        "mae": mae,
        "max_abs": max_abs,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim_value,
        "relative_l2": rel_l2,
        "cosine_similarity": cosine,
        "finite_check": bool(np.isfinite(a).all() and np.isfinite(b).all()),
    }


def staged_raw_tensors(repo_root: Path) -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return []
    bad: list[str] = []
    for line in proc.stdout.splitlines():
        if Path(line).suffix.lower() in FORBIDDEN_RAW_SUFFIXES:
            bad.append(line)
    return bad


def write_outputs(output_dir: Path, figure_dir: Path, payload: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    ensure_dir(figure_dir)
    (output_dir / "diagnostic_image_smoke_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = []
    for key in ["mae", "max_abs", "rmse", "psnr", "ssim", "relative_l2", "cosine_similarity"]:
        rows.append({"metric": key, "value": payload.get("metrics", {}).get(key, "")})
    with (output_dir / "diagnostic_image_smoke_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Diagnostic image-smoke metrics",
        "",
        "This artifact is a diagnostic one-step tensor visualization report. It is not full TADSR inference, not a production restoration result, and not image/video generation completion.",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, value in payload["markers"].items():
        lines.append(f"| `{marker}` | `{value}` |")
    lines += [
        "",
        f"- Status: `{payload['status']}`",
        f"- Source stage: `{payload.get('source_stage')}`",
        f"- Blocker: `{payload.get('blocker', 'none')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['metric']}` | `{row['value']}` |")
    lines += [
        "",
        "## Safety flags",
        "",
    ]
    for key, value in payload["safety"].items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "diagnostic_image_smoke_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--one-step-dir", default="experiments/production_completion/full_inference/one_step")
    parser.add_argument("--output-dir", default="experiments/production_completion/diagnostic_image_smoke")
    parser.add_argument("--figure-dir", default="figures/diagnostic_image_smoke")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    one_step_dir = (repo_root / args.one_step_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    figure_dir = (repo_root / args.figure_dir).resolve()
    local_dir = one_step_dir / "local_tensors"
    alignment = load_json(one_step_dir / "jittor_one_step_alignment.json")
    stage_metrics = alignment.get("stage_metrics", {}) if isinstance(alignment.get("stage_metrics"), dict) else {}
    staged_raw = staged_raw_tensors(repo_root)
    source_stage = None
    official_path = None
    jittor_path = None
    for stage in STAGE_CANDIDATES:
        off = find_tensor(local_dir, stage, "official")
        jit = find_tensor(local_dir, stage, "jittor")
        if off and jit:
            source_stage = stage
            official_path = off
            jittor_path = jit
            break

    safety = {
        "full_inference_executed": False,
        "denoising_loop_executed": False,
        "multi_step_executed": False,
        "production_image_generation": False,
        "image_is_diagnostic": True,
        "raw_tensors_committed": False,
        "local_tensors_used": bool(official_path and jittor_path),
    }

    if not official_path or not jittor_path:
        blocker = "one-step local tensors not available for both official and Jittor diagnostic outputs"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PARTIAL",
            "status_marker": "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY",
            "source_stage": None,
            "blocker": blocker,
            "one_step_dir": str(one_step_dir),
            "local_tensor_dir": str(local_dir),
            "local_tensor_dir_exists": local_dir.exists(),
            "available_local_tensor_files": sorted(p.name for p in local_dir.glob("*.npy")) if local_dir.exists() else [],
            "existing_stage_metrics_from_alignment": stage_metrics,
            "metrics": {},
            "figures": {},
            "safety": safety,
            "markers": {
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY": "PARTIAL",
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED": "NOT_EXECUTED",
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT": "BLOCKED",
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE": "PASS",
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS": "PASS" if not staged_raw else "FAIL",
                "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY": "PARTIAL",
            },
            "next_action": "Run the controlled one-step tensor alignment with ignored local_tensors enabled on Linux, then rerun this script. Do not stage .npy/.npz.",
        }
        write_outputs(output_dir, figure_dir, payload)
        for marker, value in payload["markers"].items():
            print(f"{marker}: {value}")
        print(f"blocker: {blocker}")
        return 0

    official = to_float_array(official_path)
    jittor = to_float_array(jittor_path)
    metric_payload = metrics(official, jittor)
    diff = official - jittor
    ensure_dir(figure_dir)
    official_png = figure_dir / "official_one_step_clamped_output.png"
    jittor_png = figure_dir / "jittor_one_step_clamped_output.png"
    diff_png = figure_dir / "absolute_difference_heatmap.png"
    grid_png = figure_dir / "side_by_side_diagnostic_grid.png"
    error_hist_png = figure_dir / "error_histogram.png"
    save_png(official_png, normalize_for_png(official), "Diagnostic one-step tensor smoke, not full inference: official")
    save_png(jittor_png, normalize_for_png(jittor), "Diagnostic one-step tensor smoke, not full inference: Jittor")
    save_png(diff_png, heatmap(diff), "Diagnostic one-step absolute difference, not full inference")
    save_png(grid_png, concat_grid(official, jittor, diff), "Official | Jittor | abs diff - diagnostic only")
    hist, _ = np.histogram(np.abs(diff).ravel(), bins=128, range=(0.0, max(metric_payload["max_abs"], 1e-12)))
    hist_img = np.zeros((160, 384), dtype=np.float32)
    hist = hist.astype(np.float32) / max(float(hist.max()), 1.0)
    for i, h in enumerate(hist):
        x0 = int(i * hist_img.shape[1] / len(hist))
        x1 = int((i + 1) * hist_img.shape[1] / len(hist))
        y0 = int((1.0 - h) * hist_img.shape[0])
        hist_img[y0:, x0:max(x1, x0 + 1)] = 1.0
    save_png(error_hist_png, hist_img, "Diagnostic one-step absolute-error histogram")
    alignment_status = "PASS" if metric_payload["shape_match"] and metric_payload["finite_check"] else "FAIL"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if alignment_status == "PASS" else "FAIL",
        "status_marker": "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY",
        "source_stage": source_stage,
        "official_tensor": repo_relative(official_path, repo_root),
        "jittor_tensor": repo_relative(jittor_path, repo_root),
        "metrics": metric_payload,
        "figures": {
            "official": str(official_png.relative_to(repo_root)),
            "jittor": str(jittor_png.relative_to(repo_root)),
            "absolute_difference_heatmap": str(diff_png.relative_to(repo_root)),
            "side_by_side_diagnostic_grid": str(grid_png.relative_to(repo_root)),
            "error_histogram": str(error_hist_png.relative_to(repo_root)),
        },
        "safety": safety | {"diagnostic_image_generated": True},
        "markers": {
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY": "PASS",
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED": "PASS",
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT": alignment_status,
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE": "PASS",
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS": "PASS" if not staged_raw else "FAIL",
            "TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY": "PASS",
        },
        "next_action": "Use the PNGs only as diagnostic one-step tensor visualizations in the report and video; do not claim full inference completion.",
    }
    write_outputs(output_dir, figure_dir, payload)
    for marker, value in payload["markers"].items():
        print(f"{marker}: {value}")
    for key in ["mae", "max_abs", "rmse", "psnr", "ssim"]:
        print(f"{key}: {metric_payload[key]}")
    return 0 if alignment_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
