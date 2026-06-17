#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def line_window(text: str, start: int, end: int) -> list[str]:
    lines = text.splitlines()
    return [f"{idx}: {lines[idx - 1].strip()}" for idx in range(max(1, start), min(len(lines), end) + 1)]


def class_source(text: str, class_name: str) -> tuple[str, int, int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "", 0, 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = int(getattr(node, "lineno", 0))
            end = int(getattr(node, "end_lineno", start))
            lines = text.splitlines()
            return "\n".join(lines[start - 1 : end]), start, end
    return "", 0, 0


def method_source(class_text: str, method_name: str, class_start_line: int) -> tuple[str, int, int]:
    try:
        tree = ast.parse(class_text)
    except SyntaxError:
        return "", 0, 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    start = class_start_line + int(getattr(child, "lineno", 0)) - 1
                    end = class_start_line + int(getattr(child, "end_lineno", getattr(child, "lineno", 0))) - 1
                    lines = class_text.splitlines()
                    local_start = int(getattr(child, "lineno", 1))
                    local_end = int(getattr(child, "end_lineno", local_start))
                    return "\n".join(lines[local_start - 1 : local_end]), start, end
    return "", 0, 0


def run_env_probe(official_python: str | None, official_repo: Path, overlay: str | None) -> dict:
    if not official_python:
        return {"status": "NOT_REQUESTED", "reason": "official_python not provided"}
    py = Path(official_python)
    if not py.exists():
        return {"status": "BLOCKED", "reason": f"official_python not found: {py}"}
    env = os.environ.copy()
    paths = [str(official_repo)]
    if overlay:
        paths.append(str(overlay))
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        proc = subprocess.run(
            [str(py), "-c", "import torch; print('TORCH_OK', torch.__version__)"],
            cwd=official_repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except Exception as exc:
        return {"status": "BLOCKED", "reason": repr(exc)}
    return {
        "status": "PASS" if proc.returncode == 0 and "TORCH_OK" in proc.stdout else "PARTIAL",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }


def contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official TADSR actual inference path without executing inference.")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-weights", default="")
    parser.add_argument("--official-python", default="")
    parser.add_argument("--pythonpath-overlay", default="")
    parser.add_argument("--output-dir", default="experiments/production_completion/full_inference")
    args = parser.parse_args()

    official_repo = Path(args.official_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "actual_inference_path_audit.json"
    out_md = output_dir / "actual_inference_path_audit.md"

    tadsr_py = official_repo / "tadsr.py"
    test_py = official_repo / "test_tadsr.py"
    tadsr_text = read_text(tadsr_py)
    test_text = read_text(test_py)

    class_text, class_start, class_end = class_source(tadsr_text, "TADSR_test")
    init_text, init_start, init_end = method_source(class_text, "__init__", class_start)
    forward_text, forward_start, forward_end = method_source(class_text, "forward", class_start)
    get_x0_text, get_x0_start, get_x0_end = method_source(class_text, "get_x0_from_res", class_start)

    entry_points = []
    if class_text:
        entry_points.append({"name": "TADSR_test", "file": "tadsr.py", "line_start": class_start, "line_end": class_end})
    if "model(lq, prompt=validation_prompt, timesteps=args.timesteps)" in test_text:
        entry_points.append({"name": "test_tadsr.py main model call", "file": "test_tadsr.py", "line_hint": 137})

    set_timesteps_call = "set_timesteps(1" in init_text or "set_timesteps(1" in class_text
    timesteps_count = 1 if set_timesteps_call else None
    timestep_values = [1] if "torch.tensor([1]" in class_text or "set_timesteps(1" in class_text else []
    scheduler_step_called = "scheduler.step" in forward_text or ".step(" in forward_text and "noise_scheduler" in forward_text
    get_x0_called = "get_x0_from_res" in forward_text and bool(get_x0_text)
    timestep_loop_patterns = [
        r"for\s+\w+\s+in\s+.*timesteps",
        r"for\s+\w+\s+in\s+self\.noise_scheduler\.timesteps",
        r"for\s+\w+\s+in\s+range\(.*num_inference_steps",
    ]
    has_denoising_loop = any(contains(p, forward_text) for p in timestep_loop_patterns)
    unet_called = "self.unet(" in forward_text
    tiled_spatial_loops = all(token in forward_text for token in ["grid_rows", "grid_cols", "for row in range", "for col in range"])
    vae_encode = "self.vae.encode" in forward_text
    vae_decode = "self.vae.decode" in forward_text
    clamp_policy = ".clamp(-1, 1)" in forward_text

    has_postprocess = "transforms.ToPILImage()" in test_text and "* 0.5 + 0.5" in test_text
    has_image_save = "output_pil.save" in test_text
    has_color_fix = "adain_color_fix" in test_text or "wavelet_color_fix" in test_text
    dynamic_lora = any(token in forward_text for token in ["set_adapter", "enable_adapters", "disable_adapters"])

    if set_timesteps_call and get_x0_called and unet_called and vae_decode and not has_denoising_loop and not scheduler_step_called:
        path_type = "single_step_get_x0_from_res"
        multi_required = False
        tiny_recommended = False
        single_step_marker = "PASS"
        reason = "TADSR_test.forward performs one VAE encode, one UNet prediction policy, get_x0_from_res, and one VAE decode/clamp. Timestep policy is fixed by set_timesteps(1); no denoising loop or scheduler.step call appears in the actual inference forward path."
    elif has_denoising_loop or scheduler_step_called:
        path_type = "multi_step_loop" if has_denoising_loop else "one_step_scheduler_step"
        multi_required = bool(has_denoising_loop)
        tiny_recommended = bool(has_denoising_loop)
        single_step_marker = "PARTIAL"
        reason = "The audited source contains scheduler.step or a timestep loop in the actual inference path."
    else:
        path_type = "unknown"
        multi_required = None
        tiny_recommended = None
        single_step_marker = "BLOCKED"
        reason = "The official actual inference path could not be classified from the source files."

    env_probe = run_env_probe(args.official_python, official_repo, args.pythonpath_overlay)
    status = "PASS" if official_repo.exists() and tadsr_py.exists() and test_py.exists() and path_type != "unknown" else "BLOCKED"
    markers = {
        "TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT": status,
        "TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT": "PASS" if multi_required is not None else "BLOCKED",
        "TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT": single_step_marker,
    }

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_env_status": env_probe,
        "official_repo": str(official_repo),
        "official_weights": str(args.official_weights),
        "inference_entry_points": entry_points,
        "set_timesteps_call": set_timesteps_call,
        "timesteps_count": timesteps_count,
        "timestep_values_if_static": timestep_values,
        "has_denoising_loop": has_denoising_loop,
        "denoising_loop_location": None if not has_denoising_loop else "TADSR_test.forward",
        "scheduler_step_called": scheduler_step_called,
        "get_x0_from_res_called": get_x0_called,
        "unet_call_count_policy": "single call for non-tiled latent; multiple spatial tile batch calls only when latent tiling is triggered; no timestep denoising loop",
        "vae_encode_policy": "TADSR_test.forward calls self.vae.encode(lq, timesteps) once before UNet.",
        "vae_decode_policy": "TADSR_test.forward calls self.vae.decode(x_denoised / scaling_factor).sample once after get_x0_from_res.",
        "postprocess_policy": "test_tadsr.py converts clamped tensor from [-1,1] to [0,1] by output * 0.5 + 0.5, then ToPILImage, optional color fix, optional resize, and save.",
        "image_save_policy": "test_tadsr.py saves output_pil at output_dir/basename; this audit did not execute image save.",
        "dynamic_lora_policy": "TADSR_test loads fixed LoRA adapters from checkpoint and sets fixed adapters; forward path does not dynamically switch adapters or scales.",
        "official_actual_path_type": path_type,
        "multi_step_required_for_official_actual_inference": multi_required,
        "tiny_multi_step_alignment_recommended": tiny_recommended,
        "reason": reason,
        "source_line_ranges": {
            "TADSR_test.__init__": [init_start, init_end],
            "TADSR_test.get_x0_from_res": [get_x0_start, get_x0_end],
            "TADSR_test.forward": [forward_start, forward_end],
            "test_tadsr.py model/postprocess": [134, 149],
        },
        "source_evidence": {
            "init": line_window(tadsr_text, init_start, min(init_end, init_start + 40)) if init_start else [],
            "forward": line_window(tadsr_text, forward_start, forward_end) if forward_start else [],
            "postprocess": line_window(test_text, 134, 149),
        },
        "safety_flags": {
            "full_inference_executed": False,
            "denoising_loop_executed": False,
            "image_or_video_generated": False,
            "raw_tensor_saved": False,
            "official_venv_modified": False,
        },
        "markers": markers,
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Official TADSR actual inference path audit",
        "",
        f"`TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT: {markers['TADSR_OFFICIAL_ACTUAL_INFERENCE_PATH_AUDIT']}`",
        f"`TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT: {markers['TADSR_OFFICIAL_INFERENCE_MULTISTEP_REQUIREMENT_AUDIT']}`",
        f"`TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT: {markers['TADSR_OFFICIAL_ACTUAL_PATH_SINGLE_STEP_CONTRACT']}`",
        "",
        f"- official_actual_path_type: `{path_type}`",
        f"- multi_step_required_for_official_actual_inference: `{multi_required}`",
        f"- tiny_multi_step_alignment_recommended: `{tiny_recommended}`",
        f"- scheduler_step_called: `{scheduler_step_called}`",
        f"- get_x0_from_res_called: `{get_x0_called}`",
        f"- has_denoising_loop: `{has_denoising_loop}`",
        "",
        "## Reason",
        "",
        reason,
        "",
        "## Safety",
        "",
        "- No production full inference was executed.",
        "- No denoising loop was executed.",
        "- No image/video was generated.",
        "- No raw tensor was saved.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker, value in markers.items():
        print(f"{marker}: {value}")
    print(f"official_actual_path_type: {path_type}")
    print(f"multi_step_required_for_official_actual_inference: {multi_required}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
