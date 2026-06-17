#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def run_env_probe(official_python: str | None, official_repo: Path, overlay: str | None) -> dict:
    if not official_python:
        return {"status": "NOT_REQUESTED"}
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
            [str(py), "-c", "from torchvision import transforms; from PIL import Image; print('POSTPROCESS_IMPORT_OK')"],
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
        "status": "PASS" if proc.returncode == 0 and "POSTPROCESS_IMPORT_OK" in proc.stdout else "PARTIAL",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official TADSR postprocess/output contract without saving images.")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--official-weights", default="")
    parser.add_argument("--official-python", default="")
    parser.add_argument("--pythonpath-overlay", default="")
    parser.add_argument("--output-dir", default="experiments/production_completion/full_inference")
    args = parser.parse_args()

    official_repo = Path(args.official_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "postprocess_contract_audit.json"
    out_md = output_dir / "postprocess_contract_audit.md"

    tadsr_text = read_text(official_repo / "tadsr.py")
    test_text = read_text(official_repo / "test_tadsr.py")
    clamp_policy = ".clamp(-1, 1)" in tadsr_text
    tensor_to_image_policy = "output_image[0].cpu() * 0.5 + 0.5" in test_text and "transforms.ToPILImage()" in test_text
    color_fix_policy = "adain_color_fix" in test_text and "wavelet_color_fix" in test_text
    resize_policy = "resize_flag" in test_text and "input_image.resize" in test_text
    save_policy = "output_pil.save" in test_text
    rgb_policy = ".convert('RGB')" in test_text or '.convert("RGB")' in test_text
    dtype_policy = "mixed_precision" in test_text and "weight_dtype" in test_text
    env_probe = run_env_probe(args.official_python, official_repo, args.pythonpath_overlay)

    status = "PASS" if clamp_policy and tensor_to_image_policy and save_policy else "PARTIAL"
    markers = {
        "TADSR_POSTPROCESS_CONTRACT_AUDIT": status,
        "TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT": "PASS" if save_policy else "PARTIAL",
        "TADSR_POSTPROCESS_NOT_EXECUTED_GUARD": "PASS",
    }
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_env_status": env_probe,
        "decoded_tensor_range": "TADSR_test.forward clamps decoded tensor to [-1, 1]." if clamp_policy else "not found",
        "clamp_policy": {"present": clamp_policy, "expression": ".clamp(-1, 1)"},
        "tensor_to_image_scaling_policy": {
            "present": tensor_to_image_policy,
            "expression": "output_image[0].cpu() * 0.5 + 0.5 followed by transforms.ToPILImage()",
        },
        "channel_order_policy": "PIL RGB input is used; torchvision ToPILImage handles tensor channel order C,H,W to PIL image." if rgb_policy else "RGB input policy not found",
        "dtype_conversion_policy": "mixed_precision selects fp16/fp32 weight_dtype; postprocess converts CPU tensor through torchvision ToPILImage." if dtype_policy else "not found",
        "crop_resize_policy": "input image is resized/upscaled and cropped to multiples of 8 before model; resize_flag branch exists after inference." if resize_policy else "not found",
        "color_fix_policy": "optional adain_color_fix or wavelet_color_fix after ToPILImage." if color_fix_policy else "no color-fix branch found",
        "output_save_path_policy": "output_pil.save(os.path.join(args.output_dir, bname))" if save_policy else "not found",
        "postprocess_required_for_production_full_inference": True,
        "current_project_executed_postprocess": False,
        "current_status": "documented_not_executed",
        "image_video_generated": False,
        "raw_tensor_saved": False,
        "markers": markers,
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Official TADSR postprocess/output contract audit",
        "",
        f"`TADSR_POSTPROCESS_CONTRACT_AUDIT: {markers['TADSR_POSTPROCESS_CONTRACT_AUDIT']}`",
        f"`TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT: {markers['TADSR_OUTPUT_IMAGE_SAVE_POLICY_AUDIT']}`",
        f"`TADSR_POSTPROCESS_NOT_EXECUTED_GUARD: {markers['TADSR_POSTPROCESS_NOT_EXECUTED_GUARD']}`",
        "",
        "## Contract",
        "",
        f"- decoded tensor range: {result['decoded_tensor_range']}",
        f"- tensor-to-image scaling: `{result['tensor_to_image_scaling_policy']['expression']}`",
        f"- channel order: {result['channel_order_policy']}",
        f"- color fix: {result['color_fix_policy']}",
        f"- save policy: {result['output_save_path_policy']}",
        "",
        "## Current project status",
        "",
        "- Postprocess contract is documented.",
        "- Image save was not executed.",
        "- No PNG/JPG/MP4 model output was generated.",
        "- This does not upgrade `JITTOR_FULL_INFERENCE`.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for marker, value in markers.items():
        print(f"{marker}: {value}")
    return 0 if status in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
