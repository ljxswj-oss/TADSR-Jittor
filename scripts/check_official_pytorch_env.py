#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

OFFICIAL_REPO = Path(os.environ.get("OFFICIAL_PYTORCH_REPO", "/mnt/data/sj/projects/TADSR_official_pytorch"))
if OFFICIAL_REPO.exists() and str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

CRITICAL = {"torch", "torchvision", "diffusers", "transformers", "peft", "cv2", "PIL", "numpy"}
PACKAGE_SPECS = [
    ("torch", "torch", "2.0.1", True),
    ("torchvision", "torchvision", "0.15.2", True),
    ("diffusers", "diffusers", None, True),
    ("transformers", "transformers", "4.28.1", True),
    ("peft", "peft", "0.9.0", True),
    ("cv2", "opencv-python", "4.11.0.86", True),
    ("PIL", "Pillow", "9.5.0", True),
    ("numpy", "numpy", "1.24.3", True),
    ("xformers", "xformers", "0.0.20", False),
    ("pyiqa", "pyiqa", "0.1.10", False),
    ("basicsr", "basicsr", None, False),
    ("loralib", "loralib", None, False),
    ("fairscale", "fairscale", None, False),
    ("lpips", "lpips", None, False),
    ("open_clip", "open-clip-torch", "2.20.0", False),
    ("accelerate", "accelerate", None, False),
    ("safetensors", "safetensors", None, False),
    ("einops", "einops", "0.7.0", False),
    ("yaml", "PyYAML", "6.0", False),
    ("huggingface_hub", "huggingface_hub", "0.25.0", False),
]

def version_of(dist_name: str, module) -> str | None:
    try:
        return metadata.version(dist_name)
    except Exception:
        ver = getattr(module, "__version__", None)
        if ver:
            return str(ver)
        return None

def check_package(import_name: str, dist_name: str, expected: str | None, required: bool) -> dict:
    item = {"import_name": import_name, "dist_name": dist_name, "expected_version": expected, "required": required, "critical": import_name in CRITICAL}
    try:
        module = importlib.import_module(import_name)
        version = version_of(dist_name, module)
        item.update({"ok": True, "version": version, "missing": False})
        if expected is None:
            item.update({"exact_match": None, "version_mismatch": False})
        else:
            actual_base = str(version).split("+")[0] if version is not None else None
            exact = version == expected or actual_base == expected
            item.update({"exact_match": exact, "version_mismatch": not exact})
    except Exception as exc:
        item.update({"ok": False, "missing": True, "error": repr(exc), "exact_match": False, "version_mismatch": False})
    return item

def torch_cuda_info() -> dict:
    try:
        import torch
        available = torch.cuda.is_available()
        info = {
            "available": bool(available),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }
        if available:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return info
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv-name", default="current")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    safe_name = args.venv_name.replace("/", "_")
    out_path = Path(args.output or f"experiments/full_repro/pytorch_official/env/env_check_{safe_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packages = [check_package(*spec) for spec in PACKAGE_SPECS]
    missing_packages = [p["dist_name"] for p in packages if p.get("missing")]
    critical_missing = [p["dist_name"] for p in packages if p.get("missing") and p.get("critical")]
    version_mismatches = [{"package": p["dist_name"], "expected": p["expected_version"], "actual": p.get("version")} for p in packages if p.get("version_mismatch")]
    critical_mismatches = [x for x in version_mismatches if any(p["dist_name"] == x["package"] and p.get("critical") for p in packages)]
    cuda = torch_cuda_info()
    if critical_missing:
        status = "BLOCKED_CRITICAL_MISSING"
    elif critical_mismatches:
        status = "PARTIAL_CRITICAL_VERSION_MISMATCH"
    elif missing_packages:
        status = "PARTIAL_NONCRITICAL_MISSING"
    elif version_mismatches:
        status = "PARTIAL_VERSION_MISMATCH"
    else:
        status = "PASS"
    data = {
        "venv_name": args.venv_name,
        "status": status,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": cuda,
        "missing_packages": missing_packages,
        "missing_required": missing_packages,
        "critical_missing": critical_missing,
        "version_mismatches": version_mismatches,
        "version_mismatch": version_mismatches,
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    lines = [f"# Official PyTorch Environment Check: {args.venv_name}", "", f"Status: **{status}**", "", f"Python: `{sys.executable}`", "", "| Package | Import | Critical | Expected | Actual | Status |", "|---|---|---:|---|---|---|"]
    for pkg in packages:
        if pkg.get("missing"):
            st = "MISSING"
        elif pkg.get("version_mismatch"):
            st = "VERSION_MISMATCH"
        else:
            st = "OK"
        lines.append(f"| `{pkg['dist_name']}` | `{pkg['import_name']}` | {pkg['critical']} | `{pkg.get('expected_version') or '-'}` | `{pkg.get('version') or '-'}` | {st} |")
    lines += ["", "## CUDA", "", "```json", json.dumps(cuda, indent=2), "```"]
    if critical_missing:
        lines += ["", "## Critical missing", ""] + [f"- `{x}`" for x in critical_missing]
    if missing_packages:
        lines += ["", "## Missing packages", ""] + [f"- `{x}`" for x in missing_packages]
    if version_mismatches:
        lines += ["", "## Version mismatches", ""] + [f"- `{x['package']}` expected `{x['expected']}`, got `{x['actual']}`" for x in version_mismatches]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Official PyTorch env check ({args.venv_name}): {status}")
    print(f"JSON: {out_path}")
    print(f"Markdown: {md_path}")
    if critical_missing:
        return 2
    if missing_packages or version_mismatches:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
