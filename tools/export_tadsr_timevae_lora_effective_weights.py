#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

STRICT_PYTHON = Path("/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python")
OFFICIAL_REPO = Path("/mnt/data/sj/projects/TADSR_official_pytorch")
JITTOR_REPO = Path("/mnt/data/sj/projects/TADSR-Jittor")
WEIGHTS_DIR = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights")
TADSR_CKPT = WEIGHTS_DIR / "tadsr.pkl"
POLICY_JSON = Path("experiments/full_repro/lora_alignment/audit_tadsr_lora_policy.json")
OUT_DIR = Path("experiments/full_repro/lora_alignment/timevae_lora_effective_weights")
OUT_NPZ = OUT_DIR / "converted_timevae_lora_effective_weights.npz"
OUT_METADATA = OUT_DIR / "timevae_lora_effective_weights_metadata.json"
OUT_SUMMARY = OUT_DIR / "oracle_summary.txt"


def maybe_reexec_strict() -> None:
    if os.environ.get("TADSR_TIMEVAE_LORA_NO_REEXEC") == "1":
        return
    if STRICT_PYTHON.exists() and Path(sys.executable) != STRICT_PYTHON:
        os.execv(str(STRICT_PYTHON), [str(STRICT_PYTHON), *sys.argv])


maybe_reexec_strict()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def patch_huggingface_hub_compat() -> None:
    # The vendored official diffusers tree imports cached_download, which newer
    # huggingface_hub releases removed. This audit never downloads remote files,
    # but the symbol must exist for the official module import to succeed.
    try:
        import huggingface_hub
    except Exception:
        return
    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else "_")
    return "_".join("".join(out).split("_"))


def module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    modules = dict(root.named_modules())
    if path not in modules:
        raise KeyError(f"module path not found in official TimeVAE: {path}")
    return modules[path]


def base_layer(module: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(module, "base_layer"):
        raise TypeError(f"module is not a PEFT LoRA wrapper: {type(module)}")
    return module.base_layer


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "numel": int(x.numel()),
        "mean": float(x.mean()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False)) if x.numel() else 0.0,
        "min": float(x.min()) if x.numel() else 0.0,
        "max": float(x.max()) if x.numel() else 0.0,
        "max_abs": float(x.abs().max()) if x.numel() else 0.0,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    denom = float(aa.norm() * bb.norm())
    if denom == 0.0:
        return 1.0 if float((aa - bb).abs().max()) == 0.0 else 0.0
    return float(torch.dot(aa, bb) / denom)


def make_input(layer: torch.nn.Module, module_path: str) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    seed = 1729 + sum(ord(c) for c in module_path)
    gen.manual_seed(seed)
    if isinstance(layer, torch.nn.Conv2d):
        in_ch = int(layer.in_channels)
        kh, kw = layer.kernel_size
        # Keep this tiny: this is only a single-layer contract check, not inference.
        h = max(8, kh + 4)
        w = max(8, kw + 4)
        return torch.randn((2, in_ch, h, w), generator=gen, dtype=torch.float32)
    if isinstance(layer, torch.nn.Linear):
        return torch.randn((3, int(layer.in_features)), generator=gen, dtype=torch.float32)
    raise TypeError(f"unsupported base layer for manual verify: {type(layer)}")


def manual_forward(layer: torch.nn.Module, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if isinstance(layer, torch.nn.Conv2d):
        return F.conv2d(
            x,
            weight,
            bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
    if isinstance(layer, torch.nn.Linear):
        return F.linear(x, weight, bias)
    raise TypeError(f"unsupported base layer for manual verify: {type(layer)}")


def verify_module(module: torch.nn.Module, layer: torch.nn.Module, module_path: str, effective_weight: torch.Tensor) -> dict[str, Any]:
    x = make_input(layer, module_path)
    bias = layer.bias.detach().float().cpu() if getattr(layer, "bias", None) is not None else None
    module = module.cpu()
    module.eval()
    with torch.no_grad():
        official = module(x)
        manual = manual_forward(layer.cpu(), x, effective_weight.cpu(), bias)
    diff = (official.detach().float().cpu() - manual.detach().float().cpu()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    cos = cosine(official, manual)
    if max_abs < 1e-5:
        status = "PASS"
        tolerance = 1e-5
    elif max_abs < 1e-4:
        status = "PASS"
        tolerance = 1e-4
    elif max_abs < 1e-3:
        status = "PARTIAL"
        tolerance = 1e-3
    else:
        status = "FAIL"
        tolerance = 1e-3
    return {
        "shape_match": list(official.shape) == list(manual.shape),
        "output_shape": list(official.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cos,
        "tolerance": tolerance,
        "status": status,
    }


def load_official_timevae() -> torch.nn.Module:
    patch_huggingface_hub_compat()
    sys.path.insert(0, str(OFFICIAL_REPO))
    from tadsr import initialize_vae_time

    args = SimpleNamespace(pretrained_model_name_or_path=str(WEIGHTS_DIR), lora_rank=4)
    vae, _targets = initialize_vae_time(args)
    checkpoint = torch.load(TADSR_CKPT, map_location="cpu")
    state_dict_vae = checkpoint.get("state_dict_vae", {})
    loaded = 0
    missing = []
    for name, param in vae.named_parameters():
        if "lora" not in name:
            continue
        if name in state_dict_vae:
            param.data.copy_(state_dict_vae[name].to(dtype=param.dtype, device=param.device))
            loaded += 1
        else:
            missing.append(name)
    if hasattr(vae, "set_adapter"):
        vae.set_adapter(["default_encoder"])
    vae.eval()
    vae.requires_grad_(False)
    vae._tadsr_lora_loaded_parameter_count = loaded
    vae._tadsr_lora_missing_parameters = missing
    return vae


def main() -> int:
    os.chdir(JITTOR_REPO)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_json(POLICY_JSON)
    inventory = [row for row in policy.get("module_inventory", []) if row.get("component") == "TimeVAE"]
    arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    if len(inventory) != 32:
        result = {
            "status": "FAIL",
            "timevae_active_lora_pair_count": len(inventory),
            "expected_timevae_active_lora_pair_count": 32,
            "reason": "TimeVAE active LoRA inventory count does not match the audited expected count.",
        }
        OUT_METADATA.write_text(json.dumps(result, indent=2), encoding="utf-8")
        OUT_SUMMARY.write_text("TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: FAIL\n", encoding="utf-8")
        print("TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: FAIL")
        return 1

    vae = load_official_timevae()
    loaded = int(getattr(vae, "_tadsr_lora_loaded_parameter_count", 0))
    missing_params = list(getattr(vae, "_tadsr_lora_missing_parameters", []))
    if loaded != 64 or missing_params:
        result = {
            "status": "FAIL",
            "timevae_active_lora_pair_count": len(inventory),
            "loaded_lora_parameter_count": loaded,
            "expected_loaded_lora_parameter_count": 64,
            "missing_lora_parameters": missing_params,
        }
        OUT_METADATA.write_text(json.dumps(result, indent=2), encoding="utf-8")
        OUT_SUMMARY.write_text("TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: FAIL\n", encoding="utf-8")
        print("TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: FAIL")
        return 1

    status_counts: Counter[str] = Counter()
    adapter_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    max_abs_values: list[float] = []
    for row in inventory:
        module_path = str(row["module_path"])
        adapter_name = str(row.get("active_adapters", ["default_encoder"])[0])
        module = module_by_path(vae, module_path)
        layer = base_layer(module)
        base_weight = layer.weight.detach().float().cpu()
        delta_weight = module.get_delta_weight(adapter_name).detach().float().cpu()
        effective_weight = base_weight + delta_weight
        bias = layer.bias.detach().float().cpu() if getattr(layer, "bias", None) is not None else None
        verify = verify_module(module, layer, module_path, effective_weight)
        status_counts[verify["status"]] += 1
        adapter_counts[adapter_name] += 1
        class_counts[layer.__class__.__name__] += 1
        max_abs_values.append(float(verify["max_abs_error"]))
        prefix = f"timevae::{module_path}::{adapter_name}"
        normalized = f"timevae_{sanitize(module_path)}_{sanitize(adapter_name)}"
        arrays[f"{prefix}::base_weight"] = base_weight.numpy().astype(np.float32)
        arrays[f"{prefix}::delta_weight"] = delta_weight.numpy().astype(np.float32)
        arrays[f"{prefix}::effective_weight"] = effective_weight.numpy().astype(np.float32)
        if bias is not None:
            arrays[f"{prefix}::bias"] = bias.numpy().astype(np.float32)
        metadata_rows.append({
            "component": "TimeVAE",
            "module_path": module_path,
            "adapter_name": adapter_name,
            "module_class": module.__class__.__name__,
            "base_layer_class": layer.__class__.__name__,
            "weight_type": "Conv2d" if isinstance(layer, torch.nn.Conv2d) else "Linear" if isinstance(layer, torch.nn.Linear) else layer.__class__.__name__,
            "base_weight_shape": list(base_weight.shape),
            "bias_shape": list(bias.shape) if bias is not None else None,
            "delta_weight_shape": list(delta_weight.shape),
            "effective_weight_shape": list(effective_weight.shape),
            "lora_A_shape": row.get("lora_A_shape"),
            "lora_B_shape": row.get("lora_B_shape"),
            "rank": row.get("rank"),
            "alpha": row.get("alpha"),
            "scaling": float(getattr(module, "scaling", {}).get(adapter_name, math.nan)),
            "active_in_official_inference_path": True,
            "npz_keys": {
                "base_weight": f"{prefix}::base_weight",
                "delta_weight": f"{prefix}::delta_weight",
                "effective_weight": f"{prefix}::effective_weight",
                "bias": f"{prefix}::bias" if bias is not None else None,
            },
            "normalized_key": f"{normalized}_effective_weight",
            "base_weight_stats": tensor_stats(base_weight),
            "delta_weight_stats": tensor_stats(delta_weight),
            "effective_weight_stats": tensor_stats(effective_weight),
            "manual_verify": verify,
        })

    np.savez_compressed(OUT_NPZ, **arrays)
    all_pass = status_counts.get("FAIL", 0) == 0 and status_counts.get("PARTIAL", 0) == 0 and len(metadata_rows) == 32
    result = {
        "status": "PASS" if all_pass else "PARTIAL" if status_counts.get("FAIL", 0) == 0 else "FAIL",
        "timevae_lora_effective_weights_audit": "PASS" if len(metadata_rows) == 32 else "FAIL",
        "timevae_lora_effective_weights_export": "PASS" if OUT_NPZ.exists() and len(arrays) >= 96 else "FAIL",
        "timevae_lora_effective_weight_manual_verify": "PASS" if all_pass else "PARTIAL" if status_counts.get("FAIL", 0) == 0 else "FAIL",
        "official_repo": str(OFFICIAL_REPO),
        "weights_dir": str(WEIGHTS_DIR),
        "checkpoint": str(TADSR_CKPT),
        "npz_path": str(OUT_NPZ),
        "npz_size_bytes": OUT_NPZ.stat().st_size if OUT_NPZ.exists() else 0,
        "timevae_active_lora_pair_count": len(metadata_rows),
        "expected_timevae_active_lora_pair_count": 32,
        "loaded_lora_parameter_count": loaded,
        "adapter_counts": dict(adapter_counts),
        "base_layer_class_counts": dict(class_counts),
        "manual_verify_status_counts": dict(status_counts),
        "manual_verify_max_abs_error_max": max(max_abs_values) if max_abs_values else None,
        "manual_verify_max_abs_error_mean": float(sum(max_abs_values) / len(max_abs_values)) if max_abs_values else None,
        "rows": metadata_rows,
    }
    OUT_METADATA.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# TimeVAE LoRA Effective Weight Export",
        "",
        f"TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: {result['timevae_lora_effective_weights_audit']}",
        f"TIME_VAE_LORA_EFFECTIVE_WEIGHTS_EXPORT: {result['timevae_lora_effective_weights_export']}",
        f"TIME_VAE_LORA_EFFECTIVE_WEIGHT_MANUAL_VERIFY: {result['timevae_lora_effective_weight_manual_verify']}",
        "",
        f"TimeVAE active LoRA pairs: {len(metadata_rows)}",
        f"Loaded LoRA parameters: {loaded}",
        f"Adapters: {dict(adapter_counts)}",
        f"Base layer classes: {dict(class_counts)}",
        f"Manual verify status counts: {dict(status_counts)}",
        f"Max manual verify max_abs_error: {result['manual_verify_max_abs_error_max']}",
        f"NPZ artifact: {OUT_NPZ}",
        f"NPZ size bytes: {result['npz_size_bytes']}",
        "",
        "| Module | Adapter | Layer | Shape | Max abs error | Mean abs error | Cosine | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in metadata_rows:
        v = item["manual_verify"]
        lines.append(
            f"| `{item['module_path']}` | `{item['adapter_name']}` | {item['base_layer_class']} | "
            f"`{item['effective_weight_shape']}` | {v['max_abs_error']:.6g} | {v['mean_abs_error']:.6g} | "
            f"{v['cosine_similarity']:.12f} | {v['status']} |"
        )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TIME_VAE_LORA_EFFECTIVE_WEIGHTS_AUDIT: {result['timevae_lora_effective_weights_audit']}")
    print(f"TIME_VAE_LORA_EFFECTIVE_WEIGHTS_EXPORT: {result['timevae_lora_effective_weights_export']}")
    print(f"TIME_VAE_LORA_EFFECTIVE_WEIGHT_MANUAL_VERIFY: {result['timevae_lora_effective_weight_manual_verify']}")
    print(f"TIME_VAE_ACTIVE_LORA_PAIR_COUNT: {len(metadata_rows)}")
    print(f"TIME_VAE_LORA_MANUAL_VERIFY_MAX_ABS: {result['manual_verify_max_abs_error_max']}")
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
