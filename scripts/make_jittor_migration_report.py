#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("experiments/full_repro/jittor_alignment")


def load(path):
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING"}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {"status": "FAIL", "error": repr(exc)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {
        "preprocess_alignment": load(OUT / "preprocess_alignment.json"),
        "scheduler_alignment": load(OUT / "scheduler_alignment.json"),
        "weight_loading_alignment": load(OUT / "weight_loading_alignment.json"),
        "lora_key_mapping": load(OUT / "lora_key_mapping.json"),
        "time_vae_loading": load(OUT / "time_vae_loading.json"),
        "diffusers_weight_verification": load("experiments/full_repro/weights/diffusers_weight_verification.json"),
        "oracle_tensor_export": load("experiments/full_repro/oracle_tensors/smoke/metadata.json"),
        "official_smoke": load("experiments/full_repro/pytorch_official/smoke/smoke_status.json"),
    }
    md = ["# Jittor Migration Evidence Report", "", "This report summarizes current Jittor migration evidence. It does not claim full Jittor TADSR inference.", "", "## 1. Assessment Requirement Review", "", "| Requirement evidence | Current status | Evidence |", "|---|---|---|", "| Jittor code | PASS/PARTIAL | `jittor_tadsr/`, `jittor_tadsr_full/` |", "| Jittor training script | PASS | `scripts/train_jittor_tiny.sh` |", "| Jittor testing script | PASS | `scripts/test_jittor_tiny.sh` |", "| PyTorch alignment logs | PASS | `experiments/alignment_report.md` and oracle smoke |", "| Performance/training logs | PASS | `experiments/jittor_tiny/logs/` |", "| Loss/PSNR curves | PASS | `experiments/jittor_tiny/curves/` |", "| Visualizations | PASS | tiny grids and official smoke visual grid |", "| README/PPT/video materials | PASS/PARTIAL | `README.md`, `docs/` |", "", "## 2. Migration Evidence", "", "| Evidence item | Status |", "|---|---|"]
    for name, data in checks.items():
        md.append(f"| `{name}` | `{data.get('status')}` |")
    md += ["", "## 3. Completed Evidence", "", "- Jittor tiny train/test loop with logs, checkpoint, loss curve, PSNR curve and visual results.", "- PyTorch tiny baseline for small-scale alignment.", "- Official PyTorch oracle smoke with official code and uploaded official checkpoints.", "- Base and diffusers weight conversion into NPZ files under `/mnt/data/sj/checkpoints/TADSR/jittor_converted/`.", "- Oracle preprocess/scheduler tensors exported for module alignment.", "- Jittor preprocess, scheduler and weight-loading checks added.", "- LoRA key mapping and Time-VAE loading checklists added.", "", "## 4. Not Completed", "", "- Full TimeAware VAE forward.", "- Full UNet forward.", "- Full LoRA merge into an executable inference graph.", "- Full Jittor TADSR inference.", "- Jittor GPU execution, still blocked by cuDNN8 availability.", "- Full paper-scale training.", "", "## 5. Why This Is Honest Stage-wise Reproduction", "", "The project uses official PyTorch as the oracle, then builds a Jittor migration evidence chain through tiny training/testing, converted weights, oracle tensors and module-level alignment. It does not fabricate full paper-scale results.", "", "## 6. Next Steps", "", "1. Port TimeAware VAE blocks.", "2. Validate LoRA merge for matched keys.", "3. Port UNet ResNet and attention blocks.", "4. Run full inference alignment once block-level outputs match."]
    (OUT / "jittor_migration_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'jittor_migration_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
