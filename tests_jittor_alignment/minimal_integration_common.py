from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jittor_tadsr_full.minimal_integration import TADSRMinimalLatentIntegrationTester
from jittor_tadsr_full.scheduler import TADSRSchedulerBoundaryTester
from jittor_tadsr_full.utils import compare_arrays, write_json
from jittor_tadsr_full.vae_time_aware import TimeVAEActualHookBehaviorTester
from tests_jittor_alignment.timevae_actual_hook_behavior_common import WEIGHTS as TIMEVAE_WEIGHTS
from tests_jittor_alignment.unet_full_forward_common import full_forward_tester


OUT = Path("experiments/full_repro/integration_alignment")
ORACLE = OUT / "oracle_tensors_minimal_latent_integration"
SCHEDULER_CONFIG = Path("/mnt/data/sj/checkpoints/TADSR/preset/weights/scheduler/scheduler_config.json")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_oracle():
    meta_path = ORACLE / "minimal_latent_integration_oracle_metadata.json"
    if not ORACLE.exists() or not meta_path.exists():
        return None, {"status": "BLOCKED", "reason": "minimal latent integration oracle missing"}
    arrays = {}
    for path in sorted(ORACLE.glob("*.npy")):
        arrays[path.stem] = np.load(path)
    required = [
        "minimal_input",
        "minimal_timestep",
        "minimal_encoder_hidden_states",
        "minimal_sample_epsilon",
        "minimal_posterior_sample",
        "minimal_scaled_latent",
        "minimal_unet_model_pred",
        "minimal_alpha_prod_t",
        "minimal_sqrt_alpha_prod_t",
        "minimal_x0_from_res",
        "minimal_decode_input",
        "minimal_decoded_output",
        "minimal_final_clamped_output",
    ]
    missing = [name for name in required if name not in arrays]
    if missing:
        return None, {"status": "BLOCKED", "reason": f"missing oracle arrays: {missing}"}
    return {"arrays": arrays, "metadata": load_json(meta_path)}, None


def cosine(a, b):
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom else 1.0


def add_metrics(got, expected, tolerance: float) -> dict:
    cmp = compare_arrays(got, expected)
    cmp["cosine_similarity"] = cosine(got, expected) if cmp.get("shape_match") else None
    cmp["tolerance"] = tolerance
    cmp["status"] = status_from_metrics(cmp)
    return cmp


def status_from_metrics(metrics: dict) -> str:
    tolerance = float(metrics.get("tolerance", 2e-3))
    if metrics.get("shape_match") and float(metrics.get("max_abs_error", 1.0)) <= tolerance:
        return "PASS"
    if metrics.get("shape_match") and float(metrics.get("max_abs_error", 1.0)) <= max(1e-3, tolerance * 10.0):
        return "PARTIAL"
    return "FAIL"


def status_from_diagnostics(diag: dict) -> str:
    statuses = [v.get("status") for v in diag.values()]
    if all(s == "PASS" for s in statuses):
        return "PASS"
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    return "PARTIAL"


def minimal_tester(scaling_factor: float) -> TADSRMinimalLatentIntegrationTester:
    return TADSRMinimalLatentIntegrationTester(
        TimeVAEActualHookBehaviorTester(TIMEVAE_WEIGHTS),
        full_forward_tester(),
        TADSRSchedulerBoundaryTester.from_config(SCHEDULER_CONFIG),
        scaling_factor=scaling_factor,
    )


def write_report(name: str, title: str, result: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"{name}.json", result)
    md = [f"# {title}", "", f"Status: **{result['status']}**", ""]
    md += ["| Stage | Status | Max abs error | Mean abs error | Cosine | Tolerance |", "|---|---|---:|---:|---:|---:|"]
    for stage, m in result.get("diagnostics", {}).items():
        md.append(f"| {stage} | `{m.get('status')}` | `{m.get('max_abs_error')}` | `{m.get('mean_abs_error')}` | `{m.get('cosine_similarity')}` | `{m.get('tolerance')}` |")
    md += [
        "",
        "No full denoising loop, production full inference, image/video generation, image save, or generic runtime LoRA was executed.",
        "",
        "```json",
        json.dumps(result.get("policy", {}), indent=2, ensure_ascii=False),
        "```",
    ]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
