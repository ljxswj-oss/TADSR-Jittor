from __future__ import annotations

import argparse
import json

from .load_weights import summarize_all
from .scheduler import MinimalDDPMScheduler
from .preprocess import preprocess_lq


def check_preprocess(image_path: str):
    out = preprocess_lq(image_path)
    return {"status": "PASS", "metadata": out["metadata"], "lq_shape": list(out["lq_0_1"].shape)}


def check_scheduler(config_path: str):
    sched = MinimalDDPMScheduler.from_config(config_path)
    return {"status": "PASS", "betas_shape": list(sched.betas.shape), "alpha_cumprod_0": float(sched.alphas_cumprod[0]), "alpha_cumprod_last": float(sched.alphas_cumprod[-1])}


def check_weights():
    return summarize_all()


def main() -> int:
    parser = argparse.ArgumentParser(description="TADSR full Jittor migration checks. Full inference is intentionally not implemented yet.")
    parser.add_argument("--check-preprocess", default=None, help="image path for preprocess check")
    parser.add_argument("--check-scheduler", default="/mnt/data/sj/checkpoints/TADSR/preset/weights/scheduler/scheduler_config.json")
    parser.add_argument("--check-weights", action="store_true")
    parser.add_argument("--full-inference", action="store_true")
    args = parser.parse_args()
    if args.full_inference:
        raise NotImplementedError("Full Jittor TADSR inference is not complete. This CLI only supports migration checks.")
    results = {}
    if args.check_preprocess:
        results["preprocess"] = check_preprocess(args.check_preprocess)
    if args.check_scheduler:
        results["scheduler"] = check_scheduler(args.check_scheduler)
    if args.check_weights:
        results["weights"] = check_weights()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
