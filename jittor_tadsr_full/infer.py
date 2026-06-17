from __future__ import annotations
import argparse

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz")
    ap.add_argument("--timestep", type=int, default=999)
    ap.add_argument("--use_cuda", type=int, default=0)
    ap.parse_args()
    raise NotImplementedError("Full Jittor UNet/VAE port is not completed yet.")

if __name__ == "__main__":
    raise SystemExit(main())
