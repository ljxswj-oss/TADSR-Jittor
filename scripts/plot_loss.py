import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for c in ["total_loss", "rec_loss", "edge_loss", "tavsd_loss"]:
        if c in df: plt.plot(df["step"], df[c], label=c)
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.savefig(out / "loss_curve.png", dpi=160); plt.close()
    if "psnr" in df:
        plt.figure(); plt.plot(df["step"], df["psnr"], label="PSNR")
        plt.xlabel("step"); plt.ylabel("PSNR(dB)"); plt.legend(); plt.tight_layout(); plt.savefig(out / "psnr_curve.png", dpi=160)
if __name__ == "__main__": main()
