# Real Smoke Run Summary

所有数值均来自本机真实运行日志。当前 run 是 16 张 synthetic tiny data、30 training steps 的 smoke test，用于证明 Jittor 训练/测试/可视化/对齐链路完整可运行。

## 环境

- Host: vip-desktop
- OS: Ubuntu 22.04.5 LTS
- GPU: 6 x RTX 4090
- Jittor: 1.3.11.0
- PyTorch: 2.11.0+cu128
- Jittor backend used for smoke: CPU fallback, because CUDA backend lacks CUDNN development headers.

## Metrics

| Framework | PSNR | SSIM | Latency ms/image | Params | Checkpoint MB |
|---|---:|---:|---:|---:|---:|
| Jittor tiny | 9.392961 | 0.062834 | 39.371 | 256003 | 0.979259 |
| PyTorch tiny | 9.375344 | 0.057573 | 20.533 | 246755 | 0.955689 |

## Real artifact paths

- Jittor train stdout: `experiments/jittor_tiny/logs/train_stdout.log`
- Jittor train CSV: `experiments/jittor_tiny/logs/train_log.csv`
- Jittor loss curve: `experiments/jittor_tiny/curves/loss_curve.png`
- Jittor PSNR curve: `experiments/jittor_tiny/curves/psnr_curve.png`
- Jittor test metrics: `experiments/jittor_tiny/logs/test_metrics.json`
- Jittor result grid: `experiments/jittor_tiny/results/compare_grid.png`
- PyTorch test metrics: `experiments/pytorch_tiny/logs/test_metrics.json`
- Alignment report: `experiments/alignment_report.md`
- Loss comparison: `experiments/compare_loss_jittor_vs_pytorch.png`
- PSNR comparison: `experiments/compare_psnr_jittor_vs_pytorch.png`
