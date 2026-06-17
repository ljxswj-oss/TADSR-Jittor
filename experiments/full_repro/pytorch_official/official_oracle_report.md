# 中文阅读说明：official_oracle_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official PyTorch Oracle Report

This report records the official PyTorch oracle state. It does not claim a full Jittor reproduction.

## 1. Official Environment

- Selected env kind: `strict-cu118`
- Selected env path: `/mnt/data/sj/venvs/tadsr_official_strict_cu118`
- Environment check status: `PASS`
- Torch CUDA: `11.8`
- cuDNN: `8700`
- CUDA available: `True`
- GPU devices: `NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4090`

## 2. Official Assets

- Assets input dir: `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights`
- Missing items: `none`

| Required item | Exists | Type | Size |
|---|---:|---|---:|
| `time_vae` | True | dir | 327.121 MB |
| `unet` | True | dir | 3.226 GB |
| `vae` | True | dir | 319.202 MB |
| `text_encoder` | True | dir | 1.268 GB |
| `tokenizer` | True | dir | 1.512 MB |
| `scheduler` | True | dir | 346.000 B |
| `feature_extractor` | True | dir | 342.000 B |
| `bert-base-uncased` | True | dir | 420.734 MB |
| `DAPE.pth` | True | file | 6.861 MB |
| `ram_swin_large_14m.pth` | True | file | 5.239 GB |
| `tadsr.pkl` | True | file | 25.064 MB |

## 3. Smoke Inference

- Status: `PASS`
- Reason: `official smoke completed`
- Input count: `4`
- Output count: `4`
- Runtime sec: `15.0`
- Visual grid: `experiments/full_repro/pytorch_official/smoke/visual_grid.png`

## 4. Subset Benchmark

| Dataset | Status | Reason |
|---|---|---|
| `RealSR` | `BLOCKED_DATASET_MISSING` | RealSR missing at /mnt/data/sj/datasets/TADSR/RealSR |
| `DRealSR` | `BLOCKED_DATASET_MISSING` | DRealSR missing at /mnt/data/sj/datasets/TADSR/DRealSR |
| `RealLR200` | `BLOCKED_DATASET_MISSING` | RealLR200 missing at /mnt/data/sj/datasets/TADSR/RealLR200 |

## 5. Current Blockers

- `PYTORCH_OFFICIAL_ENV_RELAXED_PYPI`: PARTIAL - version mismatches: torch, torchvision
- `WHEELHOUSE`: BLOCKED - 0 files in /mnt/data/sj/wheelhouse/tadsr_official_pytorch
- `JITTOR_GPU`: BLOCKED - cuDNN8 required; cuDNN9 not acceptable
- `JITTOR_FULL_PORT`: PARTIAL - skeleton only; full VAE/UNet not complete

## 6. Next Jittor Migration Action

1. Upload RealSR/DRealSR/RealLR200 benchmark datasets if paper-level metrics are needed.
2. Continue Jittor module-level alignment from the official PyTorch smoke oracle.
3. Keep official weights under /mnt/data/sj and avoid copying them into git.
4. Use the official PyTorch oracle for module-level Jittor alignment: scheduler, preprocessing, VAE, LoRA, UNet blocks, and full inference.
5. Do not claim full Jittor reproduction until oracle-aligned outputs and benchmarks pass.
