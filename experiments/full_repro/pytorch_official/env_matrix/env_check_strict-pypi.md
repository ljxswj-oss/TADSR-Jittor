# 中文阅读说明：env_check_strict-pypi.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official PyTorch Environment Check: strict-pypi

Status: **PASS**

Python: `/mnt/data/sj/venvs/tadsr_official_strict_pypi/bin/python`

| Package | Import | Critical | Expected | Actual | Status |
|---|---|---:|---|---|---|
| `torch` | `torch` | True | `2.0.1` | `2.0.1` | OK |
| `torchvision` | `torchvision` | True | `0.15.2` | `0.15.2` | OK |
| `diffusers` | `diffusers` | True | `-` | `0.25.0` | OK |
| `transformers` | `transformers` | True | `4.28.1` | `4.28.1` | OK |
| `peft` | `peft` | True | `0.9.0` | `0.9.0` | OK |
| `opencv-python` | `cv2` | True | `4.11.0.86` | `4.11.0.86` | OK |
| `Pillow` | `PIL` | True | `9.5.0` | `9.5.0` | OK |
| `numpy` | `numpy` | True | `1.24.3` | `1.24.3` | OK |
| `xformers` | `xformers` | False | `0.0.20` | `0.0.20` | OK |
| `pyiqa` | `pyiqa` | False | `0.1.10` | `0.1.10` | OK |
| `basicsr` | `basicsr` | False | `-` | `1.4.2` | OK |
| `loralib` | `loralib` | False | `-` | `0.1.2` | OK |
| `fairscale` | `fairscale` | False | `-` | `0.4.13` | OK |
| `lpips` | `lpips` | False | `-` | `0.1.4` | OK |
| `open-clip-torch` | `open_clip` | False | `2.20.0` | `2.20.0` | OK |
| `accelerate` | `accelerate` | False | `-` | `1.13.0` | OK |
| `safetensors` | `safetensors` | False | `-` | `0.7.0` | OK |
| `einops` | `einops` | False | `0.7.0` | `0.7.0` | OK |
| `PyYAML` | `yaml` | False | `6.0` | `6.0` | OK |
| `huggingface_hub` | `huggingface_hub` | False | `0.25.0` | `0.25.0` | OK |

## CUDA

```json
{
  "available": true,
  "torch_cuda": "11.7",
  "cudnn": 8500,
  "device_count": 6,
  "devices": [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4090"
  ]
}
```
