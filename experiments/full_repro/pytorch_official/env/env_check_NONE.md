# 中文阅读说明：env_check_NONE.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official PyTorch Environment Check: NONE

Status: **BLOCKED_CRITICAL_MISSING**

Python: `/mnt/data/sj/venvs/tadsr_official_pytorch/bin/python`

| Package | Import | Critical | Expected | Actual | Status |
|---|---|---:|---|---|---|
| `torch` | `torch` | True | `2.0.1` | `-` | MISSING |
| `torchvision` | `torchvision` | True | `0.15.2` | `-` | MISSING |
| `diffusers` | `diffusers` | True | `-` | `-` | MISSING |
| `transformers` | `transformers` | True | `4.28.1` | `-` | MISSING |
| `peft` | `peft` | True | `0.9.0` | `-` | MISSING |
| `opencv-python` | `cv2` | True | `4.11.0.86` | `-` | MISSING |
| `Pillow` | `PIL` | True | `9.5.0` | `-` | MISSING |
| `numpy` | `numpy` | True | `1.24.3` | `-` | MISSING |
| `xformers` | `xformers` | False | `0.0.20` | `-` | MISSING |
| `pyiqa` | `pyiqa` | False | `0.1.10` | `-` | MISSING |
| `basicsr` | `basicsr` | False | `-` | `-` | MISSING |
| `loralib` | `loralib` | False | `-` | `-` | MISSING |
| `fairscale` | `fairscale` | False | `-` | `-` | MISSING |
| `lpips` | `lpips` | False | `-` | `-` | MISSING |
| `open-clip-torch` | `open_clip` | False | `2.20.0` | `-` | MISSING |
| `accelerate` | `accelerate` | False | `-` | `-` | MISSING |
| `safetensors` | `safetensors` | False | `-` | `-` | MISSING |
| `einops` | `einops` | False | `0.7.0` | `-` | MISSING |
| `PyYAML` | `yaml` | False | `6.0` | `-` | MISSING |
| `huggingface_hub` | `huggingface_hub` | False | `0.25.0` | `-` | MISSING |

## CUDA

```json
{
  "available": false,
  "error": "ModuleNotFoundError(\"No module named 'torch'\")"
}
```

## Critical missing

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `peft`
- `opencv-python`
- `Pillow`
- `numpy`

## Missing packages

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `peft`
- `opencv-python`
- `Pillow`
- `numpy`
- `xformers`
- `pyiqa`
- `basicsr`
- `loralib`
- `fairscale`
- `lpips`
- `open-clip-torch`
- `accelerate`
- `safetensors`
- `einops`
- `PyYAML`
- `huggingface_hub`
