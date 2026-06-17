# 中文阅读说明：env_check_strict.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Official PyTorch Environment Check: strict

Status: **BLOCKED_MISSING_PACKAGES**

Python: `/mnt/data/sj/venvs/tadsr_official_pytorch/bin/python`

| Package | Import | Expected | Actual | Status |
|---|---|---|---|---|
| `torch` | `torch` | `2.0.1` | `-` | MISSING |
| `torchvision` | `torchvision` | `-` | `-` | MISSING |
| `diffusers` | `diffusers` | `-` | `-` | MISSING |
| `transformers` | `transformers` | `4.28.1` | `-` | MISSING |
| `peft` | `peft` | `0.9.0` | `-` | MISSING |
| `accelerate` | `accelerate` | `-` | `-` | MISSING |
| `xformers` | `xformers` | `0.0.20` | `-` | MISSING |
| `open-clip-torch` | `open_clip` | `2.20.0` | `-` | MISSING |
| `einops` | `einops` | `0.7.0` | `-` | MISSING |
| `Pillow` | `PIL` | `9.5.0` | `-` | MISSING |
| `PyYAML` | `yaml` | `6.0` | `-` | MISSING |
| `huggingface_hub` | `huggingface_hub` | `0.25.0` | `-` | MISSING |
| `numpy` | `numpy` | `1.24.3` | `-` | MISSING |
| `opencv-python` | `cv2` | `4.11.0.86` | `-` | MISSING |
| `safetensors` | `safetensors` | `-` | `-` | missing optional |
| `pyiqa` | `pyiqa` | `0.1.10` | `-` | MISSING |
| `basicsr` | `basicsr` | `-` | `-` | MISSING |
| `loralib` | `loralib` | `-` | `-` | MISSING |
| `fairscale` | `fairscale` | `-` | `-` | MISSING |
| `lpips` | `lpips` | `-` | `-` | MISSING |

## CUDA

```json
{
  "available": false,
  "error": "ModuleNotFoundError(\"No module named 'torch'\")"
}
```

## Missing required packages

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `peft`
- `accelerate`
- `xformers`
- `open-clip-torch`
- `einops`
- `Pillow`
- `PyYAML`
- `huggingface_hub`
- `numpy`
- `opencv-python`
- `pyiqa`
- `basicsr`
- `loralib`
- `fairscale`
- `lpips`
