# 中文阅读说明：upload_integrity_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Uploaded TADSR Weights Integrity Report

Status: **PASS**

Upload dir: `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights`
Selected venv: `/mnt/data/sj/venvs/tadsr_official_strict_cu118`

## Required Items

| Item | Expected | Exists | File count | Size |
|---|---|---:|---:|---:|
| `time_vae` | dir | True | 2 | 327.121 MB |
| `unet` | dir | True | 2 | 3.226 GB |
| `vae` | dir | True | 2 | 319.202 MB |
| `text_encoder` | dir | True | 2 | 1.268 GB |
| `tokenizer` | dir | True | 4 | 1.512 MB |
| `scheduler` | dir | True | 1 | 346.000 B |
| `feature_extractor` | dir | True | 1 | 342.000 B |
| `bert-base-uncased` | dir | True | 5 | 420.734 MB |
| `DAPE.pth` | file | True | - | 6.861 MB |
| `ram_swin_large_14m.pth` | file | True | - | 5.239 GB |
| `tadsr.pkl` | file | True | - | 25.064 MB |

## Missing

None

## Empty dirs

None

## Zero-size files

None

## Suspicious

None

## Safetensors Checks

| File | Status | Key count |
|---|---|---:|
| `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/text_encoder/model.safetensors` | PASS | 373 |
| `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/unet/diffusion_pytorch_model.safetensors` | PASS | 686 |

## torch.load Checks

| File | Status | Type | Key count | First keys |
|---|---|---|---:|---|
| `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/tadsr.pkl` | PASS | dict | 9 | `state_dict_vae_time, vae_lora_encoder_modules, unet_lora_encoder_modules, unet_lora_decoder_modules, unet_lora_others_modules` |
