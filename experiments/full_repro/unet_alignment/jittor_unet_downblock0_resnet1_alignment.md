# 中文阅读说明：jittor_unet_downblock0_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.0.resnets.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm1 | `2.384185791015625e-07` | `1.760267654728198e-08` | `7.515876719475014e-08` | `0.9999999999999962` | `0.0001` |
| conv1 | `7.152557373046875e-07` | `8.561973516840737e-08` | `2.1575665889504777e-07` | `0.9999999999999775` | `0.0001` |
| time_emb_proj | `2.384185791015625e-07` | `2.0809238776564598e-08` | `1.1682594574447797e-07` | `0.9999999999999902` | `0.0001` |
| conv2 | `1.1026859283447266e-06` | `1.7648123313840357e-07` | `3.8219380766891404e-07` | `0.9999999999999261` | `0.0001` |
| output | `1.1920928955078125e-06` | `1.76531742113184e-07` | `2.968001703160554e-07` | `0.999999999999959` | `0.0001` |

Note: Does not enter attentions.1, downsampler, full down_blocks.0, or full UNet.
