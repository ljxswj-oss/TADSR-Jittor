# 中文阅读说明：jittor_unet_upblock3_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.resnets.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| concat_input | `0.0` | `0.0` | `0.0` | `1.0000000000000002` | `1e-06` |
| norm1 | `2.980232238769531e-07` | `1.1337631913507784e-08` | `8.100818208705424e-08` | `0.9999999999999961` | `0.0001` |
| conv1 | `7.152557373046875e-07` | `7.338655176880593e-08` | `2.578336677880157e-07` | `0.9999999999999681` | `0.0002` |
| time_emb_proj | `1.7881393432617188e-07` | `1.806547516025603e-08` | `1.1680654885968535e-07` | `0.9999999999999895` | `0.0001` |
| after_temb_add | `7.152557373046875e-07` | `7.756764972555174e-08` | `2.2591668509253215e-07` | `0.9999999999999748` | `0.0002` |
| norm2 | `1.9073486328125e-06` | `1.1453761627113617e-07` | `2.5089655459958206e-07` | `0.9999999999999699` | `0.0001` |
| conv2 | `1.3113021850585938e-06` | `1.9772324879596682e-07` | `3.337009793187614e-07` | `0.999999999999944` | `0.0001` |
| shortcut | `1.2516975402832031e-06` | `1.1502214745107153e-07` | `1.726669406479938e-07` | `0.9999999999999846` | `0.0001` |
| output | `1.6689300537109375e-06` | `2.3189018847347142e-07` | `2.906548968351266e-07` | `0.9999999999999583` | `0.0001` |

Note: Only up_blocks.3.resnets.1 is executed after official residual concat. It stops before up_blocks.3.attentions.1, full up_blocks.3, full UNet forward, runtime LoRA, and full TADSR inference.
