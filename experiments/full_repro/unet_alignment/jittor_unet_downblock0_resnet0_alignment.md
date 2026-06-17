# 中文阅读说明：jittor_unet_downblock0_resnet0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.0.resnets.0 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm1 | `7.152557373046875e-07` | `2.7601924224539276e-08` | `8.777170917065951e-08` | `0.9999999999999957` | `0.0001` |
| conv1 | `2.86102294921875e-06` | `1.9809070472831536e-07` | `1.2165873140556353e-07` | `0.9999999999999928` | `0.0001` |
| time_emb_proj | `4.76837158203125e-07` | `5.54980942979455e-08` | `1.2099655286620712e-07` | `0.9999999999999892` | `0.0001` |
| conv2 | `1.1920928955078125e-06` | `1.0156748757594869e-07` | `2.2909321921173648e-07` | `0.9999999999999778` | `0.0001` |
| output | `1.430511474609375e-06` | `1.0156065641808709e-07` | `2.205985929432103e-07` | `0.99999999999998` | `0.0001` |

Note: Does not enter attentions.0, resnets.1, downsampler, full down_blocks.0, or full UNet.
