# 中文阅读说明：jittor_unet_downblock0_attention1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.0.attentions.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| proj_out_sequence | `1.138448715209961e-05` | `2.12727735373619e-06` | `3.85446561803591e-06` | `0.9999999999929698` | `0.001` |
| proj_out_nchw | `1.138448715209961e-05` | `2.12727735373619e-06` | `3.85446561803591e-06` | `0.9999999999929701` | `0.001` |
| output | `1.1444091796875e-05` | `2.127268334106702e-06` | `2.8085169476042263e-06` | `0.9999999999959186` | `0.002` |

Note: Stops before downsampler/full down_blocks.0.
