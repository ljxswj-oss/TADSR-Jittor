# 中文阅读说明：jittor_unet_downblock0_attention0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.0.attentions.0 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| proj_out_sequence | `7.852911949157715e-06` | `1.3148315283650902e-06` | `3.9559708346726905e-06` | `0.9999999999920258` | `0.001` |
| proj_out_nchw | `7.852911949157715e-06` | `1.3148315283650902e-06` | `3.9559708346726905e-06` | `0.9999999999920267` | `0.001` |
| output | `7.867813110351562e-06` | `1.3148283755981537e-06` | `2.216273479623449e-06` | `0.9999999999972097` | `0.002` |

Note: No resnets.1, attentions.1, downsampler, full down_blocks.0, or full UNet.
