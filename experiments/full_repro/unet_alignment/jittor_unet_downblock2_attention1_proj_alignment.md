# 中文阅读说明：jittor_unet_downblock2_attention1_proj_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.2.attentions.1 Projection Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm | `2.637505531311035e-06` | `7.902313344798628e-07` | `3.026842982026046e-06` | `0.9999999999944281` | `0.0001` |
| sequence_input | `2.637505531311035e-06` | `7.902313344798628e-07` | `3.026842982026046e-06` | `0.999999999994428` | `0.0001` |
| proj_in | `6.0558319091796875e-05` | `2.5415007268492217e-06` | `4.4312742814722205e-06` | `0.999999999986162` | `0.0001` |
