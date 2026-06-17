# 中文阅读说明：jittor_unet_upblock1_attention2_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.1.attentions.2 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `2.602115273475647e-06` | `5.802737223975507e-07` | `2.7128332518813413e-06` | `0.9999999999964571` | `0.0001` |
| ff_geglu_proj | `2.002716064453125e-05` | `1.2557491908316009e-06` | `2.026727861619603e-06` | `0.9999999999982381` | `0.0001` |
| ff_geglu_output | `6.198883056640625e-06` | `3.0405592080285685e-07` | `3.947205673312936e-06` | `0.9999999999933159` | `0.001` |
| ff_output | `2.288818359375e-05` | `1.5675189040109672e-06` | `3.22717605306916e-06` | `0.9999999999974363` | `0.001` |
| transformer0_output | `3.24249267578125e-05` | `3.486889659143344e-06` | `2.834205292587789e-06` | `0.9999999999965797` | `0.001` |
