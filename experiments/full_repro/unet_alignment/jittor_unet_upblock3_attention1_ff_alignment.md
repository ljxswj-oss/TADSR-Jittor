# 中文阅读说明：jittor_unet_upblock3_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.attentions.1 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `8.07642936706543e-06` | `1.7186920717492463e-06` | `4.010248831952932e-06` | `0.9999999999914473` | `0.0001` |
| ff_geglu_proj | `1.4543533325195312e-05` | `1.4250515957082987e-06` | `3.5169272690363693e-06` | `0.9999999999935216` | `0.0001` |
| ff_geglu_output | `7.748603820800781e-06` | `5.434562869427631e-07` | `5.5777886340176465e-06` | `0.9999999999877172` | `0.0001` |
| ff_output | `6.67572021484375e-06` | `1.510394378456681e-06` | `5.565349023842451e-06` | `0.9999999999850793` | `0.001` |
| transformer0_output | `1.430511474609375e-05` | `2.7606187757101e-06` | `4.163919689931624e-06` | `0.9999999999913821` | `0.001` |
