# 中文阅读说明：jittor_unet_downblock0_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet attention0 FF Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `5.9604644775390625e-06` | `1.2569394206879414e-06` | `4.1097762583887145e-06` | `0.9999999999914639` | `0.0001` |
| ff_geglu_proj | `1.1682510375976562e-05` | `1.1488335286691242e-06` | `2.691532892190253e-06` | `0.9999999999964682` | `0.0001` |
| ff_output | `3.6656856536865234e-06` | `8.446310183707339e-07` | `4.160452273322196e-06` | `0.9999999999916812` | `0.001` |
| transformer0_output | `9.268522262573242e-06` | `1.5968250266951145e-06` | `4.2742869299669646e-06` | `0.9999999999908823` | `0.001` |
