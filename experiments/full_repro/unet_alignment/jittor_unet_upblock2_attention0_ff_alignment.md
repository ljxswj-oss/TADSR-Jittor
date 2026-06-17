# 中文阅读说明：jittor_unet_upblock2_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.attentions.0 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `9.179115295410156e-06` | `2.2129327565553748e-06` | `7.245021021660439e-06` | `0.9999999999753637` | `0.0001` |
| ff_geglu_proj | `2.47955322265625e-05` | `3.2608207406359392e-06` | `5.289885366524749e-06` | `0.9999999999871868` | `0.0001` |
| ff_geglu_output | `2.276897430419922e-05` | `1.2140057595541404e-06` | `9.590641105212973e-06` | `0.9999999999566787` | `0.0001` |
| ff_output | `2.2470951080322266e-05` | `5.461708857623648e-06` | `9.446446586926474e-06` | `0.9999999999577853` | `0.001` |
| transformer0_output | `6.866455078125e-05` | `1.354277212328725e-05` | `7.306693331292052e-06` | `0.9999999999798952` | `0.001` |
