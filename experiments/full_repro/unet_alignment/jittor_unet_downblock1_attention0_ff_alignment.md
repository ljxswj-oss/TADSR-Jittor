# 中文阅读说明：jittor_unet_downblock1_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.1.attentions.0 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `2.205371856689453e-05` | `2.241092028393288e-06` | `9.882651042901133e-06` | `0.9999999999480366` | `0.0001` |
| ff_geglu_proj | `3.421306610107422e-05` | `3.022975103728065e-06` | `7.0380608075947216e-06` | `0.9999999999774319` | `0.0001` |
| ff_geglu_output | `1.7881393432617188e-05` | `7.568786407934058e-07` | `1.231955914699643e-05` | `0.9999999999363125` | `0.001` |
| ff_output | `1.436471939086914e-05` | `2.9839771038098204e-06` | `1.0936195241171268e-05` | `0.9999999999446779` | `0.001` |
| transformer0_output | `5.269050598144531e-05` | `6.623554048701407e-06` | `1.0313242159408332e-05` | `0.9999999999438531` | `0.001` |
