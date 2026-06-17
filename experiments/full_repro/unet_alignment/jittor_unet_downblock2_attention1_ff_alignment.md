# 中文阅读说明：jittor_unet_downblock2_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.2.attentions.1 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `3.0463561415672302e-06` | `6.639690742105131e-07` | `3.450630804695938e-06` | `0.9999999999941114` | `0.0001` |
| ff_geglu_proj | `1.2874603271484375e-05` | `1.3572877453418841e-06` | `1.691379303067478e-06` | `0.9999999999989763` | `0.0001` |
| ff_geglu_output | `8.52346420288086e-06` | `2.3380058628590279e-07` | `4.392184031195224e-06` | `0.9999999999904055` | `0.001` |
| ff_output | `8.106231689453125e-06` | `1.2444822282020595e-06` | `3.593211782282242e-06` | `0.9999999999966186` | `0.001` |
| transformer0_output | `2.9087066650390625e-05` | `3.664563810445998e-06` | `3.790872659756759e-06` | `0.9999999999933266` | `0.001` |
