# 中文阅读说明：jittor_unet_upblock2_attention2_proj_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.attentions.2 Projection Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm | `7.212162017822266e-06` | `2.763199609507466e-06` | `7.801765515797704e-06` | `0.9999999999601805` | `0.0001` |
| sequence_input | `7.212162017822266e-06` | `2.763199609507466e-06` | `7.801765515797704e-06` | `0.9999999999601795` | `0.0001` |
| proj_in | `2.8073787689208984e-05` | `5.153772181287764e-06` | `1.0133525754982106e-05` | `0.9999999999468853` | `0.0002` |
