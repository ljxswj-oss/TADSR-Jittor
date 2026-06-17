# 中文阅读说明：jittor_unet_upblock3_attention2_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.attentions.2 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `5.9604644775390625e-06` | `1.3110910062701618e-06` | `3.2477922059852866e-06` | `0.9999999999949268` | `0.0001` |
| ff_geglu_proj | `1.0192394256591797e-05` | `1.1704816261046136e-06` | `3.064788882224978e-06` | `0.9999999999951084` | `0.0001` |
| ff_geglu_output | `1.2993812561035156e-05` | `3.107378490130794e-07` | `4.886678511880766e-06` | `0.9999999999886625` | `0.0001` |
| ff_output | `5.185604095458984e-06` | `1.1079449235396055e-06` | `4.884973764420477e-06` | `0.9999999999873506` | `0.001` |
| transformer0_output | `9.357929229736328e-06` | `1.8736238303063146e-06` | `3.6225776475912203e-06` | `0.9999999999941211` | `0.001` |
