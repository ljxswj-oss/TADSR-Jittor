# 中文阅读说明：jittor_unet_upblock2_attention2_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.attentions.2 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `1.7434358596801758e-05` | `3.851855912956337e-06` | `1.5317352640357114e-05` | `0.9999999998871575` | `0.0001` |
| ff_geglu_proj | `9.465217590332031e-05` | `4.683359844570578e-06` | `1.053474336214867e-05` | `0.9999999999394913` | `0.0001` |
| ff_geglu_output | `3.0040740966796875e-05` | `1.2601434213250219e-06` | `1.8830730874644827e-05` | `0.999999999803231` | `0.0001` |
| ff_output | `3.8623809814453125e-05` | `6.286818080880608e-06` | `2.0337202857868123e-05` | `0.9999999999026664` | `0.001` |
| transformer0_output | `6.073713302612305e-05` | `1.3861702927897569e-05` | `1.5636131725976824e-05` | `0.999999999879836` | `0.001` |
