# 中文阅读说明：jittor_unet_upblock2_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.attentions.1 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `1.0907649993896484e-05` | `2.3381390796977515e-06` | `8.415146601582566e-06` | `0.9999999999690826` | `0.0001` |
| ff_geglu_proj | `3.8743019104003906e-05` | `3.380077563153172e-06` | `6.468145186564811e-06` | `0.9999999999788586` | `0.0001` |
| ff_geglu_output | `3.8623809814453125e-05` | `1.2725186318366822e-06` | `1.1591080330904914e-05` | `0.9999999999335262` | `0.0001` |
| ff_output | `4.673004150390625e-05` | `5.528114597375122e-06` | `1.1059615299599578e-05` | `0.9999999999423465` | `0.001` |
| transformer0_output | `5.078315734863281e-05` | `1.1030992448013422e-05` | `9.337827416743291e-06` | `0.9999999999636658` | `0.001` |
