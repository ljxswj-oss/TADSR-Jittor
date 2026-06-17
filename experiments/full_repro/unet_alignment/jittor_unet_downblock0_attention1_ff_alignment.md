# 中文阅读说明：jittor_unet_downblock0_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.0.attentions.1 FF Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `5.543231964111328e-06` | `1.1209657079469104e-06` | `3.245018149592327e-06` | `0.9999999999947286` | `0.0001` |
| ff_geglu_proj | `7.3909759521484375e-06` | `1.1311207596897966e-06` | `2.7910648566816365e-06` | `0.9999999999961972` | `0.0001` |
| ff_output | `7.852911949157715e-06` | `1.3259743690596792e-06` | `4.619417601054244e-06` | `0.9999999999895703` | `0.001` |
| transformer0_output | `9.804964065551758e-06` | `1.7393367642171142e-06` | `3.7833550818169824e-06` | `0.9999999999931395` | `0.001` |

Note: Stops before proj_out.
