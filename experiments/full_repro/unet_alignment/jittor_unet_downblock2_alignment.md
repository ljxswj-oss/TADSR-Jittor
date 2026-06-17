# 中文阅读说明：jittor_unet_downblock2_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Full Local down_blocks.2 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| downblock2_resnet0_output | `1.621246337890625e-05` | `6.887760946483467e-07` | `4.1290576784414494e-07` | `0.9999999999999039` | `0.002` |
| downblock2_attention0_output | `1.52587890625e-05` | `1.1699399578901647e-06` | `8.66306799998582e-07` | `0.9999999999996496` | `0.002` |
| downblock2_resnet1_output | `1.430511474609375e-05` | `1.2273492075109972e-06` | `8.606696363538291e-07` | `0.9999999999996445` | `0.002` |
| downblock2_attention1_output | `1.1920928955078125e-05` | `1.4968647860769125e-06` | `1.2011930324226923e-06` | `0.9999999999993158` | `0.002` |
| downblock2_downsampler_output | `1.9073486328125e-05` | `2.7139457642988417e-06` | `9.89879301982645e-07` | `0.9999999999995772` | `0.002` |
| downblock2_hidden_states_output | `1.9073486328125e-05` | `2.7139457642988417e-06` | `9.89879301982645e-07` | `0.9999999999995772` | `0.002` |
| output_state_0 | `1.52587890625e-05` | `1.1699399578901647e-06` | `8.66306799998582e-07` | `0.9999999999996496` | `0.002` |
| output_state_1 | `1.1920928955078125e-05` | `1.4968647860769125e-06` | `1.2011930324226923e-06` | `0.9999999999993158` | `0.002` |
| output_state_2 | `1.9073486328125e-05` | `2.7139457642988417e-06` | `9.89879301982645e-07` | `0.9999999999995772` | `0.002` |

Note: Completes local down_blocks.2 only. Full UNet forward, down_blocks.3, mid_block, up_blocks and full TADSR inference remain NotImplemented.
