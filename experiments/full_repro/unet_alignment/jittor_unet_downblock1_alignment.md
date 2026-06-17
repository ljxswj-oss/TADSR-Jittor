# 中文阅读说明：jittor_unet_downblock1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Full Local down_blocks.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| downblock1_resnet0_output | `4.0531158447265625e-06` | `3.108353894276661e-07` | `3.203363142094117e-07` | `0.9999999999999482` | `0.002` |
| downblock1_attention0_output | `6.9141387939453125e-06` | `9.39450683290488e-07` | `8.868342742026402e-07` | `0.9999999999996102` | `0.002` |
| downblock1_resnet1_output | `6.318092346191406e-06` | `1.0323331792960745e-06` | `9.94641730344596e-07` | `0.9999999999995235` | `0.002` |
| downblock1_attention1_output | `3.4332275390625e-05` | `1.066969463181522e-06` | `1.1697552748512852e-06` | `0.9999999999993859` | `0.002` |
| downblock1_downsampler_output | `1.2636184692382812e-05` | `2.071532463787662e-06` | `8.653053221313012e-07` | `0.9999999999996687` | `0.002` |
| downblock1_hidden_states_output | `1.2636184692382812e-05` | `2.071532463787662e-06` | `8.653053221313012e-07` | `0.9999999999996687` | `0.002` |
| output_state_0 | `6.9141387939453125e-06` | `9.39450683290488e-07` | `8.868342742026402e-07` | `0.9999999999996102` | `0.002` |
| output_state_1 | `3.4332275390625e-05` | `1.066969463181522e-06` | `1.1697552748512852e-06` | `0.9999999999993859` | `0.002` |
| output_state_2 | `1.2636184692382812e-05` | `2.071532463787662e-06` | `8.653053221313012e-07` | `0.9999999999996687` | `0.002` |

Note: Completes local down_blocks.1 only. Full UNet forward, down_blocks.2, mid_block, up_blocks and full TADSR inference remain NotImplemented.
