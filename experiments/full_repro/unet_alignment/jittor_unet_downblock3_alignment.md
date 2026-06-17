# 中文阅读说明：jittor_unet_downblock3_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Full Local down_blocks.3 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| downblock3_resnet0_output | `7.62939453125e-06` | `6.143059181340504e-07` | `2.3127072980770804e-07` | `0.9999999999999732` | `0.002` |
| downblock3_resnet1_output | `7.62939453125e-06` | `7.068809281918221e-07` | `2.6628398236413027e-07` | `0.9999999999999654` | `0.002` |
| downblock3_hidden_states_output | `7.62939453125e-06` | `7.068809281918221e-07` | `2.6628398236413027e-07` | `0.9999999999999654` | `0.002` |
| output_state_0 | `7.62939453125e-06` | `6.143059181340504e-07` | `2.3127072980770804e-07` | `0.9999999999999732` | `0.002` |
| output_state_1 | `7.62939453125e-06` | `7.068809281918221e-07` | `2.6628398236413027e-07` | `0.9999999999999654` | `0.002` |

Note: Completes local down_blocks.3 hidden output and residual output_states tuple only. mid_block, up_blocks and full UNet forward remain NotImplemented.
