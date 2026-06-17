# 中文阅读说明：jittor_unet_upblock1_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.1.attentions.0 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `3.844499588012695e-06` | `8.927567509510459e-07` | `3.872726041171053e-06` | `0.9999999999925843` | `0.0001` |
| ff_geglu_proj | `2.4318695068359375e-05` | `1.7915893346653888e-06` | `2.653262540664655e-06` | `0.9999999999967026` | `0.0001` |
| ff_geglu_output | `2.1576881408691406e-05` | `4.3666038366548367e-07` | `5.041738898991086e-06` | `0.9999999999867508` | `0.001` |
| ff_output | `1.1622905731201172e-05` | `2.616015375878078e-06` | `5.008816888239695e-06` | `0.9999999999894357` | `0.001` |
| transformer0_output | `7.62939453125e-05` | `5.069045755590196e-06` | `4.166550628317417e-06` | `0.999999999990751` | `0.001` |
