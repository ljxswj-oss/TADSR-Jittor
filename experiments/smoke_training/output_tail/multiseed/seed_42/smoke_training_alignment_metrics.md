# 中文阅读说明：smoke_training_alignment_metrics.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Smoke Training Alignment Metrics

Overall status: **PASS**

This analysis compares the PyTorch and Jittor validation predictions for
the real TADSR output-tail `conv_out` smoke training task. It does not
represent full TADSR training or image/video generation.

PASS thresholds are set to prediction relative L2 < 0.005 and final
validation-loss relative gap < 0.02. These tolerances are intentionally
small but account for PyTorch/Jittor optimizer and CPU convolution kernel
differences during independent training runs.

## Prediction Alignment

| Metric | Value |
|---|---:|
| Relative L2 diff | 2.48876211e-04 |
| MAE | 1.49764915e-05 |
| Max abs diff | 8.42362642e-05 |
| Cosine similarity | 0.99999997 |

## Loss-Curve Alignment

| Metric | Value |
|---|---:|
| Final train relative gap | 6.78616684e-05 |
| Final validation relative gap | 1.01900115e-04 |
| Train loss correlation | 0.99999995 |
| Validation loss correlation | 0.99999996 |

- TADSR_SMOKE_TRAINING_PREDICTION_ALIGNMENT: PASS
- TADSR_SMOKE_TRAINING_VALIDATION_ALIGNMENT: PASS
