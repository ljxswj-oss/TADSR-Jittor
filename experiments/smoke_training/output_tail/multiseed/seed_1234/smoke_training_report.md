# 中文阅读说明：smoke_training_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR Output-Tail Conv-Out Smoke Training Report

Status: **PASS**

This is small-data training-pipeline validation for a real TADSR
output-tail `conv_out` head. It is not full TADSR training and does
not run full inference or generate images/videos.

## Metrics

| Metric | PyTorch | Jittor |
|---|---:|---:|
| Train initial loss | 0.00523660 | 0.00523660 |
| Train final loss | 0.00239464 | 0.00236985 |
| Train loss drop ratio | 0.542711 | 0.547445 |
| Validation initial loss | 0.00481461 | 0.00481459 |
| Validation final loss | 0.00053915 | 0.00054116 |
| Validation loss drop ratio | 0.888018 | 0.887601 |
| Mean step time sec | 0.001276 | 0.001135 |

- Final train loss relative gap: 0.010354
- Final validation loss relative gap: 0.003710
- Train loss correlation: 0.999954
- Validation loss correlation: 0.999977
- No NaN/Inf: True
- Loss curve: `experiments/smoke_training/output_tail/train_val_loss_curve.png`
