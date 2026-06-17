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
| Train initial loss | 0.00834443 | 0.00834442 |
| Train final loss | 0.00011204 | 0.00011203 |
| Train loss drop ratio | 0.986573 | 0.986574 |
| Validation initial loss | 0.00769632 | 0.00769633 |
| Validation final loss | 0.00020333 | 0.00020336 |
| Validation loss drop ratio | 0.973580 | 0.973578 |
| Mean step time sec | 0.001241 | 0.001054 |

- Final train loss relative gap: 0.000068
- Final validation loss relative gap: 0.000102
- Train loss correlation: 1.000000
- Validation loss correlation: 1.000000
- No NaN/Inf: True
- Loss curve: `experiments/smoke_training/output_tail/train_val_loss_curve.png`
