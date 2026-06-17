# 中文阅读说明：multiseed_summary.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Multi-Seed Smoke Training Stability

Status: **PASS**

| Seed | PyTorch val final | Jittor val final | Val relative gap | Prediction relative L2 | Status |
|---:|---:|---:|---:|---:|---|
| 1234 | 0.00053915 | 0.00054116 | 3.71002037e-03 | 2.30811322e-03 | PASS |
| 2025 | 0.00021354 | 0.00021356 | 1.00845096e-04 | 4.40415484e-04 | PASS |
| 42 | 0.00020333 | 0.00020336 | 1.01900115e-04 | 2.48876211e-04 | PASS |

- mean_jittor_final_val_loss: 0.0003193581699936961
- std_jittor_final_val_loss: 0.00015688989278442506
- mean_val_loss_relative_gap: 0.001304255192851761

This is a small-data output-tail training-loop stability check, not full TADSR training.
