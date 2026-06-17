# 中文阅读说明：smoke_training_data_summary.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Output-Tail Conv-Out Smoke Training Data

Status: **PASS**

- Task: `tadsr_output_tail_conv_out_smoke_training`
- Total samples: 32
- Train/validation split: 24/8
- Feature shape: `(32, 320, 32, 32)`
- Target shape: `(32, 4, 32, 32)`
- Train feature shape: `(24, 320, 32, 32)`
- Validation feature shape: `(8, 320, 32, 32)`
- Trainable module: output-tail `conv_out` only.
- Full TADSR training: false.
- Full inference executed: false.
- Full denoising loop executed: false.
- Image/video saved: false.

This dataset is a deterministic feature-target smoke-test set for validating
PyTorch and Jittor training loops on a real TADSR output-tail component.
Raw `.npy` tensors are intentionally ignored by git; committed artifacts keep
only metadata, logs, summaries, and diagnostic plots.
