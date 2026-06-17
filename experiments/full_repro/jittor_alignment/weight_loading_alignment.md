# 中文阅读说明：weight_loading_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Weight Loading Alignment

Status: **PASS**

| Component | Status | Key count | Path |
|---|---|---:|---|
| `tadsr` | PASS | 606 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/tadsr_weights.npz` |
| `DAPE` | PASS | 248 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/DAPE_weights.npz` |
| `RAM` | PASS | 2982 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/ram_swin_large_14m_weights.npz` |
| `unet` | PASS | 686 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/unet_weights.npz` |
| `vae` | PASS | 248 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/vae_weights.npz` |
| `time_vae` | PASS | 272 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz` |
| `text_encoder` | PASS | 373 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/text_encoder_weights.npz` |
| `bert_base_uncased` | PASS | 207 | `/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/bert_base_uncased_weights.npz` |
