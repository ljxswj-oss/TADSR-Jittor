# 中文阅读说明：prepared_assets_manifest.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Prepared Official Assets Manifest

Input weights: `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights`

Output weights: `/mnt/data/sj/checkpoints/TADSR/preset/weights`

| Item | Source exists | Target | Link target | Broken symlink |
|---|---:|---|---|---:|
| `time_vae` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/time_vae` | false |
| `unet` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/unet` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/unet` | false |
| `vae` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/vae` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/vae` | false |
| `text_encoder` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/text_encoder` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/text_encoder` | false |
| `tokenizer` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/tokenizer` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/tokenizer` | false |
| `scheduler` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/scheduler` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/scheduler` | false |
| `feature_extractor` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/feature_extractor` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/feature_extractor` | false |
| `bert-base-uncased` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/bert-base-uncased` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/bert-base-uncased` | false |
| `DAPE.pth` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/DAPE.pth` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/DAPE.pth` | false |
| `ram_swin_large_14m.pth` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/ram_swin_large_14m.pth` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/ram_swin_large_14m.pth` | false |
| `tadsr.pkl` | true | `/mnt/data/sj/checkpoints/TADSR/preset/weights/tadsr.pkl` | `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights/tadsr.pkl` | false |
