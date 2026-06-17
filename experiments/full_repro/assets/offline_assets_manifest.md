# 中文阅读说明：offline_assets_manifest.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Offline Assets Manifest

Input directory: `/mnt/data/sj/incoming/TADSR_assets/TADSR_weights`

| Item | Exists | Type | Size | Extra |
|---|---:|---|---:|---|
| `time_vae` | True | dir | 327.121 MB | `files=2` |
| `unet` | True | dir | 3.226 GB | `files=2` |
| `vae` | True | dir | 319.202 MB | `files=2` |
| `text_encoder` | True | dir | 1.268 GB | `files=2` |
| `tokenizer` | True | dir | 1.512 MB | `files=4` |
| `scheduler` | True | dir | 346.000 B | `files=1` |
| `feature_extractor` | True | dir | 342.000 B | `files=1` |
| `bert-base-uncased` | True | dir | 420.734 MB | `files=5` |
| `DAPE.pth` | True | file | 6.861 MB | `a7028be2edcbe9ab0bd1c4ab6f2a2a86f4b44d32261a4faa50ae10fdd9b2feba` |
| `ram_swin_large_14m.pth` | True | file | 5.239 GB | `15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed` |
| `tadsr.pkl` | True | file | 25.064 MB | `a04263729a7d276f64401ae7c266808a51c789eee8ba1a21b7e16849357486b9` |
