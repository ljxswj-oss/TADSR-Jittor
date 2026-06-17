# 中文阅读说明：gpu_selection.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Smoke Training GPU Selection

Selected GPU: `2`
CPU fallback: `False`

| GPU | Name | Memory Used MB | Total MB | Util % |
|---:|---|---:|---:|---:|
| 0 | NVIDIA GeForce RTX 4090 | 21625 | 24564 | 64 |
| 1 | NVIDIA GeForce RTX 4090 | 14630 | 24564 | 0 |
| 2 | NVIDIA GeForce RTX 4090 | 914 | 24564 | 8 |
| 3 | NVIDIA GeForce RTX 4090 | 936 | 24564 | 16 |
| 4 | NVIDIA GeForce RTX 4090 | 1874 | 24564 | 89 |
| 5 | NVIDIA GeForce RTX 4090 | 1524 | 24564 | 18 |
