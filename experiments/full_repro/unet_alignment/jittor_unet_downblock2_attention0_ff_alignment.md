# 中文阅读说明：jittor_unet_downblock2_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.2.attentions.0 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `4.616566002368927e-06` | `8.849261556254095e-07` | `4.951051533729323e-06` | `0.9999999999883133` | `0.0001` |
| ff_geglu_proj | `1.8477439880371094e-05` | `1.9169856928402675e-06` | `2.9679235669042917e-06` | `0.9999999999971846` | `0.0001` |
| ff_geglu_output | `1.9490718841552734e-05` | `3.0293488275056934e-07` | `6.791761247905903e-06` | `0.9999999999691248` | `0.001` |
| ff_output | `9.328126907348633e-06` | `1.8508751562507086e-06` | `7.020443821419481e-06` | `0.999999999982406` | `0.001` |
| transformer0_output | `4.673004150390625e-05` | `4.236836116433551e-06` | `5.38777113511389e-06` | `0.9999999999856769` | `0.001` |
