# 中文阅读说明：jittor_unet_midblock_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet mid_block.attentions.0 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `4.291534423828125e-06` | `7.484986999095611e-07` | `2.7605142376262093e-06` | `0.9999999999962225` | `0.0001` |
| ff_geglu_proj | `7.987022399902344e-06` | `2.866831593895114e-07` | `2.5802941651942983e-06` | `0.9999999999972098` | `0.0002` |
| ff_geglu_output | `1.6450881958007812e-05` | `9.099587128073104e-08` | `4.228398768495942e-06` | `0.9999999999921229` | `0.001` |
| ff_output | `5.4836273193359375e-06` | `1.2595708653861948e-06` | `3.917807288660054e-06` | `0.9999999999928697` | `0.001` |
| transformer0_output | `1.385807991027832e-05` | `2.720778047660133e-06` | `3.06416541760476e-06` | `0.9999999999954702` | `0.001` |
