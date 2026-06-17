# 中文阅读说明：jittor_unet_upblock3_attention0_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.attentions.0 Feed-Forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `8.299946784973145e-06` | `1.4760260392224222e-06` | `3.056001493486463e-06` | `0.9999999999954321` | `0.0001` |
| ff_geglu_proj | `8.821487426757812e-06` | `1.182693859547168e-06` | `2.3406158200236935e-06` | `0.9999999999972756` | `0.0001` |
| ff_geglu_output | `1.2874603271484375e-05` | `6.136883473578343e-07` | `4.103376444073613e-06` | `0.9999999999931256` | `0.0001` |
| ff_output | `8.761882781982422e-06` | `1.837590108966225e-06` | `4.729944673825917e-06` | `0.9999999999896902` | `0.001` |
| transformer0_output | `1.6808509826660156e-05` | `2.6080048087351313e-06` | `3.5997598672449925e-06` | `0.9999999999937006` | `0.001` |
