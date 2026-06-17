# 中文阅读说明：jittor_unet_downblock1_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.1.attentions.1 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `9.521842002868652e-06` | `2.128932775316561e-06` | `8.904052170019577e-06` | `0.9999999999577325` | `0.0001` |
| ff_geglu_proj | `6.437301635742188e-05` | `3.329574982657846e-06` | `7.257522792443485e-06` | `0.9999999999708687` | `0.0001` |
| ff_geglu_output | `3.218650817871094e-05` | `6.996105923149184e-07` | `1.2523764971379588e-05` | `0.9999999998980711` | `0.001` |
| ff_output | `3.790855407714844e-05` | `3.3726168503278585e-06` | `1.1861775394503556e-05` | `0.9999999999377391` | `0.001` |
| transformer0_output | `4.208087921142578e-05` | `6.91158721224383e-06` | `9.338600272864409e-06` | `0.9999999999551206` | `0.001` |
