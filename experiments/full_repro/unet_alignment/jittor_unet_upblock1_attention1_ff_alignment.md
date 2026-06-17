# 中文阅读说明：jittor_unet_upblock1_attention1_ff_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.1.attentions.1 Feed-forward Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm3 | `3.874301910400391e-06` | `6.525325828565798e-07` | `2.8906903008123292e-06` | `0.9999999999960675` | `0.0001` |
| ff_geglu_proj | `1.430511474609375e-05` | `1.5828763266778623e-06` | `1.925837319508693e-06` | `0.9999999999988823` | `0.0001` |
| ff_geglu_output | `7.68899917602539e-06` | `3.059199835007048e-07` | `4.264535230631321e-06` | `0.9999999999909727` | `0.001` |
| ff_output | `9.5367431640625e-06` | `1.6754222398773777e-06` | `3.862657140018532e-06` | `0.9999999999959095` | `0.001` |
| transformer0_output | `9.5367431640625e-05` | `4.041934266751923e-06` | `3.1391875750230053e-06` | `0.9999999999953545` | `0.001` |
