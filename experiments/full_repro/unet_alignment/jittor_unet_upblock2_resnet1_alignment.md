# 中文阅读说明：jittor_unet_upblock2_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.resnets.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| concat_input | `0.0` | `0.0` | `0.0` | `1.0` | `1e-06` |
| norm1 | `9.5367431640625e-07` | `8.393661744300618e-09` | `6.269029924300027e-08` | `0.999999999999997` | `0.0001` |
| conv1 | `3.814697265625e-06` | `1.267831809315112e-07` | `2.4948893315497943e-07` | `0.9999999999999649` | `0.0002` |
| time_emb_proj | `4.172325134277344e-07` | `2.3987922759260982e-08` | `1.0606681194180571e-07` | `0.9999999999999913` | `0.0001` |
| after_temb_add | `3.814697265625e-06` | `1.3245608343481763e-07` | `2.4364378100896757e-07` | `0.9999999999999679` | `0.0002` |
| norm2 | `5.245208740234375e-06` | `1.723123661145909e-07` | `2.6868825478949683e-07` | `0.9999999999999638` | `0.0001` |
| conv2 | `3.337860107421875e-06` | `5.055642365192625e-07` | `4.421948776850375e-07` | `0.999999999999902` | `0.0001` |
| shortcut | `3.814697265625e-06` | `3.4331246752117294e-07` | `1.9638553570177143e-07` | `0.9999999999999807` | `0.0001` |
| output | `4.291534423828125e-06` | `6.20188542654887e-07` | `3.005983681311281e-07` | `0.9999999999999544` | `0.0001` |

Note: Only up_blocks.2.resnets.1 is executed after official residual concat. It stops before up_blocks.2.attentions.1, full up_blocks.2, full UNet forward, runtime LoRA, and full TADSR inference.
