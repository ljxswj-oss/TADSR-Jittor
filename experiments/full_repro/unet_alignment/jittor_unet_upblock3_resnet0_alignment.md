# 中文阅读说明：jittor_unet_upblock3_resnet0_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.resnets.0 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| concat_input | `0.0` | `0.0` | `0.0` | `1.0` | `1e-06` |
| norm1 | `4.76837158203125e-07` | `2.2333670912673177e-08` | `6.157521390886978e-08` | `0.9999999999999974` | `0.0001` |
| conv1 | `1.7881393432617188e-06` | `1.830683490133822e-07` | `2.735375272673643e-07` | `0.9999999999999631` | `0.0002` |
| time_emb_proj | `2.384185791015625e-07` | `2.569431671872735e-08` | `1.075543226928517e-07` | `0.9999999999999922` | `0.0001` |
| after_temb_add | `1.9073486328125e-06` | `1.8719873011718847e-07` | `2.60625987373152e-07` | `0.9999999999999669` | `0.0002` |
| norm2 | `2.384185791015625e-06` | `1.6119294050792598e-07` | `2.814119760461301e-07` | `0.999999999999955` | `0.0001` |
| conv2 | `2.2649765014648438e-06` | `3.3675691071266554e-07` | `5.306078831938887e-07` | `0.9999999999998592` | `0.0001` |
| shortcut | `3.337860107421875e-06` | `2.1528592597519491e-07` | `2.167208613864919e-07` | `0.9999999999999757` | `0.0001` |
| output | `3.5762786865234375e-06` | `4.062679465732799e-07` | `3.4812133431780633e-07` | `0.9999999999999387` | `0.0001` |

Note: Only up_blocks.3.resnets.0 is executed after official residual concat. It stops before up_blocks.3.attentions.0/resnets.1, full up_blocks.3, full UNet forward, runtime LoRA, and full TADSR inference.
