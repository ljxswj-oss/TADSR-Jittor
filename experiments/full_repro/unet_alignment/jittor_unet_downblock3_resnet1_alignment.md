# 中文阅读说明：jittor_unet_downblock3_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet down_blocks.3.resnets.1 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| norm1 | `4.76837158203125e-07` | `1.3711407653715923e-08` | `6.024514863377002e-08` | `0.9999999999999969` | `0.0001` |
| conv1 | `2.86102294921875e-06` | `9.954225674846384e-08` | `2.1430654157802636e-07` | `0.9999999999999754` | `0.0001` |
| time_emb_proj | `3.5762786865234375e-07` | `1.5632213035132735e-08` | `1.3628367694096606e-07` | `0.9999999999999851` | `0.0001` |
| after_temb_add | `2.86102294921875e-06` | `1.0264898264722433e-07` | `2.1041054273651545e-07` | `0.9999999999999755` | `0.0001` |
| norm2 | `3.0994415283203125e-06` | `1.3201620077119003e-07` | `2.5574850032805513e-07` | `0.9999999999999701` | `0.0001` |
| conv2 | `3.5762786865234375e-06` | `3.319087113595742e-07` | `3.3945243294837374e-07` | `0.9999999999999432` | `0.0001` |
| shortcut | `0.0` | `0.0` | `0.0` | `0.9999999999999998` | `0.0001` |
| output | `3.814697265625e-06` | `3.3328724384773524e-07` | `1.255502178987302e-07` | `0.9999999999999918` | `0.0001` |

Note: Only down_blocks.3.resnets.1 is executed. Local down_blocks.3 is checked in test_unet_downblock3_local_alignment.py; mid_block/full UNet remain unopened.
