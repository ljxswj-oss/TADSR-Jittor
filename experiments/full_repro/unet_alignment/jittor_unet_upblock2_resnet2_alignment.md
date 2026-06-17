# 中文阅读说明：jittor_unet_upblock2_resnet2_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.2.resnets.2 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| concat_input | `0.0` | `0.0` | `0.0` | `1.0` | `1e-06` |
| norm1 | `2.384185791015625e-07` | `7.88851807221232e-09` | `6.826741864611764e-08` | `0.9999999999999969` | `0.0001` |
| conv1 | `1.9073486328125e-06` | `9.265431355487408e-08` | `2.414505403757355e-07` | `0.9999999999999706` | `0.0002` |
| time_emb_proj | `4.76837158203125e-07` | `2.2233143681660293e-08` | `1.0829313612802824e-07` | `0.9999999999999901` | `0.0001` |
| after_temb_add | `1.9073486328125e-06` | `9.819340363037554e-08` | `2.1709769406797885e-07` | `0.9999999999999772` | `0.0002` |
| norm2 | `2.384185791015625e-06` | `1.0882179242166501e-07` | `2.437273935129518e-07` | `0.9999999999999741` | `0.0001` |
| conv2 | `2.0265579223632812e-06` | `2.700515748799148e-07` | `4.158346780784866e-07` | `0.9999999999999145` | `0.0001` |
| shortcut | `5.245208740234375e-06` | `4.558602743287565e-07` | `3.5371507193188964e-07` | `0.9999999999999293` | `0.0001` |
| output | `6.198883056640625e-06` | `5.399668566496985e-07` | `3.763390767743322e-07` | `0.9999999999999237` | `0.0001` |

Note: Only up_blocks.2.resnets.2 is executed after official residual concat. It stops before up_blocks.2.attentions.2, full up_blocks.2, full UNet forward, runtime LoRA, and full TADSR inference.
