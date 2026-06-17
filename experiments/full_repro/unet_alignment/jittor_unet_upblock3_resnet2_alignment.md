# 中文阅读说明：jittor_unet_upblock3_resnet2_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet up_blocks.3.resnets.2 Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| concat_input | `0.0` | `0.0` | `0.0` | `0.9999999999999998` | `1e-06` |
| norm1 | `2.384185791015625e-07` | `7.3338969644261746e-09` | `6.806425329035773e-08` | `0.9999999999999974` | `0.0001` |
| conv1 | `1.1920928955078125e-06` | `8.490121166104813e-08` | `2.0649189782193398e-07` | `0.9999999999999791` | `0.0002` |
| time_emb_proj | `1.1920928955078125e-07` | `1.856533344835043e-08` | `1.121344098188147e-07` | `0.9999999999999918` | `0.0001` |
| after_temb_add | `1.1920928955078125e-06` | `8.881822992634625e-08` | `2.0655543843260454e-07` | `0.9999999999999792` | `0.0002` |
| norm2 | `1.5497207641601562e-06` | `1.1508425051855175e-07` | `2.4543879758838004e-07` | `0.9999999999999684` | `0.0001` |
| conv2 | `1.9073486328125e-06` | `2.522557080553156e-07` | `2.658023298816416e-07` | `0.999999999999964` | `0.0001` |
| shortcut | `1.3113021850585938e-06` | `1.0698144490284278e-07` | `1.4553164590081098e-07` | `0.9999999999999886` | `0.0001` |
| output | `2.1457672119140625e-06` | `2.783274709372563e-07` | `3.146907552133618e-07` | `0.9999999999999523` | `0.0001` |

Note: Only up_blocks.3.resnets.2 is executed after official residual concat. It stops before up_blocks.3.attentions.2, full up_blocks.3, full UNet forward, runtime LoRA, and full TADSR inference.
