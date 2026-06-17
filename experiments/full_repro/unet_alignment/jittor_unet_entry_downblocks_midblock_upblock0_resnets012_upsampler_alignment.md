# 中文阅读说明：jittor_unet_entry_downblocks_midblock_upblock0_resnets012_upsampler_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry -> DownBlocks -> MidBlock -> UpBlock0 ResNets012 -> Upsampler Bridge Alignment

Status: **PASS**

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| conv_in | `8.344650268554688e-07` | `1.2484131417522804e-08` | `8.945803719865577e-08` | `0.9999999999999959` | `0.0001` |
| time_embedding | `4.76837158203125e-07` | `4.2749661588459276e-09` | `2.484385767769718e-07` | `0.9999999999999917` | `0.0001` |
| downblock0_output | `1.2159347534179688e-05` | `1.706075145335717e-06` | `1.1445761910420367e-06` | `0.9999999999994224` | `0.002` |
| downblock1_output | `2.2277235984802246e-05` | `2.579873682861944e-06` | `1.0776458816026287e-06` | `0.9999999999994817` | `0.002` |
| downblock2_output | `3.0517578125e-05` | `3.670575767955597e-06` | `1.3387986697652404e-06` | `0.9999999999991991` | `0.002` |
| downblock3_output | `2.47955322265625e-05` | `3.7183403492235813e-06` | `1.4007089970714264e-06` | `0.9999999999991385` | `0.002` |
| midblock_hidden_output | `2.6702880859375e-05` | `3.800192098424304e-06` | `1.3949179221757381e-06` | `0.9999999999991097` | `0.002` |
| upblock0_resnet0_output | `1.9073486328125e-05` | `1.8840433540390222e-06` | `1.1846797318030327e-06` | `0.9999999999994632` | `0.002` |
| upblock0_resnet1_output | `1.7642974853515625e-05` | `1.4457379279519956e-06` | `1.1873439544292683e-06` | `0.9999999999993816` | `0.002` |
| upblock0_resnet2_output | `7.62939453125e-06` | `7.90000240158406e-07` | `1.3618544663611344e-06` | `0.9999999999992338` | `0.002` |
| upblock0_upsampler_interpolation_output | `7.62939453125e-06` | `7.90000240158406e-07` | `1.3618544663611344e-06` | `0.999999999999234` | `0.002` |
| upblock0_upsampler_conv_output | `1.704692840576172e-05` | `7.323542648762782e-07` | `1.2861741591879397e-06` | `0.999999999999486` | `0.002` |
| upblock0_upsampler_output | `1.704692840576172e-05` | `7.323542648762782e-07` | `1.2861741591879397e-06` | `0.999999999999486` | `0.002` |

Note: Bridge validates entry -> all down_blocks -> full local mid_block -> up_blocks.0.resnets.0/1/2 -> upsamplers.0. It still does not execute up_blocks.1 or full UNet.
