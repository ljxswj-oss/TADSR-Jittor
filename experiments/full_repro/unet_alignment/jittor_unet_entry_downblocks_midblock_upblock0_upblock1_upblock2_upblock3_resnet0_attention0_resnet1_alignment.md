# 中文阅读说明：jittor_unet_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry -> DownBlocks -> MidBlock -> UpBlock0 -> UpBlock1 -> UpBlock2 -> UpBlock3 Resnet0 -> Attention0 -> Resnet1 Bridge Alignment

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
| upblock0_output_hidden | `1.704692840576172e-05` | `7.323542648762782e-07` | `1.2861741591879397e-06` | `0.999999999999486` | `0.002` |
| upblock1_output_hidden | `8.705258369445801e-05` | `7.895466114504756e-06` | `1.889558076920784e-06` | `0.999999999998297` | `0.002` |
| upblock2_output_hidden | `8.535385131835938e-05` | `5.06996946665339e-06` | `2.210366932101455e-06` | `0.9999999999979134` | `0.002` |
| upblock3_resnet0_output | `1.6927719116210938e-05` | `2.550648282806378e-06` | `2.1855898085866294e-06` | `0.9999999999975839` | `0.002` |
| upblock3_attention0_output | `2.2172927856445312e-05` | `3.021396412350441e-06` | `2.1506846311867753e-06` | `0.9999999999977839` | `0.002` |
| upblock3_resnet1_res_hidden | `3.2186508178710938e-06` | `4.2014392249711817e-07` | `8.921249154252471e-07` | `0.9999999999996211` | `0.002` |
| upblock3_resnet1_concat_input | `2.2172927856445312e-05` | `1.7207701674237797e-06` | `1.8347046669429526e-06` | `0.9999999999979728` | `0.002` |
| upblock3_resnet1_output | `1.0848045349121094e-05` | `1.3504997063762402e-06` | `1.6927380818337679e-06` | `0.9999999999985686` | `0.002` |

Note: Bridge validates entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1. It deliberately stops before up_blocks.3.attentions.1 and does not mark full up_blocks.3 complete.
