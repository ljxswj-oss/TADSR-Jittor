# 中文阅读说明：jittor_decoder_entry_midblock_upblocks0123_tail_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE Decoder Entry+MidBlock+UpBlocks0-3+Tail Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 3, 32, 32]` |
| max_abs_error | `2.384185791015625e-06` |
| mean_abs_error | `2.509790040979472e-07` |
| relative_error | `8.708034425873334e-07` |
| cosine_similarity | `0.9999999999993984` |
| tolerance | `0.002` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| post_quant_conv | `5.960464477539063e-08` | `4.190951585769653e-09` | `1.4472931614908453e-08` | `0.9999999999999994` | `0.002` |
| decoder_conv_in | `2.384185791015625e-07` | `1.1058745030823047e-08` | `5.798317694904203e-08` | `0.9999999999999974` | `0.002` |
| decoder_midblock | `2.384185791015625e-06` | `3.6261644709156826e-07` | `3.73693783753209e-07` | `0.9999999999999295` | `0.002` |
| decoder_upblock0 | `1.33514404296875e-05` | `1.7265455696247045e-06` | `5.789675771425907e-07` | `0.9999999999998436` | `0.002` |
| decoder_upblock1 | `9.1552734375e-05` | `8.123663750581045e-06` | `7.502074734579309e-07` | `0.9999999999997227` | `0.002` |
| decoder_upblock2 | `0.00037384033203125` | `1.9698307040627583e-05` | `1.386328893226877e-06` | `0.999999999998697` | `0.002` |
| decoder_upblock3 | `0.00128173828125` | `4.334792453164482e-05` | `1.9956227696869127e-06` | `0.9999999999963466` | `0.002` |
| decoder_tail_norm_out | `3.838539123535156e-05` | `1.1149397802185735e-06` | `2.7103710343712474e-06` | `0.9999999999934749` | `0.002` |
| decoder_tail_act | `1.6897916793823242e-05` | `3.3820542715511364e-07` | `2.221356483328897e-06` | `0.9999999999950636` | `0.002` |
| decoder_tail_conv_out | `2.384185791015625e-06` | `2.509790040979472e-07` | `8.708034425873334e-07` | `0.9999999999993984` | `0.002` |
| decoder_tail | `2.384185791015625e-06` | `2.509790040979472e-07` | `8.708034425873334e-07` | `0.9999999999993984` | `0.002` |

Note: deterministic decoder stack bridge to tail; uses 2e-3 tolerance for accumulated float32 drift; this is not full VAE or full TADSR inference
