# 中文阅读说明：jittor_timevae_full_boundary_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeVAE Full Boundary Alignment

Status: **PASS**

## Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 3, 32, 32]` |
| max_abs_error | `5.960464477539062e-07` |
| mean_abs_error | `6.44552073936211e-08` |
| relative_error | `8.892221296813007e-08` |
| cosine_similarity | `0.9999999999999905` |
| tolerance | `0.002` |

## Diagnostics

| Stage | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---:|---:|---:|---:|---:|
| encode_moments | `1.1444091796875e-05` | `1.8557766452431679e-06` | `1.8077751391203726e-07` | `0.9999999999999692` | `0.002` |
| posterior_mean | `2.86102294921875e-06` | `5.674082785844803e-07` | `1.0820437083197998e-07` | `0.9999999999999912` | `0.002` |
| posterior_logvar | `1.1444091796875e-05` | `3.1441450119018555e-06` | `2.0567175441076968e-07` | `0.9999999999999651` | `0.002` |
| posterior_std | `7.450580596923828e-09` | `9.997620509238914e-10` | `1.11472192211895e-06` | `0.9999999999992591` | `0.002` |
| posterior_sample | `2.86102294921875e-06` | `5.674082785844803e-07` | `1.0820373657223335e-07` | `0.9999999999999913` | `0.002` |
| scaled_latent | `4.76837158203125e-07` | `9.672658052295446e-08` | `1.01265900972261e-07` | `0.9999999999999919` | `0.002` |
| decode_input | `2.86102294921875e-06` | `5.497131496667862e-07` | `1.0482930715960375e-07` | `0.9999999999999913` | `0.002` |
| decoded_output | `7.152557373046875e-07` | `1.359730201026347e-07` | `1.6111362068473322e-07` | `0.999999999999984` | `0.002` |
| final_clamped_output | `5.960464477539062e-07` | `6.44552073936211e-08` | `8.892221296813007e-08` | `0.9999999999999905` | `0.002` |

Note: alignment-only non-tiled deterministic full-boundary check; no scheduler, no image save, no full TADSR inference

## Remaining TimeVAE gaps
- official TADSR_test monkey-patches VAE encoder/decoder with VAEHook tiling for real-size inference; this boundary uses the non-tiled small-tensor path
- official pipeline uses stochastic posterior sample; this boundary uses exported fixed epsilon instead of an internal runtime RNG
- generic runtime LoRA integration remains partial; current Jittor checks use converted/static weights
- full scheduler/UNet/VAE image inference CLI remains intentionally NotImplemented
