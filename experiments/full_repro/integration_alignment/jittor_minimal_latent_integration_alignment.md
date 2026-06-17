# 中文阅读说明：jittor_minimal_latent_integration_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Minimal Latent Integration Alignment

Status: **PASS**

| Stage | Status | Max abs error | Mean abs error | Cosine | Tolerance |
|---|---|---:|---:|---:|---:|
| posterior_sample | `PASS` | `2.5272369384765625e-05` | `1.4443925238083466e-06` | `0.9999999999999597` | `0.002` |
| scaled_latent | `PASS` | `4.649162292480469e-06` | `2.62171056419902e-07` | `0.9999999999999595` | `0.002` |
| unet_model_pred | `PASS` | `1.0102987289428711e-05` | `9.058893795099721e-07` | `0.9999999999817799` | `0.002` |
| alpha_prod_t | `PASS` | `0.0` | `0.0` | `1.0` | `1e-08` |
| sqrt_alpha_prod_t | `PASS` | `0.0` | `0.0` | `1.0` | `1e-08` |
| x0_from_res | `PASS` | `1.049041748046875e-05` | `9.188615877064876e-07` | `0.999999999999487` | `0.002` |
| decode_input | `PASS` | `5.7220458984375e-05` | `5.048095772508532e-06` | `0.9999999999994863` | `0.002` |
| decoded_output | `PASS` | `5.9586018323898315e-06` | `2.6296383263494744e-07` | `0.9999999999998136` | `0.002` |
| final_clamped_output | `PASS` | `5.9586018323898315e-06` | `2.546225535600873e-07` | `0.999999999999816` | `0.002` |

No full denoising loop, production full inference, image/video generation, image save, or generic runtime LoRA was executed.

```json
{
  "latent_only": true,
  "decode_boundary": "PASS_TENSOR_ONLY_DECODE_CLAMP_BOUNDARY",
  "scheduler_step_executed": false,
  "full_denoising_loop_executed": false,
  "vae_decode_executed": true,
  "image_saved": false,
  "full_tadsr_inference_executed": false,
  "production_cli_used": false,
  "runtime_lora_dynamic_loading": false
}
```
