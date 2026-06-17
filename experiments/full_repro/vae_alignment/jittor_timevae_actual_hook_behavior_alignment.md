# 中文阅读说明：jittor_timevae_actual_hook_behavior_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeVAE Actual VAEHook Behavior Alignment

Status: **PASS**

Reason: Actual VAEHook behavior alignment complete.

## Primary Metrics

| Metric | Value |
|---|---:|
| shape_match | `True` |
| shape | `[1, 3, 96, 96]` |
| max_abs_error | `3.5762786865234375e-07` |
| mean_abs_error | `5.798372310689754e-08` |
| relative_error | `1.1130083550545222e-07` |
| cosine_similarity | `0.9999999999999917` |
| tolerance | `0.002` |
| status | `PASS` |

## Diagnostics

| Stage | Status | Max abs error | Mean abs error | Relative error | Cosine | Tolerance |
|---|---|---:|---:|---:|---:|---:|
| encoder_tiled_raw_output | `PASS` | `7.104873657226562e-05` | `2.5816995427400493e-06` | `3.3700296144464805e-07` | `0.9999999999997808` | `0.002` |
| encoder_moments | `PASS` | `4.696846008300781e-05` | `3.0015685802532566e-06` | `2.491893092435908e-07` | `0.9999999999999369` | `0.002` |
| posterior_mean | `PASS` | `4.696846008300781e-05` | `2.122212511797746e-06` | `3.5586508669258484e-07` | `0.9999999999997262` | `0.002` |
| posterior_logvar | `PASS` | `2.288818359375e-05` | `3.6127037472195097e-06` | `1.9963654719173033e-07` | `0.9999999999999751` | `0.002` |
| posterior_std | `PASS` | `3.4924596548080444e-09` | `3.730150016028549e-10` | `1.5543987378098531e-06` | `0.9999999999985146` | `0.002` |
| posterior_sample | `PASS` | `4.696846008300781e-05` | `2.1222129160176137e-06` | `3.558654526702314e-07` | `0.9999999999997261` | `0.002` |
| scaled_latent | `PASS` | `8.553266525268555e-06` | `3.8622986014363255e-07` | `3.555605196281559e-07` | `0.9999999999997254` | `0.002` |
| decode_input | `PASS` | `4.696846008300781e-05` | `2.1162725008455e-06` | `3.548693288300777e-07` | `0.9999999999997277` | `0.002` |
| full_decoded_output | `PASS` | `8.344650268554688e-07` | `2.2265578957524634e-07` | `4.2691311649296113e-07` | `0.999999999999985` | `0.002` |
| full_final_clamped_output | `PASS` | `7.748603820800781e-07` | `2.083022115618439e-07` | `3.998399713265184e-07` | `0.9999999999999779` | `0.002` |
| decoder_original_forward_output | `PASS` | `3.5762786865234375e-07` | `6.110969008188005e-08` | `1.1716977263669046e-07` | `0.9999999999999908` | `0.002` |
| decoder_original_forward_final | `PASS` | `3.5762786865234375e-07` | `5.798372310689754e-08` | `1.1130083550545222e-07` | `0.9999999999999917` | `0.002` |

## Policy

```json
{
  "mirror_official_actual_behavior": true,
  "encoder_hook": {
    "encoder_hook_installed": true,
    "encoder_time_vae": true,
    "encoder_tiled_capable": true,
    "jittor_exact_tiled_task_queue_implemented": true,
    "tile_size": 16,
    "pad": 32,
    "note": "The Jittor alignment helper mirrors official encoder VAEHook.vae_tile_forward task queue, split/crop/write semantics, and cross-tile GroupNorm aggregation."
  },
  "decoder_hook": {
    "decoder_hook_installed": true,
    "decoder_time_vae": false,
    "decoder_original_forward_used": true,
    "decoder_tiled_path_triggered": false,
    "decoder_tiled_path_status": "NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT"
  },
  "do_not_force_decoder_time_vae_true": true,
  "do_not_claim_corrected_tiled_decoder": true,
  "scheduler_executed": false,
  "full_tadsr_inference_executed": false
}
```

## Remaining gaps
- Keep decoder hook as official original_forward unless the official contract changes or a separate corrected-tiled-decoder experiment is explicitly requested.
- Keep TIME_VAE_FULL_ALIGNMENT and JITTOR_FULL_INFERENCE NOT_COMPLETE until scheduler integration, runtime LoRA policy, and end-to-end inference are explicitly implemented and verified.
