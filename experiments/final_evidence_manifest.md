# 中文阅读说明：final_evidence_manifest.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Final Evidence Manifest

Status: **PASS**

This manifest indexes committed metadata, reports, and audit files. It does not execute model code and does not import torch.

## Major Status

| Marker | Status |
|---|---|
| `TADSR_UNET_FULL_FORWARD_ALIGNMENT` | `PASS` |
| `TIME_VAE_ACTUAL_VAEHOOK_FULL_BOUNDARY_ALIGNMENT` | `PASS` |
| `TADSR_STATIC_EFFECTIVE_LORA_COVERAGE_AUDIT` | `PASS` |
| `TADSR_SCHEDULER_BOUNDARY_ALIGNMENT` | `PASS` |
| `TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN` | `PASS` |
| `JITTOR_FULL_INFERENCE` | `NOT_COMPLETE` |
| `JITTOR_FULL_PORT` | `PARTIAL` |
| `TIME_VAE_FULL_ALIGNMENT` | `NOT_COMPLETE` |

## UNet evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| UNet entry alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_entry_alignment.json` |  |
| UNet down_blocks alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1_alignment.json` |  |
| UNet mid_block alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_resnet0_attention0_resnet1_alignment.json` |  |
| UNet up_blocks alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_entry_downblocks_midblock_upblocks_output_tail_alignment.json` |  |
| UNet output tail alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_output_tail_alignment.json` |  |
| UNet manual wrapper alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_manual_wrapper_alignment.json` |  |
| UNet official full forward alignment | `PASS` | `True` | `experiments/full_repro/unet_alignment/jittor_unet_full_forward_alignment.json` |  |

## TimeVAE evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| Deterministic decoder alignment | `PASS` | `True` | `experiments/full_repro/time_vae_alignment/jittor_decoder_entry_midblock_upblocks0123_tail_alignment.json` |  |
| TimeVAE full boundary alignment | `PASS` | `True` | `experiments/full_repro/vae_alignment/jittor_timevae_full_boundary_alignment.json` |  |
| Actual VAEHook behavior audit | `PASS` | `True` | `experiments/full_repro/vae_alignment/audit_tadsr_vae_actual_hook_behavior.json` |  |
| Actual VAEHook oracle | `PASS` | `True` | `experiments/full_repro/vae_alignment/oracle_tensors_vae_actual_hook_behavior/vae_actual_hook_behavior_oracle_metadata.json` |  |
| Actual VAEHook alignment | `PASS` | `True` | `experiments/full_repro/vae_alignment/jittor_timevae_actual_hook_behavior_alignment.json` |  |
| Tiled decoder blocked reason | `PASS` | `True` | `experiments/full_repro/vae_alignment/audit_tadsr_vae_tiled_boundary.json` | records the truthful decoder-hook limitation |
| TIME_VAE_FULL_ALIGNMENT remains NOT_COMPLETE | `NOT_COMPLETE` | `True` | `experiments/final_audit_report.json` | full TimeVAE status is intentionally not promoted |

## LoRA evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| Official LoRA policy audit | `PASS` | `True` | `experiments/full_repro/lora_alignment/audit_tadsr_lora_policy.json` |  |
| Static effective LoRA coverage | `PASS` | `True` | `experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json` |  |
| UNet active LoRA coverage | `PASS` | `True` | `experiments/full_repro/lora_alignment/jittor_effective_lora_coverage.json` |  |
| TimeVAE active LoRA coverage | `PASS` | `True` | `experiments/full_repro/lora_alignment/timevae_lora_effective_weight_coverage_test.json` |  |
| Dynamic runtime LoRA remains design-not-implemented | `NOT_IMPLEMENTED_BY_DESIGN` | `True` | `experiments/final_audit_report.json` |  |

## Scheduler evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| Scheduler boundary audit | `PASS` | `True` | `experiments/full_repro/scheduler_alignment/audit_tadsr_scheduler_boundary.json` |  |
| Scheduler oracle | `PASS` | `True` | `experiments/full_repro/scheduler_alignment/oracle_tensors_scheduler_boundary/scheduler_boundary_oracle_metadata.json` |  |
| Jittor scheduler boundary alignment | `PASS` | `True` | `experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json` |  |
| UNet scheduler one-step alignment | `PASS` | `True` | `experiments/full_repro/scheduler_alignment/jittor_scheduler_boundary_alignment.json` |  |

## Minimal integration evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| Minimal integration source audit | `PASS` | `True` | `experiments/full_repro/integration_alignment/audit_tadsr_minimal_integration.json` |  |
| Minimal latent oracle | `PASS` | `True` | `experiments/full_repro/integration_alignment/oracle_tensors_minimal_latent_integration/minimal_latent_integration_oracle_metadata.json` |  |
| get_x0_from_res alignment | `PASS` | `True` | `experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json` |  |
| Minimal decode/clamp tensor boundary alignment | `PASS` | `True` | `experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json` |  |
| Minimal one-step decode dry-run | `PASS` | `True` | `experiments/full_repro/integration_alignment/jittor_minimal_latent_integration_alignment.json` |  |

## Safety evidence

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| Final audit report | `MISSING` | `True` | `experiments/final_audit_report.json` |  |
| Migration report | `PRESENT` | `True` | `experiments/full_repro/jittor_alignment/jittor_migration_report.md` |  |
| Full inference guard remains NotImplemented | `NOT_COMPLETE` | `True` | `jittor_tadsr_full/tadsr_full.py` |  |
| Large oracle tensors ignored by git | `PRESENT` | `True` | `.gitignore` |  |

## Remaining gaps

| Evidence | Status | Exists | Path | Note |
|---|---|---:|---|---|
| JITTOR_FULL_INFERENCE | `NOT_COMPLETE` | `True` | `experiments/final_audit_report.json` | must remain NOT_COMPLETE |
| JITTOR_FULL_PORT | `PARTIAL` | `True` | `experiments/final_audit_report.json` | must remain PARTIAL |
| TIME_VAE_FULL_ALIGNMENT | `NOT_COMPLETE` | `True` | `experiments/final_audit_report.json` | must remain NOT_COMPLETE |
| Generic runtime LoRA | `NOT_IMPLEMENTED_BY_DESIGN` | `True` | `experiments/final_audit_report.json` | not implemented by design |
| Image/video generation | `NOT_IMPLEMENTED` | `True` | `docs/production_cli_design_audit.md` |  |
| Full denoising loop | `NOT_IMPLEMENTED` | `True` | `docs/production_cli_design_audit.md` |  |

## Missing Evidence Files

None.
