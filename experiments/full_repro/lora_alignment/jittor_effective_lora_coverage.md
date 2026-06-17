# 中文阅读说明：jittor_effective_lora_coverage.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Jittor Static Effective LoRA Coverage Audit

Overall static effective LoRA coverage: **PASS**

## Summary

- Active official LoRA modules: 290
- Covered modules: 290
- Partial modules: 0
- Missing modules: 0
- Effective-weight artifacts: 47
- Large NPZ artifacts tracked by git: 1

## Marker statuses

| Marker | Status |
|---|---|
| `static_effective_lora_coverage_audit` | `PASS` |
| `active_lora_module_coverage` | `PASS` |
| `effective_weight_artifact_coverage` | `PASS` |
| `dynamic_runtime_lora_implementation` | `NOT_IMPLEMENTED_BY_DESIGN` |
| `timevae_lora_effective_artifact_coverage` | `PASS` |
| `timevae_active_lora_module_coverage` | `PASS` |

## Component coverage

| Component | Total | PASS | PARTIAL | Missing |
|---|---:|---:|---:|---:|
| UNet | 258 | 258 | 0 | 0 |
| TimeVAE | 32 | 32 | 0 | 0 |

## Missing or partial active LoRA modules

| Component | Module | Status | Evidence |
|---|---|---|---|
| - | - | - | No missing or partial active LoRA modules. |

## Effective-weight artifacts

| Path | Component | Size bytes | Git ignored | Git tracked |
|---|---|---:|---|---|
| `experiments/full_repro/lora_alignment/timevae_lora_effective_weights/converted_timevae_lora_effective_weights.npz` | TimeVAE | 380694872 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz` | UNet | 10058807 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz` | UNet | 10061118 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz` | UNet | 3430372 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz` | UNet | 8386933 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz` | UNet | 8392496 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz` | UNet | 35287274 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz` | UNet | 35292588 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz` | UNet | 13711093 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz` | UNet | 24398733 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz` | UNet | 30518256 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz` | UNet | 131290548 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz` | UNet | 131275620 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz` | UNet | 54658431 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz` | UNet | 91327179 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz` | UNet | 115587438 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz` | UNet | 115917434 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz` | UNet | 115891162 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_entry_effective_weights.npz` | UNet | 7670557 | False | True |
| `experiments/full_repro/unet_alignment/converted_unet_midblock_attention0_effective_weights.npz` | UNet | 131511175 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_midblock_resnet0_effective_weights.npz` | UNet | 115909889 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_midblock_resnet1_effective_weights.npz` | UNet | 115807829 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_output_tail_effective_weights.npz` | UNet | 92762 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet0_effective_weights.npz` | UNet | 182865053 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet1_effective_weights.npz` | UNet | 183094288 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet2_effective_weights.npz` | UNet | 183228954 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock0_upsampler_effective_weights.npz` | UNet | 54787520 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_attention0_effective_weights.npz` | UNet | 131268821 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_attention1_effective_weights.npz` | UNet | 131272397 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_attention2_effective_weights.npz` | UNet | 131243257 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet0_effective_weights.npz` | UNet | 182817304 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet1_effective_weights.npz` | UNet | 182608625 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet2_effective_weights.npz` | UNet | 152144607 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock1_upsampler_effective_weights.npz` | UNet | 54800138 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_attention0_effective_weights.npz` | UNet | 35296731 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_attention1_effective_weights.npz` | UNet | 35294371 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_attention2_effective_weights.npz` | UNet | 35292890 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet0_effective_weights.npz` | UNet | 62553308 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet1_effective_weights.npz` | UNet | 47319354 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet2_effective_weights.npz` | UNet | 39696152 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock2_upsampler_effective_weights.npz` | UNet | 13709912 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_attention0_effective_weights.npz` | UNet | 10060589 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_attention1_effective_weights.npz` | UNet | 10061166 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_attention2_effective_weights.npz` | UNet | 10063413 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet0_effective_weights.npz` | UNet | 16434694 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet1_effective_weights.npz` | UNet | 12605142 | True | False |
| `experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet2_effective_weights.npz` | UNet | 12593738 | True | False |
