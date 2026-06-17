# 中文阅读说明：unet_migration_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TADSR UNet Entry Migration Report

This report covers only the UNet input-contract / entry stage. It does not claim full UNet forward or full TADSR inference.

## Official input contract audit

| Field | Value |
|---|---|
| Official class | `models.unet_2d_condition.UNet2DConditionModel` |
| Input latent sample | `[B, 4, H, W]` |
| Synthetic oracle shape | `[1, 4, 32, 32]` |
| Official sample_size | `64` |
| Timestep used | `[1]` |
| Cross-attention dim | `1024` |
| center_input_sample | `False` |
| time_proj | `diffusers.models.embeddings.Timesteps` with attrs `{'num_channels': 320, 'flip_sin_to_cos': True, 'downscale_freq_shift': 0, 'scale': None, 'max_period': None}` |
| time_embedding | `diffusers.models.embeddings.TimestepEmbedding` |

## LoRA handling in this stage

| Item | Value |
|---|---|
| UNet LoRA target counts | `{'encoder': 94, 'decoder': 148, 'others': 16}` |
| Loaded LoRA params | `516` |
| Active conv_in adapters | `['default_encoder', 'default_decoder', 'default_others']` |
| Effective conv_in delta records | `[{'adapter': 'default_encoder', 'shape': [320, 4, 3, 3], 'max_abs': 0.043385569006204605}]` |
| Runtime LoRA integration | `DEFERRED; Jittor consumes static effective entry weights for this stage` |

## Alignment metrics

| Stage | Max abs error | Mean abs error | Tolerance |
|---|---:|---:|---:|
| `center_input_sample` | `0.0` | `0.0` | `0.0` |
| `conv_in` | `8.344650268554688e-07` | `1.2484131417522804e-08` | `0.0001` |
| `time_proj` | `1.1920928955078125e-07` | `4.547564458334819e-09` | `1e-05` |
| `time_embedding.linear_1` | `9.5367431640625e-07` | `2.1003188521717675e-08` | `0.0001` |
| `time_embedding.act` | `4.76837158203125e-07` | `5.591943086358242e-09` | `0.0001` |
| `time_embedding.full` | `4.76837158203125e-07` | `4.34172306995606e-09` | `0.001` |

## Status summary

| Check | Status |
|---|---|
| `TADSR_UNET_OVERVIEW_AUDIT` | `PASS` |
| `TADSR_UNET_ENTRY_AUDIT` | `PASS` |
| `TADSR_UNET_ENTRY_ORACLE_TENSORS` | `PASS` |
| `TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS` | `PASS` |
| `TADSR_UNET_CENTER_INPUT_ALIGNMENT` | `PASS` |
| `TADSR_UNET_CONV_IN_ALIGNMENT` | `PASS` |
| `TADSR_UNET_TIME_PROJ_ALIGNMENT` | `PASS` |
| `TADSR_UNET_TIME_EMBED_LINEAR1_ALIGNMENT` | `PASS` |
| `TADSR_UNET_TIME_EMBED_ACT_ALIGNMENT` | `PASS` |
| `TADSR_UNET_TIME_EMBED_ALIGNMENT` | `PASS` |
| `TADSR_UNET_ENTRY_ALIGNMENT` | `PASS` |
| `TADSR_UNET_ENTRY_VAE_MODE_ALIGNMENT` | `NOT_APPLICABLE` |

## Explicitly not completed

- Full UNet forward.
- `down_blocks.0` or later UNet blocks.
- Cross-attention.
- Generic runtime LoRA integration.
- Full TADSR inference.

NEXT_ACTION: Continue TADSR UNet `down_blocks.0` audit/export/port after UNet entry and timestep embedding alignment; keep full inference NotImplemented until full UNet and LoRA runtime integration pass.

## TADSR UNet up_blocks.1.attentions.1 alignment update

- `up_blocks.1.attentions.1` has now been audited against the official PyTorch TADSR UNet.
- PyTorch oracle tensors and static effective weights were exported under `experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_attention1/` and `converted_unet_upblock1_attention1_effective_weights.npz`.
- Jittor validates the attention module progressively:
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_NORM_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_SEQUENCE_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_PROJ_IN_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_QKV_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN1_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_QKV_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_ATTN2_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_FF_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_TRANSFORMER0_ALIGNMENT: PASS`
  - `TADSR_UNET_UPBLOCK1_ATTENTION1_ALIGNMENT: PASS`
- The verified bridge path is `UNet entry -> all down_blocks -> local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1`.
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_PRE_RESNET2_ALIGNMENT: PASS`
- Full `up_blocks.1`, full UNet forward, generic runtime LoRA integration, and full TADSR inference remain intentionally incomplete: `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`, `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`, `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`, `JITTOR_FULL_INFERENCE: NOT_COMPLETE`.
- Next target: `up_blocks.1.resnets.2`.


## TADSR UNet down_blocks.0.attentions.0 Migration Update

- `down_blocks.0.attentions.0` has a dedicated official audit, oracle export, effective-weight export, and Jittor alignment tests.
- The aligned scope is limited to the local Transformer2DModel attention module and the bridge `UNet entry -> down_blocks.0.resnets.0 -> down_blocks.0.attentions.0`.
- This stage uses exported effective LoRA-merged weights for proj_in/proj_out, self-attention, cross-attention, and feed-forward projections.
- Full `down_blocks.0` remains incomplete because `resnets.1`, `attentions.1`, and the downsampler remain unopened.
- Full UNet forward and full TADSR inference remain intentionally `NotImplemented`.

## TADSR UNet Down Blocks 0 ResNet 1 Alignment

- `down_blocks.0.resnets.1` has been audited against the official PyTorch TADSR UNet and aligned in Jittor using exported effective weights.
- The verified bridge path is `UNet entry -> down_blocks.0.resnets.0 -> down_blocks.0.attentions.0 -> down_blocks.0.resnets.1`.
- This remains a partial UNet migration. `down_blocks.0.attentions.1`, `down_blocks.0.downsampler`, full `down_blocks.0`, full UNet forward, generic runtime LoRA and full TADSR inference are still not claimed.

## TADSR UNet Down Blocks 0 Attention 1 Alignment

- `down_blocks.0.attentions.1` has been audited against the official PyTorch TADSR UNet and aligned in Jittor using exported effective weights.
- The verified bridge path is `UNet entry -> down_blocks.0.resnets.0 -> down_blocks.0.attentions.0 -> down_blocks.0.resnets.1 -> down_blocks.0.attentions.1`.
- The down_blocks.0 pre-downsampler path is now aligned, but full `down_blocks.0` remains incomplete because `down_blocks.0.downsamplers.0` is still unopened.
- Full UNet forward, generic runtime LoRA and full TADSR inference remain not claimed.

## TADSR UNet Down Blocks 0 Downsampler and Full Local Down Blocks 0 Alignment

- `down_blocks.0.downsamplers.0` has been audited against the official PyTorch TADSR UNet and aligned in Jittor using exported effective LoRA-merged weights.
- The verified full local path is `UNet entry -> down_blocks.0.resnets.0 -> down_blocks.0.attentions.0 -> down_blocks.0.resnets.1 -> down_blocks.0.attentions.1 -> down_blocks.0.downsamplers.0`.
- This completes local `down_blocks.0` alignment only. Later UNet blocks, full UNet forward, generic runtime LoRA and full TADSR inference remain not claimed.
- `TADSR_UNET_DOWNBLOCK0_DOWNSAMPLER_ALIGNMENT` and `TADSR_UNET_DOWNBLOCK0_ALIGNMENT` are expected to be PASS after running the downsampler oracle export and alignment tests.

## TADSR UNet down_blocks.1.resnets.0 local alignment

This repository now includes a local, verified Jittor bridge for `down_blocks.1.resnets.0`: `entry -> down_blocks.0 -> down_blocks.1.resnets.0`. The scope is intentionally narrow. `down_blocks.1.attentions.0`, `down_blocks.1.resnets.1`, `down_blocks.1.attentions.1`, `down_blocks.1.downsamplers.0`, `mid_block`, `up_blocks`, full UNet forward, generic runtime LoRA, and full TADSR inference remain incomplete.

Expected audit status after this stage:

- `TADSR_UNET_DOWNBLOCK1_RESNET0_BRIDGE_ALIGNMENT: PASS`
- Historical note: this earlier stage left `TADSR_UNET_DOWNBLOCK1_ALIGNMENT` incomplete; current status is `TADSR_UNET_DOWNBLOCK1_ALIGNMENT: PASS` after `down_blocks.1.downsamplers.0` alignment.
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

## TADSR UNet down_blocks.1.attentions.0 local alignment

This repository now includes a local, verified Jittor bridge for `down_blocks.1.attentions.0`: `entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> down_blocks.1.attentions.0`. This stage uses exported effective attention weights and does not implement generic runtime LoRA.

The scope is intentionally narrow. `down_blocks.1.resnets.1`, `down_blocks.1.attentions.1`, `down_blocks.1.downsamplers.0`, `mid_block`, `up_blocks`, full UNet forward, generic runtime LoRA, and full TADSR inference remain incomplete.

Expected audit status after this stage:

- `TADSR_UNET_DOWNBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT: PASS`
- Historical note: this earlier stage left `TADSR_UNET_DOWNBLOCK1_ALIGNMENT` incomplete; current status is `TADSR_UNET_DOWNBLOCK1_ALIGNMENT: PASS` after `down_blocks.1.downsamplers.0` alignment.
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

## TADSR UNet down_blocks.1.resnets.1 local alignment

This repository now includes a local, verified Jittor bridge for `down_blocks.1.resnets.1`: `entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> down_blocks.1.attentions.0 -> down_blocks.1.resnets.1`. This stage reuses the generalized ResNet tester with exported effective static weights.

The scope is intentionally narrow. `down_blocks.1.attentions.1`, `down_blocks.1.downsamplers.0`, `mid_block`, `up_blocks`, full UNet forward, generic runtime LoRA, and full TADSR inference remain incomplete.

Expected audit status after this stage:

- `TADSR_UNET_DOWNBLOCK1_RESNET1_BRIDGE_ALIGNMENT: PASS`
- Historical note: this earlier stage left `TADSR_UNET_DOWNBLOCK1_ALIGNMENT` incomplete; current status is `TADSR_UNET_DOWNBLOCK1_ALIGNMENT: PASS` after `down_blocks.1.downsamplers.0` alignment.
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

### TADSR UNet down_blocks.1.attentions.1 migration note
- Added official audit/export and Jittor numerical alignment tests for `down_blocks.1.attentions.1` only.
- The local bridge now covers entry -> down_blocks.0 -> down_blocks.1.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1.
- Historical note: at the attention1-only stage, `down_blocks.1.downsamplers.0` was not ported. Current status is superseded by the downsampler stage: `TADSR_UNET_DOWNBLOCK1_ALIGNMENT: PASS`.
- Full UNet forward, full TADSR inference and generic LoRA runtime remain incomplete; this stage uses exported effective weights.

## TADSR UNet down_blocks.1.downsamplers.0 and full local down_blocks.1 alignment

This stage completes the local `down_blocks.1` chain only. The official PyTorch module was audited first, then oracle tensors and LoRA-merged effective weights were exported, and finally the Jittor/Numpy tester was compared against the official tensors.

Verified local order:

```text
entry -> down_blocks.0
      -> down_blocks.1.resnets.0
      -> down_blocks.1.attentions.0
      -> down_blocks.1.resnets.1
      -> down_blocks.1.attentions.1
      -> down_blocks.1.downsamplers.0
```

Important official behavior discovered in the audit:

- `down_blocks.1.downsamplers.0` is a single effective LoRA-merged stride-2 convolution with padding 1.
- `down_blocks.1` returns three local residual samples in this order: attention0 output, attention1 output, downsampler output.
- The manual official chain and official `block.forward` output have max absolute error `0.0`.

Status:

| Check | Status |
|---|---|
| `TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_AUDIT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_LORA_AUDIT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_LOCAL_FORWARD_AUDIT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_AUDIT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ORACLE_TENSORS` | PASS |
| `TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_EFFECTIVE_WEIGHTS` | PASS |
| `TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ORACLE_TENSORS` | PASS |
| `TADSR_UNET_DOWNBLOCK1_DOWNSAMPLER_ALIGNMENT` | PASS |
| `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_OUTPUT_HIDDEN_ALIGNMENT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_OUTPUT_STATES_ALIGNMENT` | PASS |
| `TADSR_UNET_DOWNBLOCK1_ALIGNMENT` | PASS |

Still not claimed:

- `TADSR_UNET_DOWNBLOCK2_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next action: migrate `down_blocks.2.resnets.0` using the same audit -> oracle export -> local Jittor alignment workflow.

<!-- downblock2-resnet0-update -->

## Latest migration update: UNet down_blocks.2.resnets.0

This stage ports and verifies only the first leaf module of `down_blocks.2`: `down_blocks.2.resnets.0`. The verified local path is:

`UNet entry -> full local down_blocks.0 -> full local down_blocks.1 -> down_blocks.2.resnets.0`

Evidence generated in this stage:

- Official PyTorch audit: `TADSR_UNET_DOWNBLOCK2_AUDIT: PASS`, `TADSR_UNET_DOWNBLOCK2_RESNET0_AUDIT: PASS`, `TADSR_UNET_DOWNBLOCK2_RESNET0_LORA_AUDIT: PASS`.
- Oracle/effective-weight export: `TADSR_UNET_DOWNBLOCK2_RESNET0_ORACLE_TENSORS: PASS`, `TADSR_UNET_DOWNBLOCK2_RESNET0_EFFECTIVE_WEIGHTS: PASS`.
- Jittor ResNet leaf alignment: norm1, conv1, time embedding projection, conv2, shortcut and output all PASS. Final output max_abs_error is `1.621246337890625e-05` under tolerance `1e-4`.
- Synthetic isolated check: `TADSR_UNET_DOWNBLOCK2_RESNET0_SYNTHETIC_ALIGNMENT: PASS`.
- Bridge check: `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ALIGNMENT: PASS`.
- Full alignment suite still passes after adding these tests.

Scope boundaries remain strict:

- `TADSR_UNET_DOWNBLOCK2_ALIGNMENT: NOT_COMPLETE` because attention0/resnet1/attention1/downsampler are not yet ported.
- `TADSR_UNET_DOWNBLOCK2_ATTENTION0_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`.
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`.

Next migration target: `down_blocks.2.resnets.1` audit/export/port, still without opening full UNet forward or full TADSR inference.

<!-- downblock2-attention0-update -->

## Latest migration update: UNet down_blocks.2.attentions.0

This stage ports and verifies only `down_blocks.2.attentions.0`. The verified local bridge is:

`UNet entry -> full local down_blocks.0 -> full local down_blocks.1 -> down_blocks.2.resnets.0 -> down_blocks.2.attentions.0`

Evidence generated in this stage:

- Official PyTorch audit: `TADSR_UNET_DOWNBLOCK2_ATTENTION0_AUDIT: PASS`, `TADSR_UNET_DOWNBLOCK2_ATTENTION0_TRANSFORMER_AUDIT: PASS`, `TADSR_UNET_DOWNBLOCK2_ATTENTION0_LORA_AUDIT: PASS`.
- Oracle/effective-weight export: `TADSR_UNET_DOWNBLOCK2_ATTENTION0_ORACLE_TENSORS: PASS`, `TADSR_UNET_DOWNBLOCK2_ATTENTION0_EFFECTIVE_WEIGHTS: PASS`.
- Projection path: `TADSR_UNET_DOWNBLOCK2_ATTENTION0_NORM_ALIGNMENT: PASS`, `TADSR_UNET_DOWNBLOCK2_ATTENTION0_SEQUENCE_ALIGNMENT: PASS`, `TADSR_UNET_DOWNBLOCK2_ATTENTION0_PROJ_IN_ALIGNMENT: PASS`.
- Transformer block: attn1 Q/K/V, attn1 output/residual, attn2 Q/K/V, attn2 output/residual, feed-forward and transformer0 output all PASS.
- Full local attention leaf: `TADSR_UNET_DOWNBLOCK2_ATTENTION0_ALIGNMENT: PASS`.
- Bridge path: `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_ALIGNMENT: PASS`.
- Final audit report records `TADSR_UNET_DOWNBLOCK2_ATTENTION0_BRIDGE_ALIGNMENT: PASS` and `TADSR_UNET_DOWNBLOCK2_PRE_RESNET1_ALIGNMENT: PASS`.

Scope boundaries remain strict:

- `TADSR_UNET_DOWNBLOCK2_ALIGNMENT: NOT_COMPLETE` because `resnets.1`, `attentions.1`, and `downsamplers.0` are not yet ported.
- `TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_CROSS_ATTENTION_ALIGNMENT: NOT_COMPLETE`; this is a local attention leaf alignment, not global UNet cross-attention completion.
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`.
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`; this stage uses exported effective static weights, not a generic runtime LoRA implementation.
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`.

Next migration target: `down_blocks.2.resnets.1` audit/export/port, still without opening full UNet forward or full TADSR inference.

<!-- downblock2-resnet1-update -->
### TADSR UNet down_blocks.2.resnets.1 alignment update

- `down_blocks.2.resnets.1` has now been audited against the official PyTorch TADSR UNet, exported as deterministic oracle tensors, implemented through the existing generalized Jittor `UNetResnetBlock2DTester`, and numerically aligned.
- The validated bridge path is `UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> down_blocks.2.attentions.0 -> down_blocks.2.resnets.1`.
- New PASS markers include `TADSR_UNET_DOWNBLOCK2_RESNET1_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_RESNET1_SYNTHETIC_ALIGNMENT`, `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_RESNET1_BRIDGE_ALIGNMENT`, and `TADSR_UNET_DOWNBLOCK2_PRE_ATTENTION1_ALIGNMENT`.
- This is still only a partial `down_blocks.2` migration. `down_blocks.2.attentions.1` and `down_blocks.2.downsamplers.0` are not ported yet, so `TADSR_UNET_DOWNBLOCK2_ALIGNMENT` remains `NOT_COMPLETE`.
- Full UNet forward and full TADSR inference remain intentionally `NOT_COMPLETE`; generic runtime LoRA integration is still `PARTIAL` because this migration uses exported effective weights.
- Next action: audit/export/port `down_blocks.3.resnets.0` only.

<!-- downblock2-attention1-update -->
### TADSR UNet down_blocks.2.attentions.1 alignment update

- `down_blocks.2.attentions.1` has now been audited against the official PyTorch TADSR UNet, exported as deterministic oracle tensors, implemented through the existing generalized Jittor Transformer2D attention tester, and numerically aligned.
- The validated bridge path is `UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> down_blocks.2.attentions.0 -> down_blocks.2.resnets.1 -> down_blocks.2.attentions.1`.
- New PASS markers include `TADSR_UNET_DOWNBLOCK2_ATTENTION1_ALIGNMENT`, `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_ATTENTION1_BRIDGE_ALIGNMENT`, and `TADSR_UNET_DOWNBLOCK2_PRE_DOWNSAMPLER_ALIGNMENT`.
- This is still only a partial `down_blocks.2` migration. `down_blocks.2.downsamplers.0` is not ported yet, so `TADSR_UNET_DOWNBLOCK2_ALIGNMENT` remains `NOT_COMPLETE`.
- Full UNet forward, full TADSR inference, and generic runtime LoRA remain intentionally incomplete: `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`, `JITTOR_FULL_INFERENCE: NOT_COMPLETE`, `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`.
- Next action: audit/export/port `down_blocks.3.resnets.0` only.

<!-- downblock2-downsampler-update -->
### TADSR UNet down_blocks.2.downsamplers.0 and full local down_blocks.2 alignment update

- `down_blocks.2.downsamplers.0` has now been audited against the official PyTorch TADSR UNet, exported as deterministic oracle tensors, implemented with exported effective static weights, and numerically aligned in Jittor.
- The validated bridge path is `UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2.resnets.0 -> down_blocks.2.attentions.0 -> down_blocks.2.resnets.1 -> down_blocks.2.attentions.1 -> down_blocks.2.downsamplers.0`.
- The full local `down_blocks.2` residual tuple was verified: output hidden states and `output_states`/`down_block_res_samples` order match the official block-level behavior.
- New PASS markers include `TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ALIGNMENT`, `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_DOWNSAMPLER_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_OUTPUT_HIDDEN_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK2_BRIDGE_ALIGNMENT`, and `TADSR_UNET_DOWNBLOCK2_ALIGNMENT`.
- This still does not open `down_blocks.3`, `mid_block`, `up_blocks`, full UNet forward, full TADSR inference, or generic runtime LoRA.
- Next action: audit/export/port `down_blocks.3.resnets.0` only.


### TADSR UNet down_blocks.3.resnets.0 update

- `down_blocks.3.resnets.0` has now been audited against the official PyTorch TADSR UNet, exported as deterministic oracle tensors, implemented with exported effective static weights, and numerically aligned in Jittor.
- The validated bridge path is `UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3.resnets.0`.
- This is only the first leaf module of `down_blocks.3`; full `down_blocks.3`, `mid_block`, `up_blocks`, full UNet forward, full TADSR inference, and generic runtime LoRA remain unopened.
- New PASS markers include `TADSR_UNET_DOWNBLOCK3_RESNET0_ALIGNMENT`, `TADSR_UNET_DOWNBLOCK3_RESNET0_SYNTHETIC_ALIGNMENT`, `TADSR_UNET_ENTRY_DOWNBLOCK0_DOWNBLOCK1_DOWNBLOCK2_DOWNBLOCK3_RESNET0_ALIGNMENT`, and `TADSR_UNET_DOWNBLOCK3_RESNET0_BRIDGE_ALIGNMENT`.
- Next action: audit/export/port `down_blocks.3.resnets.1` only.


### Current UNet mid_block.resnets.0 status

- `TADSR_UNET_MIDBLOCK_RESNET0_ALIGNMENT: PASS` after this stage when oracle/tests are regenerated.
- `TADSR_UNET_MIDBLOCK_RESNET0_BRIDGE_ALIGNMENT: PASS` covers only `UNet entry -> all local down_blocks -> mid_block.resnets.0`.
- `TADSR_UNET_MIDBLOCK_ALIGNMENT`, `TADSR_UNET_MIDBLOCK_ATTENTION0_ALIGNMENT`, `TADSR_UNET_MIDBLOCK_RESNET1_ALIGNMENT`, `TADSR_UNET_UPBLOCKS_ALIGNMENT`, `TADSR_UNET_FULL_FORWARD_ALIGNMENT`, and `JITTOR_FULL_INFERENCE` remain `NOT_COMPLETE`.
- Generic runtime LoRA remains partial; this stage uses exported effective weights.

## TADSR UNet mid_block.attentions.0 migration update

- TADSR UNet `mid_block.attentions.0` has been audited, exported, and numerically aligned in Jittor.
- The bridge path `UNet entry -> down_blocks.0/1/2/3 -> mid_block.resnets.0 -> mid_block.attentions.0` is aligned.
- This remains a partial mid_block migration: `mid_block.resnets.1`, up_blocks, full UNet forward, generic LoRA runtime, and full TADSR inference remain NOT_COMPLETE/PARTIAL.
- This stage uses exported effective static weights for LoRA-bearing modules; no generic PEFT/LoRA runtime was introduced.


### TADSR UNet mid_block.resnets.1 local alignment

- TADSR UNet mid_block.resnets.1 has been numerically aligned in Jittor.
- The local mid_block hidden-state chain `resnets.0 -> attentions.0 -> resnets.1` is verified against official local mid_block forward.
- `TADSR_UNET_MIDBLOCK_ALIGNMENT` may be PASS after this stage, but this is still not full UNet forward.
- `up_blocks`, generic runtime LoRA integration, and full TADSR inference remain NOT_COMPLETE/PARTIAL as appropriate.

### TADSR UNet up_blocks.0.resnets.0 alignment

- TADSR UNet up_blocks.0.resnets.0 has been numerically aligned in Jittor.
- The bridge path `UNet entry -> all down_blocks -> local mid_block -> up_blocks.0.resnets.0` is aligned.
- The official residual skip consumption contract was audited: `up_blocks.0.resnets.0` consumes the last accumulated residual, concatenates on NCHW channel axis, and stops before later up-path modules.
- Full `up_blocks.0`, all later up blocks, full UNet forward, generic runtime LoRA, and full TADSR inference remain NOT_COMPLETE/PARTIAL as appropriate.

## TADSR UNet up_blocks.0.resnets.2 alignment update

This stage ports only `up_blocks.0.resnets.2` in the official TADSR UNet up path. The verified local bridge is:

`UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3 -> local mid_block -> up_blocks.0.resnets.0 -> up_blocks.0.resnets.1 -> up_blocks.0.resnets.2`

Official residual consumption was audited before the Jittor port. `up_blocks.0.resnets.2` consumes `down_blocks.2.output_state_2` from the accumulated residual tuple after `resnets.0` and `resnets.1` consume the two `down_blocks.3` residuals. The Jittor tester uses exported static effective weights and validates isolated, synthetic, and full local bridge outputs.

Status after this stage:

- `TADSR_UNET_UPBLOCK0_RESNET2_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK0_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK0_RESNET2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK0_RESNET2_SYNTHETIC_ALIGNMENT: PASS`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_RESNET0_RESNET1_RESNET2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK0_PRE_UPSAMPLER_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK0_UPSAMPLER_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK0_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.0.upsamplers.0`. Full UNet forward and full TADSR inference remain intentionally disabled.

## TADSR UNet up_blocks.2.attentions.2 alignment update

Scope of this stage:

`entry -> all down_blocks -> local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> up_blocks.2.resnets.0 -> up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 -> up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 -> up_blocks.2.attentions.2`

This is a local bridge ending at `up_blocks.2.attentions.2`. It does not enter
`up_blocks.2.upsamplers.0`, full `up_blocks.2`, `up_blocks.3`, full UNet
forward, or full TADSR inference.

| Evidence | Result |
|---|---|
| Official topology audit | `TADSR_UNET_UPBLOCK2_ATTENTION2_AUDIT: PASS` |
| Residual contract audit | `attention2_consumes_accumulated_residuals = False`; residual count `0 -> 0` |
| Official next module | `up_blocks.2.upsamplers.0` |
| Oracle tensors | `TADSR_UNET_UPBLOCK2_ATTENTION2_ORACLE_TENSORS: PASS` |
| Effective weights | `TADSR_UNET_UPBLOCK2_ATTENTION2_EFFECTIVE_WEIGHTS: PASS` |
| LoRA handling | 12 LoRA-wrapped leaves exported as static effective weights |
| Manual PyTorch reconstruction | `synthetic_max_abs_diff = 0.0`, `entry_max_abs_diff = 0.0` |
| Isolated attention2 alignment | max abs `0.00010395050048828125`, cosine `0.9999999999061807` |
| Full local bridge alignment | max abs `1.6927719116210938e-05`, cosine `0.9999999999979309` |

Submodule-level PASS markers:

- `TADSR_UNET_UPBLOCK2_ATTENTION2_NORM_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_PROJ_IN_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_SEQUENCE_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_QKV_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN1_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN1_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_QKV_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_ATTN2_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_AFTER_ATTN2_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_FF_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_TRANSFORMER0_ALIGNMENT`
- `TADSR_UNET_UPBLOCK2_ATTENTION2_ALIGNMENT`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT`

The final audit now reports `TADSR_UNET_UPBLOCK2_ATTENTION2_BRIDGE_ALIGNMENT:
PASS` and `TADSR_UNET_UPBLOCK2_PRE_UPSAMPLER_ALIGNMENT: PASS`, while keeping
`TADSR_UNET_UPBLOCK2_ALIGNMENT`, `TADSR_UNET_UPBLOCKS_ALIGNMENT`,
`TADSR_UNET_FULL_FORWARD_ALIGNMENT`, and `JITTOR_FULL_INFERENCE` as
`NOT_COMPLETE`.

## Latest update: UNet up_blocks.2.upsamplers.0 alignment

This stage migrates only `up_blocks.2.upsamplers.0`. The selected bridge now
runs:

`entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 ->
full local up_blocks.1 -> up_blocks.2.resnets.0 -> attentions.0 -> resnets.1
-> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0`.

The official audit and oracle export record the upsampler as nearest-neighbor
2x interpolation followed by an effective convolution. It consumes no
accumulated residual samples. The Jittor port adds
`TADSRUNetUpBlock2UpsamplerTester`, which deliberately stops before
`up_blocks.3`, full UNet forward, runtime LoRA, and full TADSR inference.

Evidence:

| Check | Status |
|---|---|
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_AUDIT` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_ORACLE_TENSORS` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_EFFECTIVE_WEIGHTS` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_INTERPOLATION_ALIGNMENT` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_CONV_ALIGNMENT` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_ALIGNMENT` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_SYNTHETIC_ALIGNMENT` | PASS |
| `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_THROUGH_UPSAMPLER_ALIGNMENT` | PASS |
| `TADSR_UNET_UPBLOCK2_UPSAMPLER_BRIDGE_ALIGNMENT` | PASS |

Boundary status remains strict:

- `TADSR_UNET_UPBLOCK2_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: full local `up_blocks.2` aggregate verification.

## TADSR UNet up_blocks.1.attentions.0 alignment update

This stage ports only `up_blocks.1.attentions.0` after the already aligned `up_blocks.1.resnets.0`. The verified bridge is:

`UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3 -> local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0`

The official audit confirms that `up_blocks.1.attentions.0` is a Transformer2D attention block with one BasicTransformerBlock. The Jittor tester reuses the generic `UNetAttention0Transformer2DTester` with prefix `up_blocks_1_attentions_0` and exported static effective weights. The alignment checks cover group norm, sequence reshape, `proj_in`, self-attention Q/K/V and output, cross-attention Q/K/V and output, residual states after attention, feed-forward, transformer output, local attention output, and the end-to-end bridge through all previous aligned UNet segments.

Status after this stage:

- `TADSR_UNET_UPBLOCK1_ATTENTION0_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_EFFECTIVE_WEIGHTS: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_NORM_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_SEQUENCE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_PROJ_IN_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_QKV_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_QKV_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ATTN2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_AFTER_ATTN2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_FF_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_TRANSFORMER0_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION0_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_PRE_RESNET1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.1.resnets.1`. Full `up_blocks.1`, full UNet forward, generic runtime LoRA, and full TADSR inference remain intentionally incomplete.

## TADSR UNet up_blocks.1.resnets.1 alignment update

This stage ports only `up_blocks.1.resnets.1` after the verified
`up_blocks.1.resnets.0 -> up_blocks.1.attentions.0` path. The verified bridge
is:

`UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3 -> local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1`

The official residual contract was audited before the Jittor port. Since
`up_blocks.1.attentions.0` does not consume residuals, `up_blocks.1.resnets.1`
consumes the next residual from the local tuple:
`down_blocks.2.output_state_0`, global accumulated residual index `7`, shape
`[1, 1280, 8, 8]`. The concat input shape is `[1, 2560, 8, 8]`, and the
`resnets.1` output shape is `[1, 1280, 8, 8]`.

Status after this stage:

- `TADSR_UNET_UPBLOCK1_RESNET1_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_EFFECTIVE_WEIGHTS: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_RESIDUAL_CONCAT_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_NORM1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_CONV1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_TEMB_PROJ_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_CONV2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_SHORTCUT_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_SYNTHETIC_ALIGNMENT: PASS`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET1_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_PRE_ATTENTION1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.1.attentions.1`. Full `up_blocks.1`, full UNet forward,
generic runtime LoRA, and full TADSR inference remain intentionally incomplete.

## TADSR UNet up_blocks.1.resnets.2 alignment update

This stage ports only `up_blocks.1.resnets.2` after the verified
`up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1
-> up_blocks.1.attentions.1` path. The verified bridge is:

`UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3 -> local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2`

The official residual contract was audited before the Jittor port. Since
`up_blocks.1.attentions.1` does not consume residuals, `up_blocks.1.resnets.2`
consumes the next residual from the local tuple:
`down_blocks.1.output_state_2`, global accumulated residual index `6`, shape
`[1, 640, 8, 8]`. The concat input shape is `[1, 1920, 8, 8]`, and the
`resnets.2` output shape is `[1, 1280, 8, 8]`.

Status after this stage:

- `TADSR_UNET_UPBLOCK1_RESNET2_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_EFFECTIVE_WEIGHTS: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_RESIDUAL_CONCAT_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_NORM1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_CONV1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_TEMB_PROJ_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_CONV2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_SHORTCUT_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_SYNTHETIC_ALIGNMENT: PASS`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_RESNET2_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_PRE_ATTENTION2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.1.attentions.2`. Full `up_blocks.1`, full UNet forward,
generic runtime LoRA, and full TADSR inference remain intentionally incomplete.

## TADSR UNet up_blocks.1.attentions.2 alignment update

This stage ports only `up_blocks.1.attentions.2` after the verified
`up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1
-> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2` path. The verified bridge is:

`UNet entry -> down_blocks.0 -> down_blocks.1 -> down_blocks.2 -> down_blocks.3 -> local mid_block -> full local up_blocks.0 -> up_blocks.1.resnets.0 -> up_blocks.1.attentions.0 -> up_blocks.1.resnets.1 -> up_blocks.1.attentions.1 -> up_blocks.1.resnets.2 -> up_blocks.1.attentions.2`

The official residual contract was audited before the Jittor port. Since
`up_blocks.1.attentions.2` is an attention module, it does not consume any
residual sample. It receives the `[1, 1280, 8, 8]` hidden state from
`up_blocks.1.resnets.2`, uses the exported `encoder_hidden_states` for
cross-attention, and preserves the remaining residual tuple for later
`up_blocks.2/3`. The next unopened official module is
`up_blocks.1.upsamplers.0`.

Status after this stage:

- `TADSR_UNET_UPBLOCK1_ATTENTION2_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_LORA_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_EFFECTIVE_WEIGHTS: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_NORM_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_PROJ_IN_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_SEQUENCE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_QKV_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_QKV_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ATTN2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_AFTER_ATTN2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_FF_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_TRANSFORMER0_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_ALIGNMENT: PASS`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_RESNET0_ATTENTION0_RESNET1_ATTENTION1_RESNET2_ATTENTION2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ATTENTION2_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_PRE_UPSAMPLER_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Key numerical evidence:

| Check | Status | Max abs error | Mean abs error | Cosine |
|---|---|---:|---:|---:|
| isolated `up_blocks.1.attentions.2` | PASS | `3.337860107421875e-05` | `6.172000837523228e-06` | `0.9999999999960631` |
| bridge through `up_blocks.1.attentions.2` | PASS | `2.5391578674316406e-05` | `2.540551804486313e-06` | `0.9999999999982148` |

Next target: `up_blocks.1.upsamplers.0`. Full `up_blocks.1`, full UNet forward,
generic runtime LoRA, and full TADSR inference remain intentionally incomplete.

## Latest TADSR UNet up_blocks.1.upsamplers.0 alignment update

This stage ports only `up_blocks.1.upsamplers.0`. It does not run
`up_blocks.2`, does not compare the full local `up_blocks.1` aggregate, and
does not open full UNet forward.

Official audit:

- module: `diffusers.models.upsampling.Upsample2D`
- operation: nearest-neighbor 2x interpolation followed by effective
  LoRA-wrapped 3x3 convolution
- channels: `1280 -> 1280`
- convolution: kernel 3, stride 1, padding 1, groups 1, bias present
- forward signature: `(hidden_states, output_size=None, scale=1.0)`
- `upsample_size` keyword is not accepted by this official module
- upsampler consumes no residual samples
- actual next module after this stage: `up_blocks.2`

Status after this stage:

- `TADSR_UNET_UPBLOCK1_UPSAMPLER_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_LORA_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_EFFECTIVE_WEIGHTS: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_RESIDUAL_CONTRACT_ORACLE: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_INTERPOLATION_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_CONV_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_SYNTHETIC_ALIGNMENT: PASS`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_THROUGH_UPSAMPLER_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_UPSAMPLER_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK1_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Key numerical evidence:

| Check | Status | Max abs error | Mean abs error | Cosine |
|---|---|---:|---:|---:|
| isolated interpolation | PASS | `0.0` | `0.0` | `1.0` |
| isolated conv/output | PASS | `1.71661376953125e-05` | `1.1045599434211796e-06` | `0.9999999999999649` |
| synthetic output | PASS | `3.337860107421875e-06` | `3.307456019285837e-07` | `0.9999999999999574` |
| selected bridge output | PASS | `8.705258369445801e-05` | `7.895466114504756e-06` | `0.999999999998297` |

Next target: full local `up_blocks.1` aggregate verification. `up_blocks.2/3`,
full UNet forward, generic runtime LoRA, and full TADSR inference remain
intentionally incomplete.
## Latest TADSR UNet up_blocks.1 local aggregate verification update

This stage does not open `up_blocks.2`; it only verifies full local
`up_blocks.1` after the individual leaves have already passed:

`up_blocks.1.resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0`.

Official PyTorch local forward was audited with the same deterministic entry
path used by the Jittor bridge:

`sample -> center -> conv_in -> time_proj -> time_embedding -> all down_blocks -> mid_block -> up_blocks.0 -> local up_blocks.1`.

Key official findings:

- official class: `diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D`
- official local output shape: `[1, 1280, 16, 16]`
- local output states: `NOT_APPLICABLE` in this configuration
- manual chain vs official local `up_blocks.1(...)`: `max_abs_error = 0.0`,
  `mean_abs_error = 0.0`, `cosine_similarity = 1.0`
- the audit does not run full UNet forward, does not enter `up_blocks.2/3`, and
  does not run full TADSR inference

Residual contract:

| Local step | Residual behavior |
|---|---|
| `up_blocks.1.resnets.0` | consumes `down_blocks.2.output_state_1` |
| `up_blocks.1.attentions.0` | consumes no residual |
| `up_blocks.1.resnets.1` | consumes `down_blocks.2.output_state_0` |
| `up_blocks.1.attentions.1` | consumes no residual |
| `up_blocks.1.resnets.2` | consumes `down_blocks.1.output_state_2` |
| `up_blocks.1.attentions.2` | consumes no residual |
| `up_blocks.1.upsamplers.0` | consumes no residual |

Jittor alignment evidence:

| Check | Status | Max abs error | Mean abs error | Cosine |
|---|---|---:|---:|---:|
| `TADSR_UNET_UPBLOCK1_OUTPUT_HIDDEN_ALIGNMENT` | PASS | `8.705258369445801e-05` | `7.895466114504756e-06` | `0.999999999998297` |
| `TADSR_UNET_UPBLOCK1_OUTPUT_STATES_ALIGNMENT` | NOT_APPLICABLE | n/a | n/a | n/a |
| `TADSR_UNET_UPBLOCK1_ALIGNMENT` | PASS | aggregate | aggregate | aggregate |

New artifacts:

- `tools/audit_official_tadsr_unet_upblock1_local.py`
- `tools/export_tadsr_unet_upblock1_local_oracle.py`
- `tests_jittor_alignment/test_unet_upblock1_local_alignment.py`
- `experiments/full_repro/unet_alignment/audit_tadsr_unet_upblock1_local.json`
- `experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock1_local/unet_upblock1_local_oracle_metadata.json`
- `experiments/full_repro/unet_alignment/jittor_unet_upblock1_local_alignment.json`

Final audit after this stage:

- `TADSR_UNET_UPBLOCK1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK2_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.2.resnets.0`. Full UNet forward and full TADSR
inference remain intentionally disabled.

## Latest TADSR UNet up_blocks.2.resnets.0 alignment update

This stage opens only the first leaf of `up_blocks.2`: `up_blocks.2.resnets.0`.
It does not run `up_blocks.2.attentions.0`, the rest of `up_blocks.2`,
`up_blocks.3`, full UNet forward, generic runtime LoRA, or full TADSR
inference.

Official audit results:

- `up_blocks.2` class: `diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D`
- topology: `resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2 -> upsamplers.0`
- current target: `up_blocks.2.resnets.0`
- actual next module after this stage: `up_blocks.2.attentions.0`
- consumed residual: `down_blocks.1.output_state_1`
- hidden input shape: `[1, 1280, 16, 16]`
- residual shape: `[1, 640, 16, 16]`
- concat shape: `[1, 1920, 16, 16]`
- output shape: `[1, 640, 16, 16]`

Alignment evidence:

| Check | Status | Max abs error | Mean abs error | Cosine |
|---|---|---:|---:|---:|
| residual concat | PASS | `0.0` | `0.0` | `1.0` |
| isolated output | PASS | `2.288818359375e-05` | `1.204517431574459e-06` | `0.9999999999999304` |
| synthetic output | PASS | `3.814697265625e-05` | `7.778327926644124e-06` | `0.9999999999947845` |
| bridge residual | PASS | `3.4332275390625e-05` | `1.2935612232922721e-06` | `0.9999999999990887` |
| bridge concat | PASS | `8.705258369445801e-05` | `5.694831150767262e-06` | `0.9999999999983171` |
| bridge output | PASS | `5.8650970458984375e-05` | `5.473751028262086e-06` | `0.999999999998551` |

Final audit after this stage:

- `TADSR_UNET_UPBLOCK2_RESNET0_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK2_PRE_ATTENTION0_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK2_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.2.attentions.0`. Full `up_blocks.2`, full UNet
forward, generic runtime LoRA, and full TADSR inference remain intentionally
disabled.

## TADSR UNet up_blocks.2 local aggregate alignment update

This stage completes the full local aggregate verification for `up_blocks.2` only.
The official PyTorch audit verifies that direct `up_blocks.2(...)` output matches
the manual chain `resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 ->
resnets.2 -> attentions.2 -> upsamplers.0`. On the Jittor side,
`TADSRUNetUpBlock2LocalTester` composes the already aligned leaves and compares
the entry bridge output against the local oracle.

New status markers:

- `TADSR_UNET_UPBLOCK2_LOCAL_FORWARD_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK2_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK2_LOCAL_ORACLE_TENSORS: PASS`
- `TADSR_UNET_UPBLOCK2_OUTPUT_HIDDEN_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK2_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE`
- `TADSR_UNET_UPBLOCK2_ALIGNMENT: PASS`

Boundary after this stage remains strict: `up_blocks.3` is unopened,
`TADSR_UNET_UPBLOCKS_ALIGNMENT` and `TADSR_UNET_FULL_FORWARD_ALIGNMENT` remain
`NOT_COMPLETE`, generic runtime LoRA integration remains `PARTIAL`, and full
Jittor TADSR inference remains `NotImplemented`. The next target is
`up_blocks.3.resnets.0` only.

## Latest update: UNet up_blocks.3.resnets.0 alignment

This stage migrates only `up_blocks.3.resnets.0`, the first ResNet in the final
TADSR UNet up block. It deliberately stops before
`up_blocks.3.attentions.0`; the remaining `up_blocks.3` modules, the full
up-block aggregate, full UNet forward, runtime LoRA, and full TADSR inference
remain out of scope.

Official audit summary:

- `up_blocks.3` class: `diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D`
- topology: `resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2`
- upsampler count: `0`
- current target: `up_blocks.3.resnets.0`
- actual next module after this stage: `up_blocks.3.attentions.0`
- consumed residual: `down_blocks.0.output_state_1`
- hidden input shape: `[1, 640, 32, 32]`
- residual shape: `[1, 320, 32, 32]`
- concat shape: `[1, 960, 32, 32]`
- output shape: `[1, 320, 32, 32]`

Alignment evidence:

| Check | Status |
|---|---|
| official topology audit | PASS |
| residual contract audit | PASS |
| LoRA/effective static weight audit | PASS |
| PyTorch oracle tensors | PASS |
| effective static weights | PASS |
| residual concat | PASS |
| norm1 / conv1 / temb projection | PASS |
| norm2 / conv2 / shortcut | PASS |
| isolated output | PASS |
| synthetic output | PASS |
| entry-to-upblock3-resnet0 bridge | PASS |
| pre-attention0 boundary | PASS |

The verified bridge is:

`entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0`.

Final-audit statuses after this stage:

- `TADSR_UNET_UPBLOCK3_RESNET0_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_PRE_ATTENTION0_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target after this stage was `up_blocks.3.attentions.0`; that target is now completed by the following update.

## Latest update: UNet up_blocks.3.attentions.0 alignment

This stage migrates only `up_blocks.3.attentions.0`, the first attention module
after the already aligned `up_blocks.3.resnets.0`. It deliberately stops before
`up_blocks.3.resnets.1`; the remaining `up_blocks.3` modules, the full
up-block aggregate, full UNet forward, runtime LoRA, and full TADSR inference
remain out of scope.

Official audit summary:

- target module: `up_blocks.3.attentions.0`
- module type: `Transformer2DModel`
- transformer block count: `1`
- input/output hidden shape: `[1, 320, 32, 32]`
- encoder hidden-state shape: `[1, 77, 1024]`
- residual behavior: consumes no accumulated residual samples
- next official module after attention0: `up_blocks.3.resnets.1`

Evidence:

| Check | Status |
|---|---|
| official attention0 audit | PASS |
| transformer audit | PASS |
| LoRA/effective static weight audit | PASS |
| residual contract audit | PASS |
| PyTorch oracle tensors | PASS |
| effective static weights | PASS |
| norm / sequence / proj_in | PASS |
| self-attention QKV/output/residual | PASS |
| cross-attention QKV/output/residual | PASS |
| feed-forward / transformer block | PASS |
| isolated attention0 output | PASS |
| entry-to-attention0 bridge | PASS |
| attention0 bridge aggregate | PASS |
| pre-resnet1 boundary | PASS |

The verified bridge is:

`entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0`.

Numerical evidence:

- isolated attention0 output max_abs_error: `2.288818359375e-05`;
- isolated attention0 output cosine: `0.9999999999943571`;
- long bridge final output max_abs_error: `2.2172927856445312e-05`;
- long bridge final output cosine: `0.9999999999977839`.

Final-audit statuses after this stage:

- `TADSR_UNET_UPBLOCK3_ATTENTION0_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_PRE_RESNET1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target after that stage was `up_blocks.3.resnets.1`; that target is now completed by the following update.

## Latest update: UNet up_blocks.3.resnets.1 alignment

This stage migrates only `up_blocks.3.resnets.1`, the ResNet block immediately
after the already aligned `up_blocks.3.attentions.0`. It deliberately stops
before `up_blocks.3.attentions.1`; the remaining `up_blocks.3` modules, the full
up-block aggregate, full UNet forward, runtime LoRA, and full TADSR inference
remain out of scope.

Official audit summary:

- target module: `up_blocks.3.resnets.1`
- module type: `ResnetBlock2D`
- official `up_blocks.3` order: `resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2 -> attentions.2`
- hidden input from attention0: `[1, 320, 32, 32]`
- consumed residual: `down_blocks.0.output_state_0`
- consumed residual global index: `1`
- consumed residual shape: `[1, 320, 32, 32]`
- concat axis: channel dimension (`axis=1`)
- concat input shape: `[1, 640, 32, 32]`
- output shape: `[1, 320, 32, 32]`
- next official module after resnet1: `up_blocks.3.attentions.1`

Evidence:

| Check | Status |
|---|---|
| official resnet1 audit | PASS |
| residual contract audit | PASS |
| LoRA/effective static weight audit | PASS |
| PyTorch oracle tensors | PASS |
| effective static weights | PASS |
| residual contract oracle | PASS |
| residual concat | PASS |
| norm1 / conv1 / temb projection / conv2 / shortcut | PASS |
| isolated resnet1 output | PASS |
| synthetic resnet1 output | PASS |
| entry-to-resnet1 bridge | PASS |
| resnet1 bridge aggregate | PASS |
| pre-attention1 boundary | PASS |

Numerical evidence:

- isolated resnet1 output max_abs_error: `1.6689300537109375e-06`;
- isolated resnet1 output cosine: `0.9999999999999583`;
- synthetic resnet1 output max_abs_error: `7.510185241699219e-06`;
- synthetic resnet1 output cosine: `0.9999999999969984`;
- long bridge final output max_abs_error: `1.0848045349121094e-05`;
- long bridge final output cosine: `0.9999999999985686`.

Oracle/effective artifacts:

- oracle directory: `experiments/full_repro/unet_alignment/oracle_tensors_unet_upblock3_resnet1/`
- effective weights: `experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet1_effective_weights.npz`
- oracle tensor directory size observed on the server: `66M`
- effective weights archive size observed on the server: `13M`
- large `.npy` / `.npz` files are generated for verification and ignored by git.

Final-audit statuses after this stage:

- `TADSR_UNET_UPBLOCK3_RESNET1_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_PRE_ATTENTION1_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_FULL_FORWARD_ALIGNMENT: NOT_COMPLETE`
- `TADSR_UNET_LORA_RUNTIME_INTEGRATION: PARTIAL`
- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`

Next target: `up_blocks.3.attentions.1`.
## up_blocks.3.attentions.1 update

This stage opens only `up_blocks.3.attentions.1` after the previously aligned path through `up_blocks.3.resnets.1`. The official PyTorch audit, oracle tensor export, static effective-weight export, isolated Jittor attention checks, and the entry-to-attention1 bridge all pass.

New PASS markers:

- `TADSR_UNET_UPBLOCK3_ATTENTION1_AUDIT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER_AUDIT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_LORA_AUDIT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_AUDIT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ORACLE_TENSORS`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_EFFECTIVE_WEIGHTS`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_RESIDUAL_CONTRACT_ORACLE`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_NORM_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_PROJ_IN_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_SEQUENCE_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_QKV_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN1_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN1_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_QKV_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ATTN2_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_AFTER_ATTN2_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_FF_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_TRANSFORMER0_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_ALIGNMENT`
- `TADSR_UNET_ENTRY_DOWNBLOCKS_MIDBLOCK_UPBLOCK0_UPBLOCK1_UPBLOCK2_UPBLOCK3_RESNET0_ATTENTION0_RESNET1_ATTENTION1_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_ATTENTION1_BRIDGE_ALIGNMENT`
- `TADSR_UNET_UPBLOCK3_PRE_RESNET2_ALIGNMENT`

`up_blocks.3.attentions.1` does not consume residual samples. The remaining residual tuple is unchanged and reserved for `up_blocks.3.resnets.2`. Full `up_blocks.3`, all up blocks, full UNet forward, generic runtime LoRA integration, and full TADSR inference remain `NOT_COMPLETE`.

### Latest UNet up_blocks.3.resnets.2 Alignment Update

- `up_blocks.3.resnets.2` has been audited against the official PyTorch TADSR UNet and numerically aligned in Jittor.
- The verified local bridge now reaches: entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 -> up_blocks.3.resnets.1 -> up_blocks.3.attentions.1 -> up_blocks.3.resnets.2.
- The official residual contract was verified: `up_blocks.3.resnets.2` consumes the final remaining up_blocks.3 skip residual, concatenates it with the attention1 output on channel axis, and leaves zero local residuals for the following attention module.
- `TADSR_UNET_UPBLOCK3_RESNET2_BRIDGE_ALIGNMENT: PASS` and `TADSR_UNET_UPBLOCK3_PRE_ATTENTION2_ALIGNMENT: PASS`.
- `up_blocks.3.attentions.2`, full `up_blocks.3`, full UNet forward, generic runtime LoRA integration, and full TADSR inference remain `NOT_COMPLETE`/`PARTIAL` by design.
- Next migration target: `up_blocks.3.attentions.2`.

### Latest UNet migration checkpoint: up_blocks.3.attentions.2
- `TADSR_UNET_UPBLOCK3_ATTENTION2_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_ATTENTION2_BRIDGE_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_PRE_LOCAL_AGGREGATE_ALIGNMENT: PASS`
- Still intentionally `NOT_COMPLETE`: full local `up_blocks.3` aggregate, full `up_blocks`, full UNet forward, runtime LoRA integration, and full TADSR inference.
- Next recommended step: verify the full local `up_blocks.3` aggregate/output-tail boundary only; do not open full UNet forward yet.

## Latest update: UNet up_blocks.3 local aggregate alignment

This stage verifies the full local `up_blocks.3` aggregate without executing
the output tail. The official local `up_blocks.3(...)` forward matches the
manual sequence `resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 ->
resnets.2 -> attentions.2`, and the Jittor bridge validates the output after:

`entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> full local up_blocks.3 aggregate`

Key markers:

- `TADSR_UNET_UPBLOCK3_LOCAL_FORWARD_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK3_RESIDUAL_CONTRACT_AUDIT: PASS`
- `TADSR_UNET_UPBLOCK3_OUTPUT_HIDDEN_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_OUTPUT_STATES_ALIGNMENT: NOT_APPLICABLE`
- `TADSR_UNET_OUTPUT_TAIL_BOUNDARY_AUDIT: PASS`
- `TADSR_UNET_OUTPUT_TAIL_BOUNDARY_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCK3_ALIGNMENT: PASS`
- `TADSR_UNET_UPBLOCKS_ALIGNMENT: PASS`

The output tail is the next boundary and remains unported. Full UNet forward,
runtime LoRA integration, and full TADSR inference remain `NOT_COMPLETE` /
`PARTIAL` as appropriate.

## Latest update: UNet output tail alignment

The output tail has now been audited against the official PyTorch TADSR UNet,
exported as oracle tensors, ported as a minimal Jittor tester, and numerically
aligned. The tested chain is exactly:

`hidden_states -> conv_norm_out -> conv_act -> conv_out`

The long bridge now validates:

`entry -> all down_blocks -> full local mid_block -> full local up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 -> full local up_blocks.3 -> output tail`

New evidence:

- official output-tail topology/LoRA/local execution audit: `PASS`;
- PyTorch output-tail oracle tensors: `PASS`;
- LoRA-merged static effective weights for `conv_out`: `PASS`;
- isolated output-tail norm/act/conv alignment: `PASS`;
- synthetic output-tail alignment: `PASS`;
- entry-to-output-tail bridge alignment: `PASS`.

The migration still deliberately stops short of official full `unet.forward`.
`TADSR_UNET_FULL_FORWARD_ALIGNMENT` remains `NOT_COMPLETE`,
`TADSR_UNET_LORA_RUNTIME_INTEGRATION` remains `PARTIAL`, and
`JITTOR_FULL_INFERENCE` remains `NOT_COMPLETE`.
## Manual full-chain wrapper alignment

The UNet path now has a dedicated manual full-chain wrapper tester:

`sample -> center -> conv_in -> time_proj -> time_embedding -> down_blocks.0/1/2/3 -> mid_block -> up_blocks.0/1/2/3 -> output tail`

The PyTorch side exports an oracle for this exact manual chain without invoking
official `unet.forward`. The Jittor side reuses the existing local block testers
and compares entry, down block, mid block, up block, output-tail, and final
output tensors.

This proves the already ported local pieces compose correctly through the output
tail. It does not prove official full UNet forward parity; that remains the next
separate migration stage.

## Official full-forward alignment

The official PyTorch `unet.forward` has now been audited and exported as the
UNet full-forward oracle. The Jittor alignment-only full-forward tester reuses
the manual chain and matches the official output tensor.

This stage changes only the UNet full-forward numerical alignment status. It
does not enable production inference, scheduler denoising, VAE integration,
runtime LoRA, or image generation.
