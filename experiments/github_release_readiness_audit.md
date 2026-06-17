# GitHub Release Readiness Audit

Status: **PASS**

## Markers

| Marker | Status |
|---|---|
| `TADSR_GITHUB_HEAD_ARTIFACT_AUDIT` | `PASS` |
| `TADSR_GITHUB_DELIVERABLE_SIZE_AUDIT` | `PASS` |
| `TADSR_GITHUB_WORKTREE_LARGE_ARTIFACT_AUDIT` | `PASS` |
| `TADSR_GIT_HISTORY_LARGE_BLOB_AUDIT` | `PASS` |
| `TADSR_EVIDENCE_DEPENDENCY_AUDIT` | `PASS` |
| `TADSR_GITHUB_RELEASE_SLIMMING_DECISION` | `PASS` |
| `TADSR_GITHUB_RELEASE_READINESS_AUDIT` | `PASS` |
| `TADSR_GITHUB_RELEASE_AFTER_PHASE5B_READY` | `PASS` |

## Size Summary

- Deliverables: 319.037 KB
- `.git`: 62.477 MB
- Working tree excluding `.git`: 134.714 MB
- Tracked `.npy/.npz`: 174 files, 94.851 MB
- Ignored local `.npy/.npz`: 22 files
- Largest history blob: 30.753 MB
- Blobs >100MB: 0
- Blobs >50MB: 0

## Largest Tracked Tensor Artifacts

| Size | Path | Manifest reference | Future cleanup candidate |
|---:|---|---|---|
| 30.753 MB | `experiments/full_repro/oracle_tensors/smoke/preprocess_tensors.npz` | `False` | `True` |
| 7.315 MB | `experiments/full_repro/unet_alignment/converted_unet_entry_effective_weights.npz` | `False` | `True` |
| 4.523 MB | `experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock1/time_vae_decoder_upblock1_outputs.npz` | `False` | `True` |
| 1.816 MB | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock0/time_vae_downblock0_inputs.npz` | `False` | `True` |
| 1.701 MB | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock0/time_vae_downblock0_outputs.npz` | `False` | `True` |
| 1.256 MB | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock1/time_vae_downblock1_outputs.npz` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_act1_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_act2_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_after_temb_add_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_conv1_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_conv2_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_input.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_manual_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_norm1_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_norm2_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_shortcut_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_synthetic_unet_conv_in_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_act1_output.npy` | `False` | `True` |
| 1.250 MB | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_act2_output.npy` | `False` | `True` |

## Largest History Blobs

| Size | Kind | HEAD | Path |
|---:|---|---|---|
| 30.753 MB | `tensor` | `True` | `experiments/full_repro/oracle_tensors/smoke/preprocess_tensors.npz` |
| 7.315 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/converted_unet_entry_effective_weights.npz` |
| 4.523 MB | `tensor` | `True` | `experiments/full_repro/time_vae_alignment/oracle_tensors_decoder_upblock1/time_vae_decoder_upblock1_outputs.npz` |
| 1.816 MB | `tensor` | `True` | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock0/time_vae_downblock0_inputs.npz` |
| 1.741 MB | `other` | `True` | `experiments/full_repro/weights/weight_conversion_report.json` |
| 1.701 MB | `tensor` | `True` | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock0/time_vae_downblock0_outputs.npz` |
| 1.653 MB | `other` | `True` | `experiments/full_repro/weights/diffusers_weight_verification.json` |
| 1.559 MB | `other` | `True` | `experiments/full_repro/weights/conversion_manifest.json` |
| 1.530 MB | `other` | `True` | `experiments/full_repro/weights/key_mapping.json` |
| 1.528 MB | `other` | `True` | `experiments/full_repro/weights/official_weight_inspection.json` |
| 1.256 MB | `tensor` | `True` | `experiments/full_repro/time_vae_alignment/oracle_tensors_downblock1/time_vae_downblock1_outputs.npz` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_act1_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_act2_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_after_temb_add_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_conv1_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_conv2_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_input.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_manual_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_norm1_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/entry_downblock0_resnet0_norm2_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_act1_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_act2_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_after_temb_add_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_conv1_output.npy` |
| 1.250 MB | `tensor` | `True` | `experiments/full_repro/unet_alignment/oracle_tensors_unet_downblock0_resnet0/synthetic_downblock0_resnet0_conv2_output.npy` |

## Recommendation

Manual push is safe from GitHub hard-limit and large-blob perspectives; a separate cleanup pass is optional.

This audit did not delete files, untrack files, rewrite history, add remotes, or push to GitHub.
