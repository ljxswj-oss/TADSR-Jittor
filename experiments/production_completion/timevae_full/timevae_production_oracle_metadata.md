# TimeVAE production oracle metadata

`TADSR_TIMEVAE_PRODUCTION_ORACLE_METADATA: PASS`

- Metadata only: `True`
- Official env status: `PASS`
- Dependency status: `PASS`
- Official torch version: `2.0.1+cu118`
- Diffusers available: `True`
- Overlay used: `True`
- Strict env modified: `False`
- Forward executed: `True`
- Raw tensor saved: `False`
- Image/video output generated: `False`
- Ready for TimeVAE preflight: `True`
- Ready for one-step contract: `True`

## Required metadata fields

| Field | Value |
|---|---|
| `encode_input_shape` | `[1, 3, 64, 64]` |
| `encode_input_dtype` | `torch.float32` |
| `encode_input_range` | `[-0.9999203681945801, 0.9998066425323486]` |
| `posterior_mean_shape` | `[1, 4, 8, 8]` |
| `posterior_logvar_shape` | `[1, 4, 8, 8]` |
| `latent_shape` | `[1, 4, 8, 8]` |
| `scaling_factor` | `0.18215` |
| `scaled_latent_shape` | `[1, 4, 8, 8]` |
| `decode_input_shape` | `[1, 4, 8, 8]` |
| `decode_output_shape` | `[1, 3, 64, 64]` |
| `clamp_policy` | `torch.clamp(decoded, -1, 1), metadata only` |
| `encoder_hook_path` | `metadata_only_not_forward_executed` |
| `decoder_hook_path` | `metadata_only_not_forward_executed` |

## Safety policy

- This metadata exporter did not run full TADSR inference.
- This metadata exporter did not run the denoising loop.
- This metadata exporter did not generate restored images or videos.
- Raw tensor saving is disabled unless explicitly requested, and local tensor paths are git-ignored.
