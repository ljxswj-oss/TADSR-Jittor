# TimeVAE production metadata PARTIAL diagnosis

`TADSR_TIMEVAE_METADATA_PARTIAL_DIAGNOSIS: PASS`

## Current status

- `current_status`: `PARTIAL`
- `preflight_status`: `PARTIAL`
- `metadata_contract_status`: `PARTIAL`
- `can_attempt_live_repair_on_linux`: `True`

## Missing required fields

- `dependency_blockers`
- `encode_input_shape`
- `encode_input_dtype`
- `encode_input_range`
- `encode_input_stats`
- `quant_conv_used`
- `latent_distribution_type`
- `posterior_mean_shape`
- `posterior_logvar_shape`
- `fixed_epsilon_used`
- `latent_shape`
- `latent_dtype`
- `latent_stats`
- `scaled_latent_shape`
- `scaled_latent_stats`
- `decode_input_shape`
- `decode_input_dtype`
- `post_quant_conv_used`
- `decode_output_shape`
- `decode_output_dtype`
- `decode_output_range`
- `decode_output_stats`
- `clamp_policy`

## Blocker classification

- `blocked_by_dependency`: `False`
- `blocked_by_forward_execution`: `True`
- `blocked_by_missing_diffusers`: `False`
- `blocked_by_missing_weight`: `False`
- `blocked_by_missing_shape_contract`: `True`

## Recommended repair actions

- Run the metadata exporter in Linux official env in metadata-only mode; do not save or commit raw tensors.
- Populate encode/decode shape, dtype, range, sampling, scaling and clamp policy fields.
