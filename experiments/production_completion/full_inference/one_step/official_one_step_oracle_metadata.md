# Official one-step tensor oracle

`TADSR_ONE_STEP_OFFICIAL_ORACLE_TENSORS: PASS`

This file records a controlled one-step tensor oracle only.
It does not run the production full-inference CLI, a full denoising loop, image generation, or video generation.

## Stage summary

| Stage | Shape | DType | Min | Max | Mean | Std | Local tensor |
|---|---|---|---:|---:|---:|---:|---|
| `input_tensor` | `[1, 3, 256, 256]` | `float32` | `-1.0` | `1.0` | `-2.4835269396561444e-09` | `0.5773532390594482` | `local_tensors/input_tensor.npy` |
| `scheduler_timestep` | `[1]` | `int64` | `1.0` | `1.0` | `1.0` | `0.0` | `local_tensors/scheduler_timestep.npy` |
| `encoder_hidden_states` | `[1, 77, 1024]` | `float32` | `-0.5` | `0.5` | `9.289035318715833e-09` | `0.2886787950992584` | `local_tensors/encoder_hidden_states.npy` |
| `sample_epsilon` | `[1, 4, 32, 32]` | `float32` | `-0.5` | `0.5` | `0.0` | `0.28874561190605164` | `local_tensors/sample_epsilon.npy` |
| `encoded_latent` | `[1, 4, 32, 32]` | `float32` | `-36.4074821472168` | `20.663257598876953` | `-0.11999475955963135` | `7.406636714935303` | `local_tensors/encoded_latent.npy` |
| `scaled_latent` | `[1, 4, 32, 32]` | `float32` | `-6.631623268127441` | `3.76381254196167` | `-0.021857082843780518` | `1.3491188287734985` | `local_tensors/scaled_latent.npy` |
| `unet_model_pred` | `[1, 4, 32, 32]` | `float32` | `-1.2700740098953247` | `0.6109416484832764` | `0.06714534759521484` | `0.20003287494182587` | `local_tensors/unet_model_pred.npy` |
| `alpha_prod_t` | `[1]` | `float32` | `0.9982960224151611` | `0.9982960224151611` | `0.9982960224151611` | `0.0` | `local_tensors/alpha_prod_t.npy` |
| `sqrt_alpha_prod_t` | `[1]` | `float32` | `0.9991476535797119` | `0.9991476535797119` | `0.9991476535797119` | `0.0` | `local_tensors/sqrt_alpha_prod_t.npy` |
| `x0_from_res` | `[1, 4, 32, 32]` | `float32` | `-7.117111682891846` | `3.810164213180542` | `-0.08902102708816528` | `1.2874069213867188` | `local_tensors/x0_from_res.npy` |
| `decode_input` | `[1, 4, 32, 32]` | `float32` | `-39.07280349731445` | `20.917728424072266` | `-0.48872363567352295` | `7.0678391456604` | `local_tensors/decode_input.npy` |
| `decode_output` | `[1, 3, 256, 256]` | `float32` | `-1.1031932830810547` | `1.1275092363357544` | `-0.013643980026245117` | `0.603239893913269` | `local_tensors/decode_output.npy` |
| `clamped_output` | `[1, 3, 256, 256]` | `float32` | `-1.0` | `1.0` | `-0.014104138128459454` | `0.6008453965187073` | `local_tensors/clamped_output.npy` |

## Safety

- full_inference_executed: `False`
- denoising_loop_executed: `False`
- image_or_video_generated: `False`
- production_cli_used: `False`
- raw_tensor_policy: `raw tensors are optional local_tensors only; do not stage or commit .npy/.npz`
