# Jittor one-step tensor alignment

`TADSR_ONE_STEP_JITTOR_TENSOR_RUN: PASS`
`TADSR_ONE_STEP_TENSOR_STAGE_ALIGNMENT: PASS`
`TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT: PASS`

This is a controlled one-step tensor comparison only. It does not run a full denoising loop, production full inference, image generation, video generation, or dynamic runtime LoRA.

## Stage metrics

| Stage | Status | Shape match | Max abs | Mean abs | Relative L2 | Cosine |
|---|---|---|---:|---:|---:|---:|
| `encoded_latent` | `PASS` | `True` | `2.193450927734375e-05` | `1.5060560372148757e-06` | `3.1294610760952705e-07` | `0.999999999999964` |
| `scaled_latent` | `PASS` | `True` | `4.0531158447265625e-06` | `2.737951945164241e-07` | `3.1505096742408385e-07` | `0.999999999999963` |
| `unet_model_pred` | `PASS` | `True` | `1.0319054126739502e-05` | `8.475522434991944e-07` | `5.6136108566425206e-06` | `0.9999999999845075` |
| `alpha_prod_t` | `PASS` | `True` | `0.0` | `0.0` | `0.0` | `1.0` |
| `sqrt_alpha_prod_t` | `PASS` | `True` | `0.0` | `0.0` | `0.0` | `1.0` |
| `x0_from_res` | `PASS` | `True` | `1.0609626770019531e-05` | `8.539923328498844e-07` | `9.34630595667223e-07` | `0.9999999999995691` |
| `decode_input` | `PASS` | `True` | `5.817413330078125e-05` | `4.6951951162554906e-06` | `9.361250363444269e-07` | `0.9999999999995672` |
| `decode_output` | `PASS` | `True` | `4.308298230171204e-06` | `2.53211103048064e-07` | `5.829088563472534e-07` | `0.9999999999997996` |
| `clamped_output` | `PASS` | `True` | `4.308298230171204e-06` | `2.4488862019704055e-07` | `5.783607204842348e-07` | `0.9999999999998049` |
