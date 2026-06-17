# Diagnostic image-smoke metrics

This artifact is a diagnostic one-step tensor visualization report. It is not full TADSR inference, not a production restoration result, and not image/video generation completion.

## Markers

| Marker | Status |
|---|---|
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY` | `PASS` |

- Status: `PASS`
- Source stage: `clamped_output`
- Blocker: `none`

## Metrics

| Metric | Value |
|---|---:|
| `mae` | `2.4488862019704055e-07` |
| `max_abs` | `4.308298230171204e-06` |
| `rmse` | `3.4760109612852134e-07` |
| `psnr` | `129.1783772542593` |
| `ssim` | `1.0` |
| `relative_l2` | `5.783607204842348e-07` |
| `cosine_similarity` | `0.9999999999998049` |

## Safety flags

- `full_inference_executed`: `False`
- `denoising_loop_executed`: `False`
- `multi_step_executed`: `False`
- `production_image_generation`: `False`
- `image_is_diagnostic`: `True`
- `raw_tensors_committed`: `False`
- `local_tensors_used`: `True`
- `diagnostic_image_generated`: `True`
