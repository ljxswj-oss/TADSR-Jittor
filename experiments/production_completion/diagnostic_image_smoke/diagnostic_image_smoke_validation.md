# Diagnostic Image-Smoke Validation

Status: **PASS**

This validator reads diagnostic artifacts only. It does not run full inference, does not run a denoising loop, and does not generate production images.

## Markers

| Marker | Status |
|---|---|
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_VALIDATION` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_FALSE_CLAIM_GUARD` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACT_POLICY` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_READY` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_EXECUTED` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_ALIGNMENT` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_NOT_FULL_INFERENCE` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_NO_RAW_TENSORS` | `PASS` |
| `TADSR_DIAGNOSTIC_IMAGE_SMOKE_ARTIFACTS_READY` | `PASS` |

## Artifact check

- Metrics JSON exists: `True`
- Diagnostic PNG count: `5`
- Staged raw tensor count: `0`

## Blocker

None.
