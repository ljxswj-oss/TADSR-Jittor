# Full inference metadata dry-run contract

`TADSR_FULL_INFERENCE_METADATA_DRY_RUN_CONTRACT: PARTIAL`

| Stage | Status | Detail |
|---|---|---|
| preprocess contract | PASS | PASS |
| condition/control contract | PARTIAL | missing feasibility summary |
| TimeVAE encode metadata | PASS | production oracle metadata |
| scheduler timesteps evidence | PASS | PASS |
| UNet full forward evidence | PASS | PASS |
| get_x0_from_res evidence | PASS | PASS |
| TimeVAE decode metadata | PASS | production oracle metadata |
| postprocess contract | PARTIAL | documented in controlled plan |
| dynamic LoRA status | PASS | Existing evidence supports the static effective-weight policy for fixed adapter / fixed scale inference. Generic dynamic runtime LoRA adapter switching remains NOT_IMPLEMENTED_BY_DESIGN and is not upgraded in Phase 1. |
| full inference guard | PASS | NotImplementedError preserved |
| final image/video output | NOT_EXECUTED | no image/video generated in this metadata contract |

## Phase 4-A metadata gate markers

- `TADSR_TIMEVAE_METADATA_REPAIR_ATTEMPT: PASS`
- `TADSR_TIMEVAE_METADATA_FIELDS_COMPLETE: PASS`
- `TADSR_TIMEVAE_METADATA_READY_FOR_ONE_STEP: PASS`
- `TADSR_TIMEVAE_LIVE_METADATA_COMPLETION: PASS`
- `TADSR_TIMEVAE_LIVE_ENCODE_METADATA: PASS`
- `TADSR_TIMEVAE_LIVE_DECODE_METADATA: PASS`
- `TADSR_TIMEVAE_LIVE_SAFETY_FLAGS: PASS`
