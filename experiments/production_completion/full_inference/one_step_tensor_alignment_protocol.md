# Controlled one-step tensor alignment protocol

`TADSR_FULL_INFERENCE_ONE_STEP_ALIGNMENT_PROTOCOL_READY: PASS`

This file is a protocol for the next phase. It does not execute one-step tensor alignment, does not run production full inference, does not run a full denoising loop, and does not generate images or videos.

## Scope

controlled one-step tensor alignment protocol for the next phase

## Official PyTorch oracle stage

- load official PyTorch TADSR assets in strict environment
- prepare one in-memory low-resolution input tensor with fixed seed
- run exactly one TimeVAE encode metadata-compatible boundary
- run exactly one UNet forward tensor boundary with fixed scheduler timestep
- run exactly one scheduler step tensor boundary
- run one decode/clamp tensor boundary only if prior tensor gates pass

## Jittor stage

- load converted Jittor weights already covered by existing reports
- mirror the same one-step tensor inputs and timestep
- compare tensor outputs at stage boundaries with absolute/relative/cosine metrics

## Metrics

- shape match
- dtype policy match
- max absolute error
- mean absolute error
- relative L2 error
- cosine similarity
- finite value check

## Stop conditions

- any shape mismatch
- any non-finite tensor statistics
- unexpected scheduler loop or multi-step denoising
- attempted final image/video generation
- attempted raw .npy/.npz commit
- full inference guard removed or bypassed

## Must-remain status markers

- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`
- `JITTOR_FULL_PORT: PARTIAL`
- `TIME_VAE_FULL_ALIGNMENT: NOT_COMPLETE`
- `TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION: NOT_IMPLEMENTED_BY_DESIGN`
