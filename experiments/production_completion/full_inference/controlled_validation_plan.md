# Full inference controlled validation plan

`TADSR_FULL_INFERENCE_CONTROLLED_VALIDATION_PLAN: PASS`

本轮只生成 plan，没有执行 full denoising loop，没有生成最终恢复图片或视频，也没有改变 full inference guard。

| Stage | 本轮是否执行 | 目的 |
|---|---|---|
| 4.1 metadata-only dry run | no | 检查 CLI/config/module/tensor contract |
| 4.2 one-step tensor alignment | no | 对齐一个 denoising step 的关键 tensor |
| 4.3 tiny multi-step alignment | no | 检查少量 step 的误差传播 |
| 4.4 image smoke alignment | no | 只在前序阶段通过后做 diagnostic 图像比较 |
| 4.5 production claim gate | no | 作为状态升级门槛 |

必须继续保留：

- `JITTOR_FULL_INFERENCE: NOT_COMPLETE`
- `JITTOR_FULL_PORT: PARTIAL`
