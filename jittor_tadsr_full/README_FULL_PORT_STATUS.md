# TADSR-Jittor full port 状态说明

当前 `JITTOR_FULL_PORT: PARTIAL`，`JITTOR_FULL_INFERENCE: NOT_COMPLETE`。

已完成主要边界级证据：UNet full-forward boundary、actual VAEHook boundary、static effective LoRA coverage、Scheduler boundary、minimal one-step decode dry-run 和 small-data smoke training alignment。

未完成且不声明完成：production full TADSR inference、full denoising loop、最终图片/视频生成、generic runtime LoRA。
