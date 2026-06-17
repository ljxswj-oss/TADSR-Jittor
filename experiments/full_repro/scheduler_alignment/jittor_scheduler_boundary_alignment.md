# 中文阅读说明：jittor_scheduler_boundary_alignment.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Jittor Scheduler Boundary Alignment

Status: **PASS**

| Check | Status | Max abs error | Tolerance |
|---|---|---:|---:|
| `timesteps` | PASS | 0.0 | 0.0 |
| `scale_model_input` | PASS | 0.0 | 1e-08 |
| `scheduler_prev_sample` | PASS | 4.656612873077393e-10 | 0.0001 |
| `scheduler_pred_original_sample` | PASS | 0.0 | 1e-05 |
| `unet_scheduler_prev_sample` | PASS | 4.656612873077393e-10 | 0.0001 |
| `unet_scheduler_pred_original_sample` | PASS | 0.0 | 1e-05 |

- `TADSR_SCHEDULER_TIMESTEPS_ALIGNMENT`: `PASS`
- `TADSR_SCHEDULER_SCALE_MODEL_INPUT_ALIGNMENT`: `NOT_APPLICABLE_NOOP`
- `TADSR_SCHEDULER_STEP_ALIGNMENT`: `PASS`
- `TADSR_UNET_SCHEDULER_ONE_STEP_ALIGNMENT`: `PASS`
- `TADSR_SCHEDULER_BOUNDARY_ALIGNMENT`: `PASS`

No denoising loop, VAE decode, image save, or full TADSR inference was executed.
