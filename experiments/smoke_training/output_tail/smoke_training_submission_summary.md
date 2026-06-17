# Small-data smoke-training submission summary

`TADSR_SMOKE_TRAINING_SUBMISSION_SUMMARY: PASS`

本摘要把已经完成的小数据训练对齐证据集中到一个提交入口。它不是完整 TADSR 训练，而是 output-tail `conv_out` 子任务上的 PyTorch-vs-Jittor 小规模训练可行性验证。

## Task definition

- Task: output-tail conv_out small-data regression using exported feature-target pairs
- Train / validation split: 24 train / 8 validation over 32 output-tail samples
- Limitation: small-data output-tail smoke training only; not full TADSR end-to-end training

## Loss evidence

| Framework | Steps | First loss | Last loss | Decrease | Relative decrease |
|---|---:|---:|---:|---:|---:|
| pytorch | 300 | 0.005236602388322353 | 0.00014055753126740456 | 0.005096044857054949 | 0.973159 |
| jittor | 300 | 0.005236602853983641 | 0.00014113690122030675 | 0.005095465952763334 | 0.973048 |

## Alignment summary

- Alignment metrics status: `PASS`
- Multi-seed summary status: `PASS`
- Prediction alignment file: `experiments/smoke_training/output_tail/prediction_alignment.csv`

## Visualizations

- `experiments/smoke_training/output_tail/visualizations/loss_gap_curve.png`
- `experiments/smoke_training/output_tail/visualizations/performance_samples_per_sec.png`
- `experiments/smoke_training/output_tail/visualizations/performance_step_time.png`
- `experiments/smoke_training/output_tail/visualizations/prediction_error_heatmap.png`
- `experiments/smoke_training/output_tail/visualizations/prediction_target_heatmap.png`
- `experiments/smoke_training/output_tail/visualizations/pytorch_jittor_prediction_heatmap.png`
- `experiments/smoke_training/output_tail/visualizations/relative_loss_gap_curve.png`
- `experiments/smoke_training/output_tail/visualizations/train_val_loss_curve.png`
- `experiments/smoke_training/output_tail/loss_curve.png`
- `experiments/smoke_training/output_tail/train_val_loss_curve.png`

## Safety

- `does_not_train_now`: `True`
- `does_not_generate_image_or_video`: `True`
- `does_not_save_new_weights`: `True`
- `does_not_claim_full_tadsr_training`: `True`
