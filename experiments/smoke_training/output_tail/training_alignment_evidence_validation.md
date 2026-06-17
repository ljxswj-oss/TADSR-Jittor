# 训练证据完整性审计

总体状态：**PASS**

本审计只读取已经生成的 smoke training 文件，不重新训练，不导入 `torch` 或 `jittor`。

## Marker 汇总

| Marker | Status |
|---|---|
| `TADSR_TRAINING_LOSS_CURVE_EVIDENCE` | `PASS` |
| `TADSR_TRAINING_PERFORMANCE_LOG_EVIDENCE` | `PASS` |
| `TADSR_TRAINING_OUTPUT_ALIGNMENT_EVIDENCE` | `PASS` |
| `TADSR_TRAINING_GRAD_PARAM_UPDATE_EVIDENCE` | `PARTIAL` |
| `TADSR_TRAINING_ALIGNMENT_EVIDENCE_VALIDATION` | `PASS` |
| `TADSR_TRAINING_EVIDENCE_TEACHER_READY` | `PASS` |

## 文件检查

| 检查项 | 状态 | 文件 | 说明 |
|---|---|---|---|
| PyTorch loss log | `PASS` | `experiments/smoke_training/output_tail/pytorch/loss.csv` (存在) | PyTorch 小数据训练 loss 逐步记录。 |
| Jittor loss log | `PASS` | `experiments/smoke_training/output_tail/jittor/loss.csv` (存在) | Jittor 小数据训练 loss 逐步记录。 |
| train/validation loss | `PASS` | `experiments/smoke_training/output_tail/pytorch/validation_loss.csv` (存在)<br>`experiments/smoke_training/output_tail/jittor/validation_loss.csv` (存在)<br>`experiments/smoke_training/output_tail/visualizations/train_val_loss_curve.png` (存在) | 包含训练/验证 loss 与曲线。 |
| loss gap curve | `PASS` | `experiments/smoke_training/output_tail/visualizations/loss_gap_curve.png` (存在)<br>`experiments/smoke_training/output_tail/visualizations/relative_loss_gap_curve.png` (存在) | 可视化 PyTorch 与 Jittor loss 差距。 |
| performance log | `PASS` | `experiments/smoke_training/output_tail/pytorch/performance_log.csv` (存在)<br>`experiments/smoke_training/output_tail/jittor/performance_log.csv` (存在)<br>`experiments/smoke_training/output_tail/visualizations/performance_step_time.png` (存在)<br>`experiments/smoke_training/output_tail/visualizations/performance_samples_per_sec.png` (存在) | 记录 step time 与吞吐，作为性能 log。 |
| prediction alignment | `PASS` | `experiments/smoke_training/output_tail/prediction_alignment.csv` (存在)<br>`experiments/smoke_training/output_tail/visualizations/prediction_error_heatmap.png` (存在)<br>`experiments/smoke_training/output_tail/visualizations/pytorch_jittor_prediction_heatmap.png` (存在) | 输出预测与误差热图。 |
| multi-seed stability | `PASS` | `experiments/smoke_training/output_tail/multiseed/multiseed_summary.json` (存在)<br>`experiments/smoke_training/output_tail/multiseed/multiseed_summary.md` (存在)<br>`experiments/smoke_training/output_tail/multiseed/multiseed_loss_curves.png` (存在) | 多 seed 稳定性记录。 |
| parameter update evidence | `PASS` | `experiments/smoke_training/output_tail/pytorch/final_weight_delta_summary.json` (存在)<br>`experiments/smoke_training/output_tail/jittor/final_weight_delta_summary.json` (存在) | 已有 final_weight_delta_summary，证明参数发生更新。 |
| gradient norm evidence | `MISSING_OPTIONAL` | `experiments/smoke_training/output_tail/pytorch/gradient_norm.csv` (缺失)<br>`experiments/smoke_training/output_tail/jittor/gradient_norm.csv` (缺失) | 当前原始训练日志未保存逐 step gradient norm，作为 optional enhancement 记录。 |
| training scripts | `PASS` | `tools/train_smoke_pytorch_output_tail.py` (存在)<br>`scripts/train_smoke_jittor_output_tail.py` (存在) | 训练脚本存在，可复查。 |
| false full training claim guard | `PASS` | `experiments/smoke_training/output_tail/smoke_training_submission_summary.md` (存在) | 材料明确 smoke training 不是 full TADSR training。 |

## 行数统计

| 文件 | 行数 |
|---|---:|
| pytorch_loss_rows | 300 |
| jittor_loss_rows | 300 |
| pytorch_validation_rows | 32 |
| jittor_validation_rows | 32 |

## 结论

- 当前证据覆盖 loss、validation、performance、output alignment、multi-seed 和 parameter update。
- 逐 step gradient norm 未作为原始训练日志保存，标记为 MISSING_OPTIONAL；不伪造、不重跑训练。
- 本实验是 small-data smoke training alignment，不声明完整 TADSR 训练已经完成。
