# LoRA Layer Formula Alignment

Status marker: `TADSR_RUNTIME_LORA_LAYER_FORMULA_ALIGNMENT`

Status: **PASS**

本检查只验证固定 adapter / 固定 scale 下 runtime LoRA 公式与 static effective weight 公式等价。

它不表示 dynamic runtime LoRA adapter switching 已实现，也不接入 full TADSR inference。

## Metrics

| Metric | Value |
|---|---:|
| max_abs_error | 7.105427357601002e-15 |
| mean_abs_error | 1.0400715697724681e-15 |
| relative_l2_error | 2.4066886277797065e-16 |
| cosine_similarity | 0.9999999999999999 |

## Scope

- dynamic_runtime_adapter_switching_implemented: `False`
- full_tadsr_runtime_lora_integrated: `False`
- tadsr_dynamic_runtime_lora_status: `NOT_IMPLEMENTED_BY_DESIGN`
