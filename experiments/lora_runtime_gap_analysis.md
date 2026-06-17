# LoRA Runtime Gap Analysis

Status marker: `TADSR_LORA_RUNTIME_GAP_ANALYSIS`

Status: **PASS**

`TADSR_DYNAMIC_RUNTIME_LORA_IMPLEMENTATION` remains **NOT_IMPLEMENTED_BY_DESIGN**.

## Coverage

| Item | Value |
|---|---:|
| total active LoRA pairs | 290 |
| UNet active LoRA pairs | 258 |
| TimeVAE active LoRA pairs | 32 |
| covered active LoRA pairs | 290 |

## Static effective LoRA policy

Runtime form:

```text
y = W x + scale * B(A(x))
```

Static effective form:

```text
W_eff = W + scale * B A
y = W_eff x
```

固定 adapter 和固定 scale 下二者数学等价。当前项目验证 static effective LoRA，不声称 dynamic runtime adapter switching 已完成。
