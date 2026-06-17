# 中文阅读说明：encoder_tail_key_mapping.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# Encoder Tail + QuantConv PyTorch-to-NPZ Key Mapping

| PyTorch key | Jittor NPZ key | Shape |
|---|---|---:|
| `encoder.conv_norm_out.weight` | `encoder__conv_norm_out__weight` | `[512]` |
| `encoder.conv_norm_out.bias` | `encoder__conv_norm_out__bias` | `[512]` |
| `encoder.conv_out.weight` | `encoder__conv_out__weight` | `[8, 512, 3, 3]` |
| `encoder.conv_out.bias` | `encoder__conv_out__bias` | `[8]` |
| `quant_conv.weight` | `quant_conv__weight` | `[8, 8, 1, 1]` |
| `quant_conv.bias` | `quant_conv__bias` | `[8]` |
