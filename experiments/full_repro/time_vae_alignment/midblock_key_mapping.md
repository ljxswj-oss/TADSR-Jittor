# 中文阅读说明：midblock_key_mapping.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# MidBlock PyTorch-to-NPZ Key Mapping

| PyTorch key | Jittor NPZ key | Shape |
|---|---|---:|
| `encoder.mid_block.attentions.0.group_norm.weight` | `encoder__mid_block__attentions__0__group_norm__weight` | `[512]` |
| `encoder.mid_block.attentions.0.group_norm.bias` | `encoder__mid_block__attentions__0__group_norm__bias` | `[512]` |
| `encoder.mid_block.attentions.0.to_q.weight` | `encoder__mid_block__attentions__0__query__weight` | `[512, 512]` |
| `encoder.mid_block.attentions.0.to_q.bias` | `encoder__mid_block__attentions__0__query__bias` | `[512]` |
| `encoder.mid_block.attentions.0.to_k.weight` | `encoder__mid_block__attentions__0__key__weight` | `[512, 512]` |
| `encoder.mid_block.attentions.0.to_k.bias` | `encoder__mid_block__attentions__0__key__bias` | `[512]` |
| `encoder.mid_block.attentions.0.to_v.weight` | `encoder__mid_block__attentions__0__value__weight` | `[512, 512]` |
| `encoder.mid_block.attentions.0.to_v.bias` | `encoder__mid_block__attentions__0__value__bias` | `[512]` |
| `encoder.mid_block.attentions.0.to_out.0.weight` | `encoder__mid_block__attentions__0__proj_attn__weight` | `[512, 512]` |
| `encoder.mid_block.attentions.0.to_out.0.bias` | `encoder__mid_block__attentions__0__proj_attn__bias` | `[512]` |
| `encoder.mid_block.resnets.0.norm1.weight` | `encoder__mid_block__resnets__0__norm1__weight` | `[512]` |
| `encoder.mid_block.resnets.0.norm1.bias` | `encoder__mid_block__resnets__0__norm1__bias` | `[512]` |
| `encoder.mid_block.resnets.0.conv1.weight` | `encoder__mid_block__resnets__0__conv1__weight` | `[512, 512, 3, 3]` |
| `encoder.mid_block.resnets.0.conv1.bias` | `encoder__mid_block__resnets__0__conv1__bias` | `[512]` |
| `encoder.mid_block.resnets.0.time_emb_proj.weight` | `encoder__mid_block__resnets__0__time_emb_proj__weight` | `[1024, 256]` |
| `encoder.mid_block.resnets.0.time_emb_proj.bias` | `encoder__mid_block__resnets__0__time_emb_proj__bias` | `[1024]` |
| `encoder.mid_block.resnets.0.norm2.weight` | `encoder__mid_block__resnets__0__norm2__weight` | `[512]` |
| `encoder.mid_block.resnets.0.norm2.bias` | `encoder__mid_block__resnets__0__norm2__bias` | `[512]` |
| `encoder.mid_block.resnets.0.conv2.weight` | `encoder__mid_block__resnets__0__conv2__weight` | `[512, 512, 3, 3]` |
| `encoder.mid_block.resnets.0.conv2.bias` | `encoder__mid_block__resnets__0__conv2__bias` | `[512]` |
| `encoder.mid_block.resnets.1.norm1.weight` | `encoder__mid_block__resnets__1__norm1__weight` | `[512]` |
| `encoder.mid_block.resnets.1.norm1.bias` | `encoder__mid_block__resnets__1__norm1__bias` | `[512]` |
| `encoder.mid_block.resnets.1.conv1.weight` | `encoder__mid_block__resnets__1__conv1__weight` | `[512, 512, 3, 3]` |
| `encoder.mid_block.resnets.1.conv1.bias` | `encoder__mid_block__resnets__1__conv1__bias` | `[512]` |
| `encoder.mid_block.resnets.1.time_emb_proj.weight` | `encoder__mid_block__resnets__1__time_emb_proj__weight` | `[1024, 256]` |
| `encoder.mid_block.resnets.1.time_emb_proj.bias` | `encoder__mid_block__resnets__1__time_emb_proj__bias` | `[1024]` |
| `encoder.mid_block.resnets.1.norm2.weight` | `encoder__mid_block__resnets__1__norm2__weight` | `[512]` |
| `encoder.mid_block.resnets.1.norm2.bias` | `encoder__mid_block__resnets__1__norm2__bias` | `[512]` |
| `encoder.mid_block.resnets.1.conv2.weight` | `encoder__mid_block__resnets__1__conv2__weight` | `[512, 512, 3, 3]` |
| `encoder.mid_block.resnets.1.conv2.bias` | `encoder__mid_block__resnets__1__conv2__bias` | `[512]` |
