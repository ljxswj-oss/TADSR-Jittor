# 中文阅读说明：lora_key_mapping.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# LoRA Key Mapping

Status: **PARTIAL**

Key inspection only; LoRA forward/merge is not claimed as complete.

| Key | Shape | Dtype |
|---|---|---|
| `state_dict_unet__conv_in__lora_A__default_encoder__weight` | `[4, 4, 3, 3]` | `float32` |
| `state_dict_unet__conv_in__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__proj_in__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__proj_in__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[2560, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__proj_out__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__0__proj_out__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__proj_in__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__proj_in__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[2560, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__proj_out__lora_A__default_encoder__weight` | `[4, 320]` | `float32` |
| `state_dict_unet__down_blocks__0__attentions__1__proj_out__lora_B__default_encoder__weight` | `[320, 4]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__0__conv1__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__0__conv1__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__0__conv2__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__0__conv2__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__1__conv1__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__1__conv1__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__1__conv2__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__0__resnets__1__conv2__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__0__downsamplers__0__conv__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__0__downsamplers__0__conv__lora_B__default_encoder__weight` | `[320, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__proj_in__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__proj_in__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[5120, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 2560]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__proj_out__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__0__proj_out__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__proj_in__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__proj_in__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[5120, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 2560]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__proj_out__lora_A__default_encoder__weight` | `[4, 640]` | `float32` |
| `state_dict_unet__down_blocks__1__attentions__1__proj_out__lora_B__default_encoder__weight` | `[640, 4]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv1__lora_A__default_encoder__weight` | `[4, 320, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv1__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv2__lora_A__default_encoder__weight` | `[4, 640, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv2__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv_shortcut__lora_A__default_encoder__weight` | `[4, 320, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__0__conv_shortcut__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__1__conv1__lora_A__default_encoder__weight` | `[4, 640, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__1__conv1__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__1__conv2__lora_A__default_encoder__weight` | `[4, 640, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__1__resnets__1__conv2__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__1__downsamplers__0__conv__lora_A__default_encoder__weight` | `[4, 640, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__1__downsamplers__0__conv__lora_B__default_encoder__weight` | `[640, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__proj_in__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__proj_in__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[10240, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 5120]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__proj_out__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__0__proj_out__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__proj_in__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__proj_in__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_q__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_q__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_k__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_k__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_v__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_v__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_q__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_q__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_k__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_k__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_v__lora_A__default_encoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_v__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_B__default_encoder__weight` | `[10240, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__ff__net__2__lora_A__default_encoder__weight` | `[4, 5120]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__transformer_blocks__0__ff__net__2__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__proj_out__lora_A__default_encoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__down_blocks__2__attentions__1__proj_out__lora_B__default_encoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv1__lora_A__default_encoder__weight` | `[4, 640, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv1__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv2__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv2__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv_shortcut__lora_A__default_encoder__weight` | `[4, 640, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__0__conv_shortcut__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__1__conv1__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__1__conv1__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__1__conv2__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__2__resnets__1__conv2__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__2__downsamplers__0__conv__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__2__downsamplers__0__conv__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__0__conv1__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__0__conv1__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__0__conv2__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__0__conv2__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__1__conv1__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__1__conv1__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__1__conv2__lora_A__default_encoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__down_blocks__3__resnets__1__conv2__lora_B__default_encoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv1__lora_A__default_decoder__weight` | `[4, 2560, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv_shortcut__lora_A__default_decoder__weight` | `[4, 2560, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__0__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv1__lora_A__default_decoder__weight` | `[4, 2560, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv_shortcut__lora_A__default_decoder__weight` | `[4, 2560, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__1__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv1__lora_A__default_decoder__weight` | `[4, 2560, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv_shortcut__lora_A__default_decoder__weight` | `[4, 2560, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__resnets__2__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__0__upsamplers__0__conv__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__0__upsamplers__0__conv__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__proj_in__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__proj_in__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_k__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_v__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn1__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_k__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_v__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__attn2__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__ff__net__0__proj__lora_B__default_decoder__weight` | `[10240, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__ff__net__2__lora_A__default_decoder__weight` | `[4, 5120]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__transformer_blocks__0__ff__net__2__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__proj_out__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__0__proj_out__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__proj_in__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__proj_in__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_k__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_v__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn1__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_k__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_v__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__attn2__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__ff__net__0__proj__lora_B__default_decoder__weight` | `[10240, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__ff__net__2__lora_A__default_decoder__weight` | `[4, 5120]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__transformer_blocks__0__ff__net__2__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__proj_out__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__1__proj_out__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__proj_in__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__proj_in__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_k__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_v__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn1__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_q__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_q__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_k__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_k__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_v__lora_A__default_decoder__weight` | `[4, 1024]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_v__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_out__0__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__attn2__to_out__0__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__ff__net__0__proj__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__ff__net__0__proj__lora_B__default_decoder__weight` | `[10240, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__ff__net__2__lora_A__default_decoder__weight` | `[4, 5120]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__transformer_blocks__0__ff__net__2__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__proj_out__lora_A__default_decoder__weight` | `[4, 1280]` | `float32` |
| `state_dict_unet__up_blocks__1__attentions__2__proj_out__lora_B__default_decoder__weight` | `[1280, 4]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv1__lora_A__default_decoder__weight` | `[4, 2560, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv_shortcut__lora_A__default_decoder__weight` | `[4, 2560, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__0__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv1__lora_A__default_decoder__weight` | `[4, 2560, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv_shortcut__lora_A__default_decoder__weight` | `[4, 2560, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__1__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv1__lora_A__default_decoder__weight` | `[4, 1920, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv1__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv2__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv2__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv_shortcut__lora_A__default_decoder__weight` | `[4, 1920, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__resnets__2__conv_shortcut__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
| `state_dict_unet__up_blocks__1__upsamplers__0__conv__lora_A__default_decoder__weight` | `[4, 1280, 3, 3]` | `float32` |
| `state_dict_unet__up_blocks__1__upsamplers__0__conv__lora_B__default_decoder__weight` | `[1280, 4, 1, 1]` | `float32` |
