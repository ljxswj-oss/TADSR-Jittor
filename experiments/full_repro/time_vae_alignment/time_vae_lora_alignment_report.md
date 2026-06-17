# 中文阅读说明：time_vae_lora_alignment_report.md

本文件是程序生成的实验日志、audit report 或 marker report。为了保证验证脚本和证据链可复查，下面保留原始英文状态 marker、路径、命令和数值；这些内容属于代码/专业术语/机器输出，不应强行翻译。

阅读重点：

- 如果看到 `PASS`，表示对应验证项已通过。
- 如果看到 `NOT_COMPLETE` 或 `PARTIAL`，表示项目诚实保留的未完成边界。
- 本项目最终不声称 `full inference 已完成`，不声称生成最终图片/视频。
- 更适合老师快速阅读的中文总结见 `README.md`、`deliverables/TADSR-Jittor_submission_readme.md` 和 `docs/04_video_script.md`。

---

# TimeAware VAE and LoRA Alignment Progress

| Item | Status | Evidence |
|---|---|---|
| Official TimeAware VAE audit | PASS | `official_time_vae_module_tree.md / official_time_vae_state_keys.md` |
| PyTorch TimeAware VAE oracle tensors | PASS | `oracle_tensors/time_vae_oracle_inputs.npz and outputs.npz` |
| DownBlock0 structure audit | PASS | `downblock0_audit.md / downblock0_key_mapping.md` |
| DownBlock0 PyTorch oracle tensors | PASS | `oracle_tensors_downblock0/*.npz` |
| DownBlock1 structure audit | PASS | `downblock1_audit.md / downblock1_key_mapping.md` |
| DownBlock1 PyTorch oracle tensors | PASS | `oracle_tensors_downblock1/*.npz` |
| DownBlock2 structure audit | PASS | `downblock2_audit.md / downblock2_key_mapping.md` |
| DownBlock2 PyTorch oracle tensors | PASS | `oracle_tensors_downblock2/*.npz` |
| DownBlock3 structure audit | PASS | `downblock3_audit.md / downblock3_key_mapping.md` |
| DownBlock3 PyTorch oracle tensors | PASS | `oracle_tensors_downblock3/*.npz` |
| MidBlock structure audit | PASS | `midblock_audit.md / midblock_key_mapping.md` |
| MidBlock PyTorch oracle tensors | PASS | `oracle_tensors_midblock/*.npz` |
| Encoder tail structure audit | PASS | `encoder_tail_audit.md / encoder_tail_key_mapping.md` |
| Encoder tail + quant_conv PyTorch oracle tensors | PASS | `oracle_tensors_encoder_tail/*.npz` |
| Decoder up_blocks structure audit | PASS | `audit_time_vae_decoder_upblocks.json / audit_time_vae_decoder_upblock0.json` |
| Decoder up_blocks.0 PyTorch oracle tensors | PASS | `oracle_tensors_decoder_upblock0/*.npz and *.npy` |
| Decoder up_blocks.1 PyTorch oracle tensors | PASS | `oracle_tensors_decoder_upblock1/*.npz and *.npy` |
| Decoder up_blocks.2 PyTorch oracle tensors | PASS | `oracle_tensors_decoder_upblock2/*.npz and *.npy generated locally; large tensors are not intended for git commit` |
| Decoder up_blocks.3 PyTorch oracle tensors | PASS | `oracle_tensors_decoder_upblock3/*.npz and *.npy generated locally; large tensors are not intended for git commit` |
| Decoder tail structure audit | PASS | `audit_time_vae_decoder_tail.json / audit_time_vae_decoder_tail.txt` |
| Decoder tail PyTorch oracle tensors | PASS | `oracle_tensors_decoder_tail/*.npz and *.npy generated locally; large tensors are not intended for git commit` |
| Jittor Time-VAE conv_in alignment | PASS | `jittor_conv_in_alignment.json` |
| Jittor Time-VAE down_blocks.0 resnets.0 alignment | PASS | `jittor_first_block_alignment.json` |
| Jittor Time-VAE down_blocks.0 resnets.1 alignment | PASS | `jittor_resnet1_alignment.json` |
| Jittor Time-VAE down_blocks.0 downsampler alignment | PASS | `jittor_downsampler_alignment.json` |
| Jittor Time-VAE down_blocks.0 full alignment | PASS | `jittor_downblock0_alignment.json` |
| Jittor Time-VAE down_blocks.1.resnets.0 alignment | PASS | `encoder.down_blocks.1.resnets.0` |
| Jittor Time-VAE down_blocks.1.resnets.1 alignment | PASS | `encoder.down_blocks.1.resnets.1` |
| Jittor Time-VAE down_blocks.1.downsampler0 alignment | PASS | `encoder.down_blocks.1.downsamplers.0` |
| Jittor Time-VAE down_blocks.1 final alignment | PASS | `encoder.down_blocks.1` |
| Jittor Time-VAE encoder stage01 alignment | PASS | `encoder.conv_in + encoder.down_blocks.0 + encoder.down_blocks.1` |
| Jittor Time-VAE down_blocks.2.resnets.0 alignment | PASS | `encoder.down_blocks.2.resnets.0` |
| Jittor Time-VAE down_blocks.2.resnets.1 alignment | PASS | `encoder.down_blocks.2.resnets.1` |
| Jittor Time-VAE down_blocks.2.downsampler0 alignment | PASS | `encoder.down_blocks.2.downsamplers.0` |
| Jittor Time-VAE down_blocks.2 final alignment | PASS | `encoder.down_blocks.2` |
| Jittor Time-VAE encoder stage012 alignment | PASS | `encoder.conv_in + encoder.down_blocks.0 + encoder.down_blocks.1 + encoder.down_blocks.2` |
| Jittor Time-VAE down_blocks.3.resnets.0 alignment | PASS | `encoder.down_blocks.3.resnets.0` |
| Jittor Time-VAE down_blocks.3.resnets.1 alignment | PASS | `encoder.down_blocks.3.resnets.1` |
| Jittor Time-VAE down_blocks.3.downsampler0 alignment | NOT_APPLICABLE | `encoder.down_blocks.3.downsamplers.0` |
| Jittor Time-VAE down_blocks.3 final alignment | PASS | `encoder.down_blocks.3` |
| Jittor Time-VAE encoder stage0123 alignment | PASS | `encoder.conv_in + encoder.down_blocks.0 + encoder.down_blocks.1 + encoder.down_blocks.2 + encoder.down_blocks.3` |
| Jittor Time-VAE mid_block.resnets.0 alignment | PASS | `encoder.mid_block.resnets.0` |
| Jittor Time-VAE mid_block.attentions.0 alignment | PASS | `encoder.mid_block.attentions.0` |
| Jittor Time-VAE mid_block.resnets.1 alignment | PASS | `encoder.mid_block.resnets.1` |
| Jittor Time-VAE mid_block final alignment | PASS | `encoder.mid_block` |
| Jittor Time-VAE encoder stage0123_mid alignment | PASS | `encoder.conv_in + down_blocks.0..3 + mid_block` |
| Jittor Time-VAE encoder tail conv_norm_out alignment | PASS | `encoder.conv_norm_out` |
| Jittor Time-VAE encoder tail conv_act alignment | PASS | `encoder.conv_act` |
| Jittor Time-VAE encoder tail conv_out alignment | PASS | `encoder.conv_out` |
| Jittor Time-VAE encoder tail final alignment | PASS | `encoder.conv_norm_out -> conv_act -> conv_out` |
| Jittor Time-VAE quant_conv alignment | PASS | `quant_conv` |
| Jittor Time-VAE encoder stage0123_mid_tail alignment | PASS | `encoder.conv_in + down_blocks.0..3 + mid_block + tail` |
| Jittor Time-VAE encoder stage0123_mid_tail_quant alignment | PASS | `encoder.conv_in + down_blocks.0..3 + mid_block + tail + quant_conv` |
| Jittor Time-VAE moments split alignment | PASS | `DiagonalGaussianDistribution moments split + logvar clamp` |
| Jittor Time-VAE posterior mode alignment | PASS | `DiagonalGaussianDistribution.mode() deterministic mean` |
| Jittor Time-VAE post_quant_conv alignment | PASS | `post_quant_conv synthetic latent path` |
| Jittor Time-VAE decoder.conv_in alignment | PASS | `decoder.conv_in synthetic post_quant path` |
| Jittor Time-VAE decoder entry alignment | PASS | `post_quant_conv + decoder.conv_in synthetic latent path` |
| Jittor Time-VAE quant_to_decoder_entry alignment | PASS | `quant_conv moments -> mode -> post_quant_conv -> decoder.conv_in` |
| Jittor Time-VAE encoder_to_decoder_entry alignment | PASS | `encoder.conv_in + down_blocks.0..3 + mid_block + tail + quant_conv + posterior mode + post_quant_conv + decoder.conv_in` |
| Jittor Time-VAE decoder.mid_block.resnets.0 alignment | PASS | `decoder.mid_block.resnets.0 synthetic hidden` |
| Jittor Time-VAE decoder.mid_block.attentions.0 alignment | PASS | `decoder.mid_block.attentions.0 synthetic hidden` |
| Jittor Time-VAE decoder.mid_block.resnets.1 alignment | PASS | `decoder.mid_block.resnets.1 synthetic hidden` |
| Jittor Time-VAE decoder.mid_block final alignment | PASS | `decoder.conv_in(real path output) -> decoder.mid_block` |
| Jittor Time-VAE decoder.mid_block synthetic alignment | PASS | `synthetic hidden -> decoder.mid_block` |
| Jittor Time-VAE decoder_entry_midblock alignment | PASS | `z -> post_quant_conv -> decoder.conv_in -> decoder.mid_block` |
| Jittor Time-VAE quant_to_decoder_midblock alignment | PASS | `quant_conv moments -> posterior mode -> post_quant_conv -> decoder.conv_in -> decoder.mid_block` |
| Jittor Time-VAE encoder_to_decoder_midblock alignment | PASS | `encoder -> quant_conv -> posterior mode -> post_quant_conv -> decoder.conv_in -> decoder.mid_block` |
| Jittor Time-VAE decoder.up_blocks.0.resnets.0 alignment | PASS | `decoder.up_blocks.0.resnet0` |
| Jittor Time-VAE decoder.up_blocks.0.resnets.1 alignment | PASS | `decoder.up_blocks.0.resnet1` |
| Jittor Time-VAE decoder.up_blocks.0.resnets.2 alignment | PASS | `decoder.up_blocks.0.resnet2` |
| Jittor Time-VAE decoder.up_blocks.0.upsamplers.0 alignment | PASS | `decoder.up_blocks.0.upsamplers.0` |
| Jittor Time-VAE decoder.up_blocks.0 final alignment | PASS | `real decoder.mid_block output -> decoder.up_blocks.0` |
| Jittor Time-VAE decoder.up_blocks.0 synthetic alignment | PASS | `synthetic hidden -> decoder.up_blocks.0` |
| Jittor Time-VAE decoder_midblock_upblock0 alignment | PASS | `decoder.conv_in output -> decoder.mid_block -> decoder.up_blocks.0` |
| Jittor Time-VAE decoder_entry_midblock_upblock0 alignment | PASS | `z -> post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0` |
| Jittor Time-VAE quant_to_decoder_upblock0 alignment | PASS | `quant_conv moments -> posterior mode -> post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0` |
| Jittor Time-VAE encoder_to_decoder_upblock0 alignment | PASS | `encoder -> quant_conv -> posterior mode -> decoder.mid_block -> decoder.up_blocks.0` |
| Jittor Time-VAE decoder.up_blocks.1.resnets.0 alignment | PASS | `decoder.up_blocks.1.resnet0` |
| Jittor Time-VAE decoder.up_blocks.1.resnets.1 alignment | PASS | `decoder.up_blocks.1.resnet1` |
| Jittor Time-VAE decoder.up_blocks.1.resnets.2 alignment | PASS | `decoder.up_blocks.1.resnet2` |
| Jittor Time-VAE decoder.up_blocks.1.upsamplers.0 alignment | PASS | `decoder.up_blocks.1.upsamplers.0` |
| Jittor Time-VAE decoder.up_blocks.1 final alignment | PASS | `real decoder.up_blocks.0 output -> decoder.up_blocks.1` |
| Jittor Time-VAE decoder.up_blocks.1 synthetic alignment | PASS | `synthetic hidden -> decoder.up_blocks.1` |
| Jittor Time-VAE decoder_upblock0_upblock1 alignment | PASS | `decoder.mid_block output -> decoder.up_blocks.0 -> decoder.up_blocks.1` |
| Jittor Time-VAE decoder_entry_midblock_upblocks01 alignment | PASS | `latent z -> decoder entry -> mid_block -> up_blocks.0 -> up_blocks.1` |
| Jittor Time-VAE quant_to_decoder_upblock1 alignment | PASS | `moments -> posterior mode -> decoder up_blocks.1` |
| Jittor Time-VAE encoder_to_decoder_upblock1 alignment | PASS | `encoder image path -> quant_conv -> decoder up_blocks.1` |
| Jittor Time-VAE decoder.up_blocks.2.resnets.0 alignment | PASS | `decoder.up_blocks.2.resnet0` |
| Jittor Time-VAE decoder.up_blocks.2.resnets.1 alignment | PASS | `decoder.up_blocks.2.resnet1` |
| Jittor Time-VAE decoder.up_blocks.2.resnets.2 alignment | PASS | `decoder.up_blocks.2.resnet2` |
| Jittor Time-VAE decoder.up_blocks.2.upsamplers.0 alignment | PASS | `decoder.up_blocks.2.upsamplers.0` |
| Jittor Time-VAE decoder.up_blocks.2 final alignment | PASS | `real decoder.up_blocks.1 output -> decoder.up_blocks.2` |
| Jittor Time-VAE decoder.up_blocks.2 synthetic alignment | PASS | `synthetic hidden -> decoder.up_blocks.2` |
| Jittor Time-VAE decoder_upblocks012 alignment | PASS | `decoder.mid_block output -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2` |
| Jittor Time-VAE decoder_entry_midblock_upblocks012 alignment | PASS | `latent z -> decoder entry -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2` |
| Jittor Time-VAE quant_to_decoder_upblock2 alignment | PASS | `moments -> posterior mode -> decoder up_blocks.2` |
| Jittor Time-VAE encoder_to_decoder_upblock2 alignment | PASS | `encoder image path -> quant_conv -> decoder up_blocks.2` |
| Jittor Time-VAE decoder.up_blocks.3.resnets.0 alignment | PASS | `decoder.up_blocks.3.resnet0` |
| Jittor Time-VAE decoder.up_blocks.3.resnets.1 alignment | PASS | `decoder.up_blocks.3.resnet1` |
| Jittor Time-VAE decoder.up_blocks.3.resnets.2 alignment | PASS | `decoder.up_blocks.3.resnet2` |
| Jittor Time-VAE decoder.up_blocks.3.upsamplers.0 alignment | NOT_APPLICABLE | `decoder.up_blocks.3.upsamplers.0` |
| Jittor Time-VAE decoder.up_blocks.3 final alignment | PASS | `real decoder.up_blocks.2 output -> decoder.up_blocks.3` |
| Jittor Time-VAE decoder.up_blocks.3 synthetic alignment | PASS | `synthetic hidden -> decoder.up_blocks.3` |
| Jittor Time-VAE decoder_upblocks0123 alignment | PASS | `decoder.mid_block output -> decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3` |
| Jittor Time-VAE decoder_entry_midblock_upblocks0123 alignment | PASS | `latent z -> decoder entry -> mid_block -> up_blocks.0 -> up_blocks.1 -> up_blocks.2 -> up_blocks.3` |
| Jittor Time-VAE quant_to_decoder_upblock3 alignment | PASS | `moments -> posterior mode -> decoder up_blocks.3` |
| Jittor Time-VAE encoder_to_decoder_upblock3 alignment | PASS | `encoder image path -> quant_conv -> decoder up_blocks.3` |
| Jittor Time-VAE decoder.tail.conv_norm_out alignment | PASS | `isolated decoder tail norm_out` |
| Jittor Time-VAE decoder.tail.conv_act alignment | PASS | `isolated decoder tail act` |
| Jittor Time-VAE decoder.tail.conv_out alignment | PASS | `isolated decoder tail conv_out` |
| Jittor Time-VAE decoder.tail final alignment | PASS | `isolated decoder tail: conv_norm_out -> conv_act -> conv_out` |
| Jittor Time-VAE decoder.tail synthetic alignment | PASS | `synthetic hidden -> decoder tail` |
| Jittor Time-VAE decoder_upblocks0123_to_tail alignment | PASS | `decoder.up_blocks.0~3 -> decoder tail` |
| Jittor Time-VAE decoder_entry_midblock_upblocks0123_tail alignment | PASS | `latent z -> post_quant_conv -> decoder.conv_in -> mid_block -> up_blocks.0~3 -> tail` |
| Jittor Time-VAE quant_to_decoder_tail alignment | PASS | `moments -> posterior mode -> deterministic decoder stack -> tail` |
| Jittor Time-VAE encoder_to_decoder_tail alignment | PASS | `encoder image path -> quant_conv -> posterior mode -> deterministic decoder stack -> tail` |
| LoRA pair analysis | PASS | `experiments/full_repro/lora_alignment/lora_pairs.json` |
| LoRA merge formula validation | PASS | `experiments/full_repro/lora_alignment/lora_merge_validation.json` |

## DownBlock0 Audit Summary
- resnet_count: `2`
- has_downsampler: `True`
- channel_change_detected: `None`
- downsampler: `diffusers.models.downsampling.Downsample2D`, padding=0, stride=[2, 2], kernel=[3, 3]
- padding rule: PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.

## DownBlock1 Audit Summary
- resnet_count: `2`
- has_downsampler: `True`
- channel_change_detected: `True`
- `encoder.down_blocks.1.resnets.0`: conv1=[256, 128, 3, 3], conv2=[256, 256, 3, 3], time_emb_proj=[512, 256], conv_shortcut=[256, 128, 1, 1], nin_shortcut=None
- `encoder.down_blocks.1.resnets.1`: conv1=[256, 256, 3, 3], conv2=[256, 256, 3, 3], time_emb_proj=[512, 256], conv_shortcut=None, nin_shortcut=None
- downsampler: `diffusers.models.downsampling.Downsample2D`, padding=0, stride=[2, 2], kernel=[3, 3]
- padding rule: PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.

## DownBlock2 Audit Summary
- resnet_count: `2`
- has_downsampler: `True`
- channel_change_detected: `True`
- `encoder.down_blocks.2.resnets.0`: conv1=[512, 256, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256], conv_shortcut=[512, 256, 1, 1], nin_shortcut=None
- `encoder.down_blocks.2.resnets.1`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256], conv_shortcut=None, nin_shortcut=None
- downsampler: `diffusers.models.downsampling.Downsample2D`, padding=0, stride=[2, 2], kernel=[3, 3]
- padding rule: PyTorch Downsample2D applies F.pad(hidden_states, (0, 1, 0, 1)) before conv when use_conv=True and padding=0.

## DownBlock3 Audit Summary
- resnet_count: `2`
- has_downsampler: `False`
- channel_change_detected: `False`
- `encoder.down_blocks.3.resnets.0`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256], conv_shortcut=None, nin_shortcut=None
- `encoder.down_blocks.3.resnets.1`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256], conv_shortcut=None, nin_shortcut=None
- downsampler: `NOT_APPLICABLE`

## MidBlock Audit Summary
- class: `diffusers.models.unet_2d_blocks.UNetMidBlock2D`
- resnet_count: `2`
- attention_count: `1`
- `encoder.mid_block.resnets.0`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256]
- `encoder.mid_block.resnets.1`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], time_emb_proj=[1024, 256]
- `encoder.mid_block.attentions.0`: heads=1, head_dim=512, q=[512, 512], out=[512, 512], residual=True

## Encoder Tail + QuantConv Audit Summary
- `encoder.conv_norm_out`: groups=32, channels=512, eps=1e-06
- `encoder.conv_act`: SiLU
- `encoder.conv_out`: weight=[8, 512, 3, 3], padding=[1, 1], channels=512->8
- `quant_conv`: weight=[8, 8, 1, 1], padding=[0, 0], channels=8->8
- `post_quant_conv` exists but skipped this stage: True

## Decoder UpBlock0 Audit Summary
- class: `diffusers.models.unet_2d_blocks.UpDecoderBlock2D`
- input_shape: `[1, 512, 4, 4]`, output_shape: `[1, 512, 8, 8]`
- resnet_count: `3`, upsampler_count: `1`, attention_count: `0`
- module_order: `['resnets.0', 'resnets.1', 'resnets.2', 'upsamplers.0']`
- `decoder.up_blocks.0.resnets.0`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.0.resnets.1`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.0.resnets.2`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.0.upsamplers.0`: nearest x2 then conv weight=[512, 512, 3, 3], padding=[1, 1]

## Decoder UpBlock1 Audit Summary
- class: `diffusers.models.unet_2d_blocks.UpDecoderBlock2D`
- input_shape: `[1, 512, 8, 8]`, output_shape: `[1, 512, 16, 16]`
- resnet_count: `3`, upsampler_count: `1`, attention_count: `0`
- module_order: `['resnets.0', 'resnets.1', 'resnets.2', 'upsamplers.0']`
- `decoder.up_blocks.1.resnets.0`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.1.resnets.1`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.1.resnets.2`: conv1=[512, 512, 3, 3], conv2=[512, 512, 3, 3], shortcut=False
- `decoder.up_blocks.1.upsamplers.0`: nearest x2 then conv weight=[512, 512, 3, 3], padding=[1, 1]

## Decoder UpBlock2 Audit Summary
- class: `diffusers.models.unet_2d_blocks.UpDecoderBlock2D`
- input_shape: `[1, 512, 16, 16]`, output_shape: `[1, 256, 32, 32]`
- resnet_count: `3`, upsampler_count: `1`, attention_count: `0`
- module_order: `['resnets.0', 'resnets.1', 'resnets.2', 'upsamplers.0']`
- key behavior: `resnets.0` changes channels from 512 to 256 and uses `conv_shortcut`; `upsamplers.0` upsamples 16x16 to 32x32.
- `decoder.up_blocks.2.resnets.0`: conv1=[256, 512, 3, 3], conv2=[256, 256, 3, 3], shortcut=True, shortcut_weight=[256, 512, 1, 1]
- `decoder.up_blocks.2.resnets.1`: conv1=[256, 256, 3, 3], conv2=[256, 256, 3, 3], shortcut=False, shortcut_weight=None
- `decoder.up_blocks.2.resnets.2`: conv1=[256, 256, 3, 3], conv2=[256, 256, 3, 3], shortcut=False, shortcut_weight=None
- `decoder.up_blocks.2.upsamplers.0`: nearest x2 then conv weight=[256, 256, 3, 3], padding=[1, 1]

## Decoder UpBlock3 Audit Summary
- class: `diffusers.models.unet_2d_blocks.UpDecoderBlock2D`
- input_shape: `[1, 256, 32, 32]`, output_shape: `[1, 128, 32, 32]`
- resnet_count: `3`, upsampler_count: `0`, attention_count: `0`
- module_order: `['resnets.0', 'resnets.1', 'resnets.2']`
- key behavior: `resnets.0` changes channels from 256 to 128 and uses `conv_shortcut`; there is no official upsampler in `decoder.up_blocks.3`, so upsampler0 is `NOT_APPLICABLE`.
- `decoder.up_blocks.3.resnets.0`: conv1=[128, 256, 3, 3], conv2=[128, 128, 3, 3], shortcut=True, shortcut_weight=[128, 256, 1, 1]
- `decoder.up_blocks.3.resnets.1`: conv1=[128, 128, 3, 3], conv2=[128, 128, 3, 3], shortcut=False, shortcut_weight=None
- `decoder.up_blocks.3.resnets.2`: conv1=[128, 128, 3, 3], conv2=[128, 128, 3, 3], shortcut=False, shortcut_weight=None

## Decoder Tail Audit Summary
- module_order: `['decoder.conv_norm_out', 'decoder.conv_act', 'decoder.conv_out']`
- tail_input_shape: `[1, 128, 32, 32]`, tail_output_shape: `[1, 3, 32, 32]`
- `decoder.conv_norm_out`: class=GroupNorm, groups=32, channels=128, eps=1e-06, affine=True
- `decoder.conv_act`: class=SiLU, exact_operation=SiLU = x * sigmoid(x)
- `decoder.conv_out`: weight=[3, 128, 3, 3], bias=[3], padding=[1, 1], stride=[1, 1], channels=128->3
- scaling_factor_exists=True, scaling_factor_value=0.18215, applied_inside_decoder=False
- output_clamp_or_tanh_inside_decoder=False

## Key Metrics
- conv_in: status=PASS, max_abs_error=4.76837158203125e-07, mean_abs_error=1.9618148883182585e-08, cosine=0.999999999999996
- down_blocks.0.resnets.0: status=PASS, max_abs_error=1.430511474609375e-06, mean_abs_error=1.1833916602199679e-07, cosine=0.9999999999999832
- down_blocks.0.resnets.1: status=PASS, max_abs_error=1.0728836059570312e-06, mean_abs_error=1.0344708911702583e-07, cosine=0.9999999999999901
- down_blocks.0.downsampler0: status=PASS, max_abs_error=2.86102294921875e-06, mean_abs_error=2.743584808229116e-07, cosine=0.9999999999999913
- down_blocks.0 final: status=PASS, max_abs_error=3.337860107421875e-06, mean_abs_error=4.0526577649302453e-07, cosine=0.9999999999999829
- down_blocks.1.resnets.0: status=PASS, max_abs_error=6.67572021484375e-06, mean_abs_error=2.82310594457158e-07, cosine=0.9999999999999786
- down_blocks.1.resnets.1: status=PASS, max_abs_error=1.430511474609375e-06, mean_abs_error=1.5951398069091738e-07, cosine=0.9999999999999931
- down_blocks.1.downsampler0: status=PASS, max_abs_error=7.62939453125e-06, mean_abs_error=5.907103002300573e-07, cosine=0.9999999999999893
- down_blocks.1 final: status=PASS, max_abs_error=9.5367431640625e-06, mean_abs_error=8.436727796379273e-07, cosine=0.999999999999981
- encoder stage01: status=PASS, max_abs_error=1.138448715209961e-05, mean_abs_error=1.3239924641084144e-06, cosine=0.9999999999999528
- down_blocks.2.resnets.0: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=7.525580940637155e-07, cosine=0.9999999999999628
- down_blocks.2.resnets.1: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=2.683248396806448e-07, cosine=0.9999999999999958
- down_blocks.2.downsampler0: status=PASS, max_abs_error=3.0517578125e-05, mean_abs_error=2.4967854415081092e-06, cosine=0.9999999999999929
- down_blocks.2 final: status=PASS, max_abs_error=4.57763671875e-05, mean_abs_error=3.5400487377046375e-06, cosine=0.9999999999999885
- encoder stage012: status=PASS, max_abs_error=4.57763671875e-05, mean_abs_error=5.321308321981633e-06, cosine=0.9999999999999755
- down_blocks.3.resnets.0: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=7.079052011249587e-07, cosine=0.9999999999999991
- down_blocks.3.resnets.1: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=7.311082299565896e-07, cosine=0.9999999999999994
- down_blocks.3.downsampler0: status=NOT_APPLICABLE, no metric available
- down_blocks.3 final: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=1.186162990052253e-06, cosine=0.9999999999999988
- encoder stage0123: status=PASS, max_abs_error=6.866455078125e-05, mean_abs_error=6.051835953257978e-06, cosine=0.9999999999999687
- mid_block.resnets.0: status=PASS, max_abs_error=2.288818359375e-05, mean_abs_error=9.541781764710322e-07, cosine=0.9999999999999991
- mid_block.attentions.0: status=PASS, max_abs_error=2.288818359375e-05, mean_abs_error=1.4634697436122224e-06, cosine=0.9999999999999979
- mid_block.resnets.1: status=PASS, max_abs_error=2.288818359375e-05, mean_abs_error=1.0771655070129782e-06, cosine=0.9999999999999989
- mid_block final: status=PASS, max_abs_error=4.1961669921875e-05, mean_abs_error=2.7746609703172e-06, cosine=0.9999999999999941
- encoder stage0123_mid: status=PASS, max_abs_error=7.62939453125e-05, mean_abs_error=7.5620901043294e-06, cosine=0.9999999999999617
- encoder tail conv_norm_out: status=PASS, max_abs_error=4.76837158203125e-07, mean_abs_error=4.636133965618683e-08, cosine=0.9999999999999978
- encoder tail conv_act: status=PASS, max_abs_error=2.384185791015625e-07, mean_abs_error=4.342155079939403e-09, cosine=0.9999999999999998
- encoder tail conv_out: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=5.338806658983231e-07, cosine=0.999999999999995
- encoder tail final: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=5.287583917379379e-07, cosine=0.9999999999999948
- quant_conv: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=6.395857781171799e-07, cosine=0.9999999999999958
- encoder stage0123_mid_tail: status=PASS, max_abs_error=4.76837158203125e-06, mean_abs_error=9.419163689017296e-07, cosine=0.999999999999988
- encoder stage0123_mid_tail_quant: status=PASS, max_abs_error=9.5367431640625e-06, mean_abs_error=1.689302735030651e-06, cosine=0.9999999999999774
- moments split: status=PASS, max_abs_error=0.0, mean_abs_error=0.0, cosine=1.0
- posterior mode: status=PASS, max_abs_error=0.0, mean_abs_error=0.0, cosine=1.0
- post_quant_conv: status=PASS, max_abs_error=5.960464477539063e-08, mean_abs_error=4.190951585769653e-09, cosine=0.9999999999999994
- decoder.conv_in: status=PASS, max_abs_error=2.384185791015625e-07, mean_abs_error=9.190841865347466e-09, cosine=0.9999999999999976
- decoder entry: status=PASS, max_abs_error=2.384185791015625e-07, mean_abs_error=1.1058745030823047e-08, cosine=0.9999999999999974
- quant_to_decoder_entry: status=PASS, max_abs_error=1.9073486328125e-06, mean_abs_error=9.555520819048979e-08, cosine=0.9999999999999947
- encoder_to_decoder_entry: status=PASS, max_abs_error=1.9073486328125e-06, mean_abs_error=1.9293725017632823e-07, cosine=0.9999999999999838
- decoder.mid_block.resnets.0: status=PASS, max_abs_error=9.59634780883789e-06, mean_abs_error=1.5699989717177232e-06, cosine=0.9999999999981068
- decoder.mid_block.attentions.0: status=PASS, max_abs_error=1.9073486328125e-06, mean_abs_error=1.8667833501240239e-07, cosine=0.9999999999999849
- decoder.mid_block.resnets.1: status=PASS, max_abs_error=1.9073486328125e-06, mean_abs_error=2.654055606399197e-07, cosine=0.9999999999999766
- decoder.mid_block final: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=5.09384832980686e-07, cosine=0.9999999999999267
- decoder.mid_block synthetic: status=PASS, max_abs_error=1.5974044799804688e-05, mean_abs_error=2.900418621720746e-06, cosine=0.9999999999973104
- decoder_entry_midblock: status=PASS, max_abs_error=2.384185791015625e-06, mean_abs_error=3.6261644709156826e-07, cosine=0.9999999999999295
- quant_to_decoder_midblock: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=5.609419702068408e-07, cosine=0.999999999999912
- encoder_to_decoder_midblock: status=PASS, max_abs_error=4.500150680541992e-06, mean_abs_error=6.389774540593862e-07, cosine=0.9999999999998869
- decoder.up_blocks.0.resnets.0: status=PASS, max_abs_error=1.430511474609375e-06, mean_abs_error=2.1671326067007612e-07, cosine=0.9999999999999893
- decoder.up_blocks.0.resnets.1: status=PASS, max_abs_error=2.1457672119140625e-06, mean_abs_error=3.585385002224939e-07, cosine=0.999999999999974
- decoder.up_blocks.0.resnets.2: status=PASS, max_abs_error=3.0994415283203125e-06, mean_abs_error=5.099022928334307e-07, cosine=0.9999999999999413
- decoder.up_blocks.0.upsamplers.0: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=1.2780401164036448e-06, cosine=0.9999999999999653
- decoder.up_blocks.0 final: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=1.2780401164036448e-06, cosine=0.9999999999999653
- decoder.up_blocks.0 synthetic: status=PASS, max_abs_error=7.891654968261719e-05, mean_abs_error=5.974327631719234e-06, cosine=0.9999999999955662
- decoder_midblock_upblock0: status=PASS, max_abs_error=1.239776611328125e-05, mean_abs_error=1.6619595442080026e-06, cosine=0.999999999999856
- decoder_entry_midblock_upblock0: status=PASS, max_abs_error=1.33514404296875e-05, mean_abs_error=1.7265455696247045e-06, cosine=0.9999999999998436
- quant_to_decoder_upblock0: status=PASS, max_abs_error=1.52587890625e-05, mean_abs_error=2.0016136375033966e-06, cosine=0.9999999999999187
- encoder_to_decoder_upblock0: status=PASS, max_abs_error=2.288818359375e-05, mean_abs_error=2.426172670766391e-06, cosine=0.9999999999998798
- decoder.up_blocks.1.resnets.0: status=PASS, max_abs_error=3.814697265625e-06, mean_abs_error=5.505788180926174e-07, cosine=0.999999999999994
- decoder.up_blocks.1.resnets.1: status=PASS, max_abs_error=7.62939453125e-06, mean_abs_error=9.135429195339384e-07, cosine=0.9999999999999847
- decoder.up_blocks.1.resnets.2: status=PASS, max_abs_error=9.5367431640625e-06, mean_abs_error=1.272552026421181e-06, cosine=0.999999999999966
- decoder.up_blocks.1.upsamplers.0: status=PASS, max_abs_error=6.103515625e-05, mean_abs_error=4.61722751765592e-06, cosine=0.9999999999999603
- decoder.up_blocks.1 final: status=PASS, max_abs_error=6.103515625e-05, mean_abs_error=4.61722751765592e-06, cosine=0.9999999999999603
- decoder.up_blocks.1 synthetic: status=PASS, max_abs_error=0.00063323974609375, mean_abs_error=9.810523469511168e-05, cosine=0.9999999999345668
- decoder_upblock0_upblock1: status=PASS, max_abs_error=5.030632019042969e-05, mean_abs_error=6.402784205050693e-06, cosine=0.9999999999998331
- decoder_entry_midblock_upblocks01: status=PASS, max_abs_error=9.1552734375e-05, mean_abs_error=8.123663750581045e-06, cosine=0.9999999999997227
- quant_to_decoder_upblock1: status=PASS, max_abs_error=6.866455078125e-05, mean_abs_error=7.4422425981879314e-06, cosine=0.9999999999998994
- encoder_to_decoder_upblock1: status=PASS, max_abs_error=8.0108642578125e-05, mean_abs_error=8.99469925741414e-06, cosine=0.9999999999998507
- decoder.up_blocks.2.resnets.0: status=PASS, max_abs_error=0.0001068115234375, mean_abs_error=2.3815127860871144e-06, cosine=0.9999999999999588
- decoder.up_blocks.2.resnets.1: status=PASS, max_abs_error=0.0001220703125, mean_abs_error=2.8715757594000024e-06, cosine=0.999999999999952
- decoder.up_blocks.2.resnets.2: status=PASS, max_abs_error=0.0001220703125, mean_abs_error=3.3766652336453262e-06, cosine=0.9999999999999412
- decoder.up_blocks.2.upsamplers.0: status=PASS, max_abs_error=0.0001678466796875, mean_abs_error=9.56894639259076e-06, cosine=0.9999999999999332
- decoder.up_blocks.2 final: status=PASS, max_abs_error=0.0001678466796875, mean_abs_error=9.56894639259076e-06, cosine=0.9999999999999332
- decoder.up_blocks.2 synthetic: status=PASS, max_abs_error=0.00037860870361328125, mean_abs_error=4.2609574690821717e-05, cosine=0.9999999999896408
- decoder_upblocks012: status=PASS, max_abs_error=0.00014591217041015625, mean_abs_error=1.2784620821548742e-05, cosine=0.9999999999996184
- decoder_entry_midblock_upblocks012: status=PASS, max_abs_error=0.00037384033203125, mean_abs_error=1.9698307040627583e-05, cosine=0.999999999998697
- quant_to_decoder_upblock2: status=PASS, max_abs_error=0.0001678466796875, mean_abs_error=1.2708558537077863e-05, cosine=0.9999999999998848
- encoder_to_decoder_upblock2: status=PASS, max_abs_error=0.0001678466796875, mean_abs_error=1.3979698969102117e-05, cosine=0.9999999999998602
- decoder.up_blocks.3.resnets.0: status=PASS, max_abs_error=7.62939453125e-05, mean_abs_error=3.7603155789156517e-06, cosine=0.9999999999999888
- decoder.up_blocks.3.resnets.1: status=PASS, max_abs_error=7.62939453125e-05, mean_abs_error=4.342165212278815e-06, cosine=0.999999999999987
- decoder.up_blocks.3.resnets.2: status=PASS, max_abs_error=9.1552734375e-05, mean_abs_error=5.020616782758225e-06, cosine=0.9999999999999858
- decoder.up_blocks.3.upsamplers.0: status=NOT_APPLICABLE, no metric available
- decoder.up_blocks.3 final: status=PASS, max_abs_error=9.1552734375e-05, mean_abs_error=5.020616782758225e-06, cosine=0.9999999999999858
- decoder.up_blocks.3 synthetic: status=PASS, max_abs_error=0.0002288818359375, mean_abs_error=2.627187774351114e-05, cosine=0.9999999999989243
- decoder_upblocks0123: status=PASS, max_abs_error=0.00048828125, mean_abs_error=1.7898237473445988e-05, cosine=0.999999999999662
- decoder_entry_midblock_upblocks0123: status=PASS, max_abs_error=0.00128173828125, mean_abs_error=4.334792453164482e-05, cosine=0.9999999999963466
- quant_to_decoder_upblock3: status=PASS, max_abs_error=0.000152587890625, mean_abs_error=1.0762434413891242e-05, cosine=0.9999999999999325
- encoder_to_decoder_upblock3: status=PASS, max_abs_error=0.0001983642578125, mean_abs_error=1.1409385308525088e-05, cosine=0.9999999999999254
- decoder.tail.conv_norm_out: status=PASS, max_abs_error=9.298324584960938e-06, mean_abs_error=1.8709834789104596e-06, cosine=0.9999999999937493
- decoder.tail.conv_act: status=PASS, max_abs_error=1.0251998901367188e-05, mean_abs_error=9.220940512463804e-07, cosine=0.9999999999915102
- decoder.tail.conv_out: status=PASS, max_abs_error=1.7285346984863281e-06, mean_abs_error=6.843505010086423e-07, cosine=0.9999999999987242
- decoder.tail final: status=PASS, max_abs_error=1.7285346984863281e-06, mean_abs_error=6.843505010086423e-07, cosine=0.9999999999987242
- decoder.tail synthetic: status=PASS, max_abs_error=1.7285346984863281e-06, mean_abs_error=6.843505010086423e-07, cosine=0.9999999999987242
- decoder_upblocks0123_to_tail: status=PASS, max_abs_error=3.2782554626464844e-06, mean_abs_error=2.140653426370894e-07, cosine=0.9999999999995494
- decoder_entry_midblock_upblocks0123_tail: status=PASS, max_abs_error=2.384185791015625e-06, mean_abs_error=2.509790040979472e-07, cosine=0.9999999999993984
- quant_to_decoder_tail: status=PASS, max_abs_error=7.152557373046875e-07, mean_abs_error=1.2824746894087488e-07, cosine=0.9999999999999858
- encoder_to_decoder_tail: status=PASS, max_abs_error=7.152557373046875e-07, mean_abs_error=1.435397128564849e-07, cosine=0.9999999999999828
- LoRA merge: status=PASS, max_abs_error=1.862645149230957e-09, mean_abs_error=6.026704074062965e-11, cosine=None
- LoRA pair count: `290` from `580` LoRA keys

## Current TimeAware VAE Encoder Port Progress

| Stage | Status |
|---|---|
| `conv_in` | PASS |
| `down_blocks.0` | PASS |
| `down_blocks.1` | PASS |
| `down_blocks.2` | PASS |
| `down_blocks.3.resnets.0` | PASS |
| `down_blocks.3.resnets.1` | PASS |
| `down_blocks.3.downsampler0` | NOT_APPLICABLE |
| `down_blocks.3` | PASS |
| `stage0123` | PASS |
| `mid_block.resnets.0` | PASS |
| `mid_block.attentions.0` | PASS |
| `mid_block.resnets.1` | PASS |
| `mid_block` | PASS |
| `stage0123_mid` | PASS |
| `encoder.conv_norm_out` | PASS |
| `encoder.conv_act` | PASS |
| `encoder.conv_out` | PASS |
| `encoder tail` | PASS |
| `quant_conv` | PASS |
| `stage0123_mid_tail` | PASS |
| `stage0123_mid_tail_quant` | PASS |
| `decoder.conv_in` | PASS |
| `decoder entry` | PASS |
| `decoder.mid_block` | PASS |
| `decoder.up_blocks.0.resnets.0` | PASS |
| `decoder.up_blocks.0.resnets.1` | PASS |
| `decoder.up_blocks.0.resnets.2` | PASS |
| `decoder.up_blocks.0.upsamplers.0` | PASS |
| `decoder.up_blocks.0` | PASS |
| `decoder midblock + upblock0` | PASS |
| `encoder_to_decoder_upblock0` | PASS |
| `decoder.up_blocks.1.resnets.0` | PASS |
| `decoder.up_blocks.1.resnets.1` | PASS |
| `decoder.up_blocks.1.resnets.2` | PASS |
| `decoder.up_blocks.1.upsamplers.0` | PASS |
| `decoder.up_blocks.1` | PASS |
| `decoder upblock0 + upblock1` | PASS |
| `encoder_to_decoder_upblock1` | PASS |
| `decoder.up_blocks.2.resnets.0` | PASS |
| `decoder.up_blocks.2.resnets.1` | PASS |
| `decoder.up_blocks.2.resnets.2` | PASS |
| `decoder.up_blocks.2.upsamplers.0` | PASS |
| `decoder.up_blocks.2` | PASS |
| `decoder upblocks0-2` | PASS |
| `encoder_to_decoder_upblock2` | PASS |
| `decoder.up_blocks.3.resnets.0` | PASS |
| `decoder.up_blocks.3.resnets.1` | PASS |
| `decoder.up_blocks.3.resnets.2` | PASS |
| `decoder.up_blocks.3.upsamplers.0` | NOT_APPLICABLE |
| `decoder.up_blocks.3` | PASS |
| `decoder upblocks0-3` | PASS |
| `encoder_to_decoder_upblock3` | PASS |
| `decoder.conv_norm_out` | PASS |
| `decoder.conv_act` | PASS |
| `decoder.conv_out` | PASS |
| `decoder tail` | PASS |
| `decoder upblocks0-3 -> tail` | PASS |
| `quant -> decoder tail` | PASS |
| `encoder_to_decoder_tail` | PASS |

## Current TimeAware VAE Decoder Port Progress

| Stage | Status |
|---|---|
| `posterior mode + post_quant_conv + decoder.conv_in` | PASS |
| `decoder.mid_block` | PASS |
| `decoder.up_blocks.0` | PASS |
| `decoder.mid_block -> decoder.up_blocks.0` | PASS |
| `encoder -> quant -> decoder.up_blocks.0` | PASS |
| `decoder.up_blocks.1` | PASS |
| `decoder.up_blocks.0 -> decoder.up_blocks.1` | PASS |
| `encoder -> quant -> decoder.up_blocks.1` | PASS |
| `decoder.up_blocks.2` | PASS |
| `decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2` | PASS |
| `decoder entry -> decoder.mid_block -> decoder.up_blocks.2` | PASS |
| `quant -> decoder.up_blocks.2` | PASS |
| `encoder -> quant -> decoder.up_blocks.2` | PASS |
| `decoder.up_blocks.3` | PASS |
| `decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3` | PASS |
| `decoder entry -> decoder.mid_block -> decoder.up_blocks.3` | PASS |
| `quant -> decoder.up_blocks.3` | PASS |
| `encoder -> quant -> decoder.up_blocks.3` | PASS |
| `decoder.conv_norm_out` | PASS |
| `decoder.conv_act` | PASS |
| `decoder.conv_out` | PASS |
| `decoder tail` | PASS |
| `decoder upblocks0-3 -> decoder tail` | PASS |
| `decoder entry -> mid_block -> upblocks0-3 -> tail` | PASS |
| `quant -> decoder tail` | PASS |
| `encoder -> quant -> decoder tail` | PASS |

## Deterministic Decoder Stack Status

- The TimeAware VAE deterministic decoder stack is now aligned through `post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder tail`.
- The deterministic encoder-to-decoder-tail path is also aligned through posterior mode / mean; stochastic latent sampling is not implemented.
- This is module-level deterministic alignment only, not full VAE API completion and not full TADSR inference.

## Still Missing / Not Claimed

- Full TimeAware VAE stochastic sampling/API integration is not complete.
- Full UNet forward is not ported.
- Full LoRA runtime integration is not complete; current validation covers the merge formula for one pair.
- `jittor_tadsr_full.tadsr_full --full-inference` must remain NotImplemented until UNet and LoRA runtime integration pass.

## Recommended Next Module

Continue with full UNet audit/export/port after deterministic TimeAware VAE decoder stack alignment.
