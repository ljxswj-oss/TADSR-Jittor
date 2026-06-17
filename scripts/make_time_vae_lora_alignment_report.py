#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

OUT = Path('experiments/full_repro/time_vae_alignment')
LORA = Path('experiments/full_repro/lora_alignment')


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {'status': 'FAIL', 'error': repr(exc)}


def status(path: Path, default='MISSING'):
    data = load_json(path)
    return data.get('status', default) if data else default


def metric_line(label, data):
    m = data.get('metrics', {})
    if not m:
        return f'- {label}: status={data.get("status", "MISSING")}, no metric available'
    return f"- {label}: status={data.get('status')}, max_abs_error={m.get('max_abs_error')}, mean_abs_error={m.get('mean_abs_error')}, cosine={m.get('cosine_similarity')}"


def add_downblock_audit(md, idx, audit):
    md += ['', f'## DownBlock{idx} Audit Summary']
    if not audit:
        md.append('- no audit available')
        return
    md.append(f"- resnet_count: `{audit.get('resnet_count')}`")
    md.append(f"- has_downsampler: `{audit.get('has_downsampler')}`")
    md.append(f"- channel_change_detected: `{audit.get('channel_change_detected')}`")
    for item in audit.get('resnet_info', []):
        md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, time_emb_proj={item.get('time_emb_proj_weight_shape')}, conv_shortcut={item.get('conv_shortcut_weight_shape')}, nin_shortcut={item.get('nin_shortcut_weight_shape')}")
    if audit.get('downsampler_info'):
        info = audit['downsampler_info']
        md.append(f"- downsampler: `{info.get('class_module')}.{info.get('class_name')}`, padding={info.get('padding')}, stride={info.get('conv_stride')}, kernel={info.get('conv_kernel_size')}")
        md.append(f"- padding rule: {info.get('asymmetric_padding_note')}")
    else:
        md.append('- downsampler: `NOT_APPLICABLE`')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audits = {i: load_json(OUT / f'downblock{i}_audit.json') for i in [0, 1, 2, 3]}
    mid_audit = load_json(OUT / 'midblock_audit.json')
    tail_audit = load_json(OUT / 'encoder_tail_audit.json')
    decoder_upblocks_audit = load_json(OUT / 'audit_time_vae_decoder_upblocks.json')
    decoder_upblock0_audit = load_json(OUT / 'audit_time_vae_decoder_upblock0.json')
    decoder_upblock1_audit = load_json(OUT / 'audit_time_vae_decoder_upblock1.json')
    decoder_upblock2_audit = load_json(OUT / 'audit_time_vae_decoder_upblock2.json')
    decoder_upblock3_audit = load_json(OUT / 'audit_time_vae_decoder_upblock3.json')
    decoder_tail_audit = load_json(OUT / 'audit_time_vae_decoder_tail.json')
    conv_in = load_json(OUT / 'jittor_conv_in_alignment.json')
    first = load_json(OUT / 'jittor_first_block_alignment.json')
    resnet1 = load_json(OUT / 'jittor_resnet1_alignment.json')
    downsampler = load_json(OUT / 'jittor_downsampler_alignment.json')
    downblock0 = load_json(OUT / 'jittor_downblock0_alignment.json')
    stage_data = {
        'down_blocks.1.resnets.0': load_json(OUT / 'jittor_downblock1_resnet0_alignment.json'),
        'down_blocks.1.resnets.1': load_json(OUT / 'jittor_downblock1_resnet1_alignment.json'),
        'down_blocks.1.downsampler0': load_json(OUT / 'jittor_downblock1_downsampler_alignment.json'),
        'down_blocks.1 final': load_json(OUT / 'jittor_downblock1_alignment.json'),
        'encoder stage01': load_json(OUT / 'jittor_encoder_stage01_alignment.json'),
        'down_blocks.2.resnets.0': load_json(OUT / 'jittor_downblock2_resnet0_alignment.json'),
        'down_blocks.2.resnets.1': load_json(OUT / 'jittor_downblock2_resnet1_alignment.json'),
        'down_blocks.2.downsampler0': load_json(OUT / 'jittor_downblock2_downsampler_alignment.json'),
        'down_blocks.2 final': load_json(OUT / 'jittor_downblock2_alignment.json'),
        'encoder stage012': load_json(OUT / 'jittor_encoder_stage012_alignment.json'),
        'down_blocks.3.resnets.0': load_json(OUT / 'jittor_downblock3_resnet0_alignment.json'),
        'down_blocks.3.resnets.1': load_json(OUT / 'jittor_downblock3_resnet1_alignment.json'),
        'down_blocks.3.downsampler0': load_json(OUT / 'jittor_downblock3_downsampler_alignment.json'),
        'down_blocks.3 final': load_json(OUT / 'jittor_downblock3_alignment.json'),
        'encoder stage0123': load_json(OUT / 'jittor_encoder_stage0123_alignment.json'),
        'mid_block.resnets.0': load_json(OUT / 'jittor_midblock_resnet0_alignment.json'),
        'mid_block.attentions.0': load_json(OUT / 'jittor_midblock_attention_alignment.json'),
        'mid_block.resnets.1': load_json(OUT / 'jittor_midblock_resnet1_alignment.json'),
        'mid_block final': load_json(OUT / 'jittor_midblock_alignment.json'),
        'encoder stage0123_mid': load_json(OUT / 'jittor_encoder_stage0123_mid_alignment.json'),
        'encoder tail conv_norm_out': load_json(OUT / 'jittor_encoder_tail_norm_alignment.json'),
        'encoder tail conv_act': load_json(OUT / 'jittor_encoder_tail_act_alignment.json'),
        'encoder tail conv_out': load_json(OUT / 'jittor_encoder_tail_conv_out_alignment.json'),
        'encoder tail final': load_json(OUT / 'jittor_encoder_tail_alignment.json'),
        'quant_conv': load_json(OUT / 'jittor_quant_conv_alignment.json'),
        'encoder stage0123_mid_tail': load_json(OUT / 'jittor_encoder_stage0123_mid_tail_alignment.json'),
        'encoder stage0123_mid_tail_quant': load_json(OUT / 'jittor_encoder_stage0123_mid_tail_quant_alignment.json'),
        'moments split': load_json(OUT / 'jittor_moments_split_alignment.json'),
        'posterior mode': load_json(OUT / 'jittor_posterior_mode_alignment.json'),
        'post_quant_conv': load_json(OUT / 'jittor_post_quant_conv_alignment.json'),
        'decoder.conv_in': load_json(OUT / 'jittor_decoder_conv_in_alignment.json'),
        'decoder entry': load_json(OUT / 'jittor_decoder_entry_alignment.json'),
        'quant_to_decoder_entry': load_json(OUT / 'jittor_quant_to_decoder_entry_alignment.json'),
        'encoder_to_decoder_entry': load_json(OUT / 'jittor_encoder_to_decoder_entry_alignment.json'),
        'decoder.mid_block.resnets.0': load_json(OUT / 'jittor_decoder_midblock_resnet0_alignment.json'),
        'decoder.mid_block.attentions.0': load_json(OUT / 'jittor_decoder_midblock_attention_alignment.json'),
        'decoder.mid_block.resnets.1': load_json(OUT / 'jittor_decoder_midblock_resnet1_alignment.json'),
        'decoder.mid_block final': load_json(OUT / 'jittor_decoder_midblock_alignment.json'),
        'decoder.mid_block synthetic': load_json(OUT / 'jittor_decoder_midblock_synthetic_alignment.json'),
        'decoder_entry_midblock': load_json(OUT / 'jittor_decoder_entry_midblock_alignment.json'),
        'quant_to_decoder_midblock': load_json(OUT / 'jittor_quant_to_decoder_midblock_alignment.json'),
        'encoder_to_decoder_midblock': load_json(OUT / 'jittor_encoder_to_decoder_midblock_alignment.json'),
        'decoder.up_blocks.0.resnets.0': load_json(OUT / 'jittor_decoder_upblock0_resnet0_alignment.json'),
        'decoder.up_blocks.0.resnets.1': load_json(OUT / 'jittor_decoder_upblock0_resnet1_alignment.json'),
        'decoder.up_blocks.0.resnets.2': load_json(OUT / 'jittor_decoder_upblock0_resnet2_alignment.json'),
        'decoder.up_blocks.0.upsamplers.0': load_json(OUT / 'jittor_decoder_upblock0_upsampler0_alignment.json'),
        'decoder.up_blocks.0 final': load_json(OUT / 'jittor_decoder_upblock0_alignment.json'),
        'decoder.up_blocks.0 synthetic': load_json(OUT / 'jittor_decoder_upblock0_synthetic_alignment.json'),
        'decoder_midblock_upblock0': load_json(OUT / 'jittor_decoder_midblock_upblock0_alignment.json'),
        'decoder_entry_midblock_upblock0': load_json(OUT / 'jittor_decoder_entry_midblock_upblock0_alignment.json'),
        'quant_to_decoder_upblock0': load_json(OUT / 'jittor_quant_to_decoder_upblock0_alignment.json'),
        'encoder_to_decoder_upblock0': load_json(OUT / 'jittor_encoder_to_decoder_upblock0_alignment.json'),
        'decoder.up_blocks.1.resnets.0': load_json(OUT / 'jittor_decoder_upblock1_resnet0_alignment.json'),
        'decoder.up_blocks.1.resnets.1': load_json(OUT / 'jittor_decoder_upblock1_resnet1_alignment.json'),
        'decoder.up_blocks.1.resnets.2': load_json(OUT / 'jittor_decoder_upblock1_resnet2_alignment.json'),
        'decoder.up_blocks.1.upsamplers.0': load_json(OUT / 'jittor_decoder_upblock1_upsampler0_alignment.json'),
        'decoder.up_blocks.1 final': load_json(OUT / 'jittor_decoder_upblock1_alignment.json'),
        'decoder.up_blocks.1 synthetic': load_json(OUT / 'jittor_decoder_upblock1_synthetic_alignment.json'),
        'decoder_upblock0_upblock1': load_json(OUT / 'jittor_decoder_upblock0_upblock1_alignment.json'),
        'decoder_entry_midblock_upblocks01': load_json(OUT / 'jittor_decoder_entry_midblock_upblocks01_alignment.json'),
        'quant_to_decoder_upblock1': load_json(OUT / 'jittor_quant_to_decoder_upblock1_alignment.json'),
        'encoder_to_decoder_upblock1': load_json(OUT / 'jittor_encoder_to_decoder_upblock1_alignment.json'),
        'decoder.up_blocks.2.resnets.0': load_json(OUT / 'jittor_decoder_upblock2_resnet0_alignment.json'),
        'decoder.up_blocks.2.resnets.1': load_json(OUT / 'jittor_decoder_upblock2_resnet1_alignment.json'),
        'decoder.up_blocks.2.resnets.2': load_json(OUT / 'jittor_decoder_upblock2_resnet2_alignment.json'),
        'decoder.up_blocks.2.upsamplers.0': load_json(OUT / 'jittor_decoder_upblock2_upsampler0_alignment.json'),
        'decoder.up_blocks.2 final': load_json(OUT / 'jittor_decoder_upblock2_alignment.json'),
        'decoder.up_blocks.2 synthetic': load_json(OUT / 'jittor_decoder_upblock2_synthetic_alignment.json'),
        'decoder_upblocks012': load_json(OUT / 'jittor_decoder_upblocks012_alignment.json'),
        'decoder_entry_midblock_upblocks012': load_json(OUT / 'jittor_decoder_entry_midblock_upblocks012_alignment.json'),
        'quant_to_decoder_upblock2': load_json(OUT / 'jittor_quant_to_decoder_upblock2_alignment.json'),
        'encoder_to_decoder_upblock2': load_json(OUT / 'jittor_encoder_to_decoder_upblock2_alignment.json'),
        'decoder.up_blocks.3.resnets.0': load_json(OUT / 'jittor_decoder_upblock3_resnet0_alignment.json'),
        'decoder.up_blocks.3.resnets.1': load_json(OUT / 'jittor_decoder_upblock3_resnet1_alignment.json'),
        'decoder.up_blocks.3.resnets.2': load_json(OUT / 'jittor_decoder_upblock3_resnet2_alignment.json'),
        'decoder.up_blocks.3.upsamplers.0': load_json(OUT / 'jittor_decoder_upblock3_upsampler0_alignment.json'),
        'decoder.up_blocks.3 final': load_json(OUT / 'jittor_decoder_upblock3_alignment.json'),
        'decoder.up_blocks.3 synthetic': load_json(OUT / 'jittor_decoder_upblock3_synthetic_alignment.json'),
        'decoder_upblocks0123': load_json(OUT / 'jittor_decoder_upblocks0123_alignment.json'),
        'decoder_entry_midblock_upblocks0123': load_json(OUT / 'jittor_decoder_entry_midblock_upblocks0123_alignment.json'),
        'quant_to_decoder_upblock3': load_json(OUT / 'jittor_quant_to_decoder_upblock3_alignment.json'),
        'encoder_to_decoder_upblock3': load_json(OUT / 'jittor_encoder_to_decoder_upblock3_alignment.json'),
        'decoder.tail.conv_norm_out': load_json(OUT / 'jittor_decoder_tail_norm_alignment.json'),
        'decoder.tail.conv_act': load_json(OUT / 'jittor_decoder_tail_act_alignment.json'),
        'decoder.tail.conv_out': load_json(OUT / 'jittor_decoder_tail_conv_out_alignment.json'),
        'decoder.tail final': load_json(OUT / 'jittor_decoder_tail_alignment.json'),
        'decoder.tail synthetic': load_json(OUT / 'jittor_decoder_tail_synthetic_alignment.json'),
        'decoder_upblocks0123_to_tail': load_json(OUT / 'jittor_decoder_upblocks0123_to_tail_alignment.json'),
        'decoder_entry_midblock_upblocks0123_tail': load_json(OUT / 'jittor_decoder_entry_midblock_upblocks0123_tail_alignment.json'),
        'quant_to_decoder_tail': load_json(OUT / 'jittor_quant_to_decoder_tail_alignment.json'),
        'encoder_to_decoder_tail': load_json(OUT / 'jittor_encoder_to_decoder_tail_alignment.json'),
    }
    pairs = load_json(LORA / 'lora_pairs.json')
    merge = load_json(LORA / 'lora_merge_validation.json')

    md = ['# TimeAware VAE and LoRA Alignment Progress', '', '| Item | Status | Evidence |', '|---|---|---|']
    rows = [
        ('Official TimeAware VAE audit', status(OUT / 'official_time_vae_audit.json'), 'official_time_vae_module_tree.md / official_time_vae_state_keys.md'),
        ('PyTorch TimeAware VAE oracle tensors', status(OUT / 'oracle_tensors/time_vae_hook_metadata.json'), 'oracle_tensors/time_vae_oracle_inputs.npz and outputs.npz'),
    ]
    for i in [0, 1, 2, 3]:
        rows += [
            (f'DownBlock{i} structure audit', status(OUT / f'downblock{i}_audit.json'), f'downblock{i}_audit.md / downblock{i}_key_mapping.md'),
            (f'DownBlock{i} PyTorch oracle tensors', status(OUT / f'oracle_tensors_downblock{i}/time_vae_downblock{i}_hook_metadata.json'), f'oracle_tensors_downblock{i}/*.npz'),
        ]
    rows += [
        ('MidBlock structure audit', status(OUT / 'midblock_audit.json'), 'midblock_audit.md / midblock_key_mapping.md'),
        ('MidBlock PyTorch oracle tensors', status(OUT / 'oracle_tensors_midblock/time_vae_midblock_hook_metadata.json'), 'oracle_tensors_midblock/*.npz'),
        ('Encoder tail structure audit', status(OUT / 'encoder_tail_audit.json'), 'encoder_tail_audit.md / encoder_tail_key_mapping.md'),
        ('Encoder tail + quant_conv PyTorch oracle tensors', status(OUT / 'oracle_tensors_encoder_tail/time_vae_encoder_tail_hook_metadata.json'), 'oracle_tensors_encoder_tail/*.npz'),
        ('Decoder up_blocks structure audit', status(OUT / 'audit_time_vae_decoder_upblocks.json'), 'audit_time_vae_decoder_upblocks.json / audit_time_vae_decoder_upblock0.json'),
        ('Decoder up_blocks.0 PyTorch oracle tensors', status(OUT / 'oracle_tensors_decoder_upblock0/decoder_upblock0_oracle_metadata.json'), 'oracle_tensors_decoder_upblock0/*.npz and *.npy'),
        ('Decoder up_blocks.1 PyTorch oracle tensors', status(OUT / 'oracle_tensors_decoder_upblock1/decoder_upblock1_oracle_metadata.json'), 'oracle_tensors_decoder_upblock1/*.npz and *.npy'),
        ('Decoder up_blocks.2 PyTorch oracle tensors', status(OUT / 'oracle_tensors_decoder_upblock2/decoder_upblock2_oracle_metadata.json'), 'oracle_tensors_decoder_upblock2/*.npz and *.npy generated locally; large tensors are not intended for git commit'),
        ('Decoder up_blocks.3 PyTorch oracle tensors', status(OUT / 'oracle_tensors_decoder_upblock3/decoder_upblock3_oracle_metadata.json'), 'oracle_tensors_decoder_upblock3/*.npz and *.npy generated locally; large tensors are not intended for git commit'),
        ('Decoder tail structure audit', status(OUT / 'audit_time_vae_decoder_tail.json'), 'audit_time_vae_decoder_tail.json / audit_time_vae_decoder_tail.txt'),
        ('Decoder tail PyTorch oracle tensors', status(OUT / 'oracle_tensors_decoder_tail/decoder_tail_oracle_metadata.json'), 'oracle_tensors_decoder_tail/*.npz and *.npy generated locally; large tensors are not intended for git commit'),
    ]
    rows += [
        ('Jittor Time-VAE conv_in alignment', conv_in.get('status', 'MISSING'), 'jittor_conv_in_alignment.json'),
        ('Jittor Time-VAE down_blocks.0 resnets.0 alignment', first.get('status', 'MISSING'), 'jittor_first_block_alignment.json'),
        ('Jittor Time-VAE down_blocks.0 resnets.1 alignment', resnet1.get('status', 'MISSING'), 'jittor_resnet1_alignment.json'),
        ('Jittor Time-VAE down_blocks.0 downsampler alignment', downsampler.get('status', 'MISSING'), 'jittor_downsampler_alignment.json'),
        ('Jittor Time-VAE down_blocks.0 full alignment', downblock0.get('status', 'MISSING'), 'jittor_downblock0_alignment.json'),
    ]
    for name, data in stage_data.items():
        rows.append((f'Jittor Time-VAE {name} alignment', data.get('status', 'MISSING'), data.get('target', name)))
    rows += [
        ('LoRA pair analysis', pairs.get('status', 'MISSING'), 'experiments/full_repro/lora_alignment/lora_pairs.json'),
        ('LoRA merge formula validation', merge.get('status', 'MISSING'), 'experiments/full_repro/lora_alignment/lora_merge_validation.json'),
    ]
    for name, st, ev in rows:
        md.append(f'| {name} | {st} | `{ev}` |')

    for i in [0, 1, 2, 3]:
        add_downblock_audit(md, i, audits[i])

    md += ['', '## MidBlock Audit Summary']
    if mid_audit:
        md.append(f"- class: `{mid_audit.get('mid_block_module')}.{mid_audit.get('mid_block_class')}`")
        md.append(f"- resnet_count: `{mid_audit.get('resnet_count')}`")
        md.append(f"- attention_count: `{mid_audit.get('attention_count')}`")
        for item in mid_audit.get('resnet_info', []):
            md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, time_emb_proj={item.get('time_emb_proj_weight_shape')}")
        for item in mid_audit.get('attention_info', []):
            md.append(f"- `{item.get('module_name')}`: heads={item.get('heads')}, head_dim={item.get('head_dim')}, q={item.get('to_q_weight_shape')}, out={item.get('to_out_0_weight_shape')}, residual={item.get('residual_connection')}")
    else:
        md.append('- no mid_block audit available')

    md += ['', '## Encoder Tail + QuantConv Audit Summary']
    if tail_audit:
        norm = tail_audit.get('conv_norm_out', {})
        conv_out = tail_audit.get('conv_out', {})
        q = tail_audit.get('quant_conv', {})
        md.append(f"- `encoder.conv_norm_out`: groups={norm.get('num_groups')}, channels={norm.get('num_channels')}, eps={norm.get('eps')}")
        md.append(f"- `encoder.conv_act`: {tail_audit.get('conv_act', {}).get('class_name')}")
        md.append(f"- `encoder.conv_out`: weight={conv_out.get('weight_shape')}, padding={conv_out.get('padding')}, channels={conv_out.get('in_channels')}->{conv_out.get('out_channels')}")
        md.append(f"- `quant_conv`: weight={q.get('weight_shape')}, padding={q.get('padding')}, channels={q.get('in_channels')}->{q.get('out_channels')}")
        md.append(f"- `post_quant_conv` exists but skipped this stage: {tail_audit.get('post_quant_conv_exists')}")
    else:
        md.append('- no encoder tail audit available')

    md += ['', '## Decoder UpBlock0 Audit Summary']
    if decoder_upblock0_audit:
        focus = decoder_upblock0_audit.get('upblock0', decoder_upblock0_audit)
        md.append(f"- class: `{focus.get('module')}.{focus.get('class')}`")
        md.append(f"- input_shape: `{focus.get('input_shape')}`, output_shape: `{focus.get('output_shape')}`")
        md.append(f"- resnet_count: `{focus.get('resnet_count')}`, upsampler_count: `{focus.get('upsampler_count')}`, attention_count: `{focus.get('attention_count')}`")
        md.append(f"- module_order: `{focus.get('module_order')}`")
        for item in focus.get('resnet_info', []):
            md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, shortcut={item.get('conv_shortcut_exists')}")
        for item in focus.get('upsampler_info', []):
            conv_info = item.get('conv', {})
            md.append(f"- `{item.get('module_name')}`: nearest x2 then conv weight={conv_info.get('weight_shape')}, padding={conv_info.get('padding')}")
    elif decoder_upblocks_audit:
        block = (decoder_upblocks_audit.get('up_blocks') or [{}])[0]
        md.append(f"- class: `{block.get('class')}`")
        md.append(f"- input_shape: `{block.get('input_shape')}`, output_shape: `{block.get('output_shape')}`")
        md.append(f"- module_order: `{block.get('module_order')}`")
    else:
        md.append('- no decoder upblock0 audit available')

    md += ['', '## Decoder UpBlock1 Audit Summary']
    if decoder_upblock1_audit:
        focus = decoder_upblock1_audit.get('upblock1', decoder_upblock1_audit)
        md.append(f"- class: `{focus.get('module')}.{focus.get('class')}`")
        md.append(f"- input_shape: `{focus.get('input_shape')}`, output_shape: `{focus.get('output_shape')}`")
        md.append(f"- resnet_count: `{focus.get('resnet_count')}`, upsampler_count: `{focus.get('upsampler_count')}`, attention_count: `{focus.get('attention_count')}`")
        md.append(f"- module_order: `{focus.get('module_order')}`")
        for item in focus.get('resnet_info', []):
            md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, shortcut={item.get('conv_shortcut_exists')}")
        for item in focus.get('upsampler_info', []):
            conv_info = item.get('conv', {})
            md.append(f"- `{item.get('module_name')}`: nearest x2 then conv weight={conv_info.get('weight_shape')}, padding={conv_info.get('padding')}")
    else:
        md.append('- no decoder upblock1 audit available')

    md += ['', '## Decoder UpBlock2 Audit Summary']
    if decoder_upblock2_audit:
        focus = decoder_upblock2_audit.get('upblock2', decoder_upblock2_audit)
        md.append(f"- class: `{focus.get('module')}.{focus.get('class')}`")
        md.append(f"- input_shape: `{focus.get('input_shape')}`, output_shape: `{focus.get('output_shape')}`")
        md.append(f"- resnet_count: `{focus.get('resnet_count')}`, upsampler_count: `{focus.get('upsampler_count')}`, attention_count: `{focus.get('attention_count')}`")
        md.append(f"- module_order: `{focus.get('module_order')}`")
        md.append('- key behavior: `resnets.0` changes channels from 512 to 256 and uses `conv_shortcut`; `upsamplers.0` upsamples 16x16 to 32x32.')
        for item in focus.get('resnet_info', []):
            md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, shortcut={item.get('conv_shortcut_exists')}, shortcut_weight={item.get('conv_shortcut_weight_shape')}")
        for item in focus.get('upsampler_info', []):
            conv_info = item.get('conv', {})
            md.append(f"- `{item.get('module_name')}`: nearest x2 then conv weight={conv_info.get('weight_shape')}, padding={conv_info.get('padding')}")
    else:
        md.append('- no decoder upblock2 audit available')

    md += ['', '## Decoder UpBlock3 Audit Summary']
    if decoder_upblock3_audit:
        focus = decoder_upblock3_audit.get('upblock3', decoder_upblock3_audit)
        md.append(f"- class: `{focus.get('module')}.{focus.get('class')}`")
        md.append(f"- input_shape: `{focus.get('input_shape')}`, output_shape: `{focus.get('output_shape')}`")
        md.append(f"- resnet_count: `{focus.get('resnet_count')}`, upsampler_count: `{focus.get('upsampler_count')}`, attention_count: `{focus.get('attention_count')}`")
        md.append(f"- module_order: `{focus.get('module_order')}`")
        md.append('- key behavior: `resnets.0` changes channels from 256 to 128 and uses `conv_shortcut`; there is no official upsampler in `decoder.up_blocks.3`, so upsampler0 is `NOT_APPLICABLE`.')
        for item in focus.get('resnet_info', []):
            md.append(f"- `{item.get('module_name')}`: conv1={item.get('conv1_weight_shape')}, conv2={item.get('conv2_weight_shape')}, shortcut={item.get('conv_shortcut_exists')}, shortcut_weight={item.get('conv_shortcut_weight_shape')}")
    else:
        md.append('- no decoder upblock3 audit available')

    md += ['', '## Decoder Tail Audit Summary']
    if decoder_tail_audit:
        norm = decoder_tail_audit.get('norm', {})
        act = decoder_tail_audit.get('activation', {})
        conv = decoder_tail_audit.get('conv_out', {})
        md.append(f"- module_order: `{decoder_tail_audit.get('decoder_tail_module_order')}`")
        md.append(f"- tail_input_shape: `{decoder_tail_audit.get('tail_input_shape')}`, tail_output_shape: `{decoder_tail_audit.get('tail_output_shape')}`")
        md.append(f"- `decoder.conv_norm_out`: class={norm.get('class')}, groups={norm.get('num_groups')}, channels={norm.get('num_channels')}, eps={norm.get('eps')}, affine={norm.get('affine')}")
        md.append(f"- `decoder.conv_act`: class={act.get('class')}, exact_operation={act.get('exact_operation')}")
        md.append(f"- `decoder.conv_out`: weight={conv.get('weight_shape')}, bias={conv.get('bias_shape')}, padding={conv.get('padding')}, stride={conv.get('stride')}, channels={conv.get('in_channels')}->{conv.get('out_channels')}")
        md.append(f"- scaling_factor_exists={decoder_tail_audit.get('scaling_factor_exists')}, scaling_factor_value={decoder_tail_audit.get('scaling_factor_value')}, applied_inside_decoder={decoder_tail_audit.get('scaling_factor_applied_inside_decoder')}")
        md.append(f"- output_clamp_or_tanh_inside_decoder={decoder_tail_audit.get('output_clamp_or_tanh_inside_decoder')}")
    else:
        md.append('- no decoder tail audit available')

    md += ['', '## Key Metrics']
    for label, data in [
        ('conv_in', conv_in),
        ('down_blocks.0.resnets.0', first),
        ('down_blocks.0.resnets.1', resnet1),
        ('down_blocks.0.downsampler0', downsampler),
        ('down_blocks.0 final', downblock0),
    ] + list(stage_data.items()) + [('LoRA merge', merge)]:
        md.append(metric_line(label, data))
    if pairs:
        md.append(f"- LoRA pair count: `{pairs.get('pair_count')}` from `{pairs.get('lora_key_count')}` LoRA keys")

    md += ['', '## Current TimeAware VAE Encoder Port Progress', '', '| Stage | Status |', '|---|---|']
    for label, data in [
        ('conv_in', conv_in),
        ('down_blocks.0', downblock0),
        ('down_blocks.1', stage_data['down_blocks.1 final']),
        ('down_blocks.2', stage_data['down_blocks.2 final']),
        ('down_blocks.3.resnets.0', stage_data['down_blocks.3.resnets.0']),
        ('down_blocks.3.resnets.1', stage_data['down_blocks.3.resnets.1']),
        ('down_blocks.3.downsampler0', stage_data['down_blocks.3.downsampler0']),
        ('down_blocks.3', stage_data['down_blocks.3 final']),
        ('stage0123', stage_data['encoder stage0123']),
        ('mid_block.resnets.0', stage_data['mid_block.resnets.0']),
        ('mid_block.attentions.0', stage_data['mid_block.attentions.0']),
        ('mid_block.resnets.1', stage_data['mid_block.resnets.1']),
        ('mid_block', stage_data['mid_block final']),
        ('stage0123_mid', stage_data['encoder stage0123_mid']),
        ('encoder.conv_norm_out', stage_data['encoder tail conv_norm_out']),
        ('encoder.conv_act', stage_data['encoder tail conv_act']),
        ('encoder.conv_out', stage_data['encoder tail conv_out']),
        ('encoder tail', stage_data['encoder tail final']),
        ('quant_conv', stage_data['quant_conv']),
        ('stage0123_mid_tail', stage_data['encoder stage0123_mid_tail']),
        ('stage0123_mid_tail_quant', stage_data['encoder stage0123_mid_tail_quant']),
        ('decoder.conv_in', stage_data['decoder.conv_in']),
        ('decoder entry', stage_data['decoder entry']),
        ('decoder.mid_block', stage_data['decoder.mid_block final']),
        ('decoder.up_blocks.0.resnets.0', stage_data['decoder.up_blocks.0.resnets.0']),
        ('decoder.up_blocks.0.resnets.1', stage_data['decoder.up_blocks.0.resnets.1']),
        ('decoder.up_blocks.0.resnets.2', stage_data['decoder.up_blocks.0.resnets.2']),
        ('decoder.up_blocks.0.upsamplers.0', stage_data['decoder.up_blocks.0.upsamplers.0']),
        ('decoder.up_blocks.0', stage_data['decoder.up_blocks.0 final']),
        ('decoder midblock + upblock0', stage_data['decoder_midblock_upblock0']),
        ('encoder_to_decoder_upblock0', stage_data['encoder_to_decoder_upblock0']),
        ('decoder.up_blocks.1.resnets.0', stage_data['decoder.up_blocks.1.resnets.0']),
        ('decoder.up_blocks.1.resnets.1', stage_data['decoder.up_blocks.1.resnets.1']),
        ('decoder.up_blocks.1.resnets.2', stage_data['decoder.up_blocks.1.resnets.2']),
        ('decoder.up_blocks.1.upsamplers.0', stage_data['decoder.up_blocks.1.upsamplers.0']),
        ('decoder.up_blocks.1', stage_data['decoder.up_blocks.1 final']),
        ('decoder upblock0 + upblock1', stage_data['decoder_upblock0_upblock1']),
        ('encoder_to_decoder_upblock1', stage_data['encoder_to_decoder_upblock1']),
        ('decoder.up_blocks.2.resnets.0', stage_data['decoder.up_blocks.2.resnets.0']),
        ('decoder.up_blocks.2.resnets.1', stage_data['decoder.up_blocks.2.resnets.1']),
        ('decoder.up_blocks.2.resnets.2', stage_data['decoder.up_blocks.2.resnets.2']),
        ('decoder.up_blocks.2.upsamplers.0', stage_data['decoder.up_blocks.2.upsamplers.0']),
        ('decoder.up_blocks.2', stage_data['decoder.up_blocks.2 final']),
        ('decoder upblocks0-2', stage_data['decoder_upblocks012']),
        ('encoder_to_decoder_upblock2', stage_data['encoder_to_decoder_upblock2']),
        ('decoder.up_blocks.3.resnets.0', stage_data['decoder.up_blocks.3.resnets.0']),
        ('decoder.up_blocks.3.resnets.1', stage_data['decoder.up_blocks.3.resnets.1']),
        ('decoder.up_blocks.3.resnets.2', stage_data['decoder.up_blocks.3.resnets.2']),
        ('decoder.up_blocks.3.upsamplers.0', stage_data['decoder.up_blocks.3.upsamplers.0']),
        ('decoder.up_blocks.3', stage_data['decoder.up_blocks.3 final']),
        ('decoder upblocks0-3', stage_data['decoder_upblocks0123']),
        ('encoder_to_decoder_upblock3', stage_data['encoder_to_decoder_upblock3']),
        ('decoder.conv_norm_out', stage_data['decoder.tail.conv_norm_out']),
        ('decoder.conv_act', stage_data['decoder.tail.conv_act']),
        ('decoder.conv_out', stage_data['decoder.tail.conv_out']),
        ('decoder tail', stage_data['decoder.tail final']),
        ('decoder upblocks0-3 -> tail', stage_data['decoder_upblocks0123_to_tail']),
        ('quant -> decoder tail', stage_data['quant_to_decoder_tail']),
        ('encoder_to_decoder_tail', stage_data['encoder_to_decoder_tail']),
    ]:
        md.append(f"| `{label}` | {data.get('status', 'MISSING')} |")

    md += ['', '## Current TimeAware VAE Decoder Port Progress', '', '| Stage | Status |', '|---|---|']
    for label, data in [
        ('posterior mode + post_quant_conv + decoder.conv_in', stage_data['decoder entry']),
        ('decoder.mid_block', stage_data['decoder.mid_block final']),
        ('decoder.up_blocks.0', stage_data['decoder.up_blocks.0 final']),
        ('decoder.mid_block -> decoder.up_blocks.0', stage_data['decoder_midblock_upblock0']),
        ('encoder -> quant -> decoder.up_blocks.0', stage_data['encoder_to_decoder_upblock0']),
        ('decoder.up_blocks.1', stage_data['decoder.up_blocks.1 final']),
        ('decoder.up_blocks.0 -> decoder.up_blocks.1', stage_data['decoder_upblock0_upblock1']),
        ('encoder -> quant -> decoder.up_blocks.1', stage_data['encoder_to_decoder_upblock1']),
        ('decoder.up_blocks.2', stage_data['decoder.up_blocks.2 final']),
        ('decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2', stage_data['decoder_upblocks012']),
        ('decoder entry -> decoder.mid_block -> decoder.up_blocks.2', stage_data['decoder_entry_midblock_upblocks012']),
        ('quant -> decoder.up_blocks.2', stage_data['quant_to_decoder_upblock2']),
        ('encoder -> quant -> decoder.up_blocks.2', stage_data['encoder_to_decoder_upblock2']),
        ('decoder.up_blocks.3', stage_data['decoder.up_blocks.3 final']),
        ('decoder.up_blocks.0 -> decoder.up_blocks.1 -> decoder.up_blocks.2 -> decoder.up_blocks.3', stage_data['decoder_upblocks0123']),
        ('decoder entry -> decoder.mid_block -> decoder.up_blocks.3', stage_data['decoder_entry_midblock_upblocks0123']),
        ('quant -> decoder.up_blocks.3', stage_data['quant_to_decoder_upblock3']),
        ('encoder -> quant -> decoder.up_blocks.3', stage_data['encoder_to_decoder_upblock3']),
        ('decoder.conv_norm_out', stage_data['decoder.tail.conv_norm_out']),
        ('decoder.conv_act', stage_data['decoder.tail.conv_act']),
        ('decoder.conv_out', stage_data['decoder.tail.conv_out']),
        ('decoder tail', stage_data['decoder.tail final']),
        ('decoder upblocks0-3 -> decoder tail', stage_data['decoder_upblocks0123_to_tail']),
        ('decoder entry -> mid_block -> upblocks0-3 -> tail', stage_data['decoder_entry_midblock_upblocks0123_tail']),
        ('quant -> decoder tail', stage_data['quant_to_decoder_tail']),
        ('encoder -> quant -> decoder tail', stage_data['encoder_to_decoder_tail']),
    ]:
        md.append(f"| `{label}` | {data.get('status', 'MISSING')} |")

    md += ['', '## Deterministic Decoder Stack Status', '', '- The TimeAware VAE deterministic decoder stack is now aligned through `post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks.0~3 -> decoder tail`.', '- The deterministic encoder-to-decoder-tail path is also aligned through posterior mode / mean; stochastic latent sampling is not implemented.', '- This is module-level deterministic alignment only, not full VAE API completion and not full TADSR inference.', '', '## Still Missing / Not Claimed', '', '- Full TimeAware VAE stochastic sampling/API integration is not complete.', '- Full UNet forward is not ported.', '- Full LoRA runtime integration is not complete; current validation covers the merge formula for one pair.', '- `jittor_tadsr_full.tadsr_full --full-inference` must remain NotImplemented until UNet and LoRA runtime integration pass.', '', '## Recommended Next Module', '', 'Continue with full UNet audit/export/port after deterministic TimeAware VAE decoder stack alignment.']

    (OUT / 'time_vae_lora_alignment_report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('Wrote', OUT / 'time_vae_lora_alignment_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
