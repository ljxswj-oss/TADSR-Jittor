#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import numpy as np
import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
NPZ_PATH = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')
OUT = Path('experiments/full_repro/time_vae_alignment')


def shape_or_none(obj):
    return list(obj.shape) if obj is not None and hasattr(obj, 'shape') else None


def module_order(module):
    rows = []
    for name, child in module.named_children():
        rows.append({'name': name, 'class': child.__class__.__name__, 'module': child.__class__.__module__})
    return rows


def conv_info(mod):
    if mod is None:
        return {'exists': False}
    return {
        'exists': True,
        'class': mod.__class__.__name__,
        'module': mod.__class__.__module__,
        'weight_shape': shape_or_none(getattr(mod, 'weight', None)),
        'bias_shape': shape_or_none(getattr(mod, 'bias', None)),
        'in_channels': int(getattr(mod, 'in_channels', 0)) if hasattr(mod, 'in_channels') else None,
        'out_channels': int(getattr(mod, 'out_channels', 0)) if hasattr(mod, 'out_channels') else None,
        'kernel_size': list(getattr(mod, 'kernel_size', [])) if hasattr(mod, 'kernel_size') else None,
        'stride': list(getattr(mod, 'stride', [])) if hasattr(mod, 'stride') else None,
        'padding': list(getattr(mod, 'padding', [])) if hasattr(mod, 'padding') else None,
    }


def resnet_info(resnet, name):
    conv_shortcut = getattr(resnet, 'conv_shortcut', None)
    nin_shortcut = getattr(resnet, 'nin_shortcut', None)
    time_emb_proj = getattr(resnet, 'time_emb_proj', None)
    return {
        'module_name': name,
        'class': resnet.__class__.__name__,
        'module': resnet.__class__.__module__,
        'in_channels': getattr(resnet, 'in_channels', None),
        'out_channels': getattr(resnet, 'out_channels', None),
        'time_embedding_norm': getattr(resnet, 'time_embedding_norm', None),
        'temb_channels': getattr(resnet, 'temb_channels', None),
        'groups': getattr(resnet, 'groups', None),
        'groups_out': getattr(resnet, 'groups_out', None),
        'eps': float(getattr(resnet.norm1, 'eps', 0.0)) if hasattr(resnet, 'norm1') else None,
        'non_linearity': getattr(resnet, 'non_linearity', None),
        'output_scale_factor': getattr(resnet, 'output_scale_factor', None),
        'norm1_weight_shape': shape_or_none(getattr(resnet.norm1, 'weight', None)),
        'norm1_bias_shape': shape_or_none(getattr(resnet.norm1, 'bias', None)),
        'norm2_weight_shape': shape_or_none(getattr(resnet.norm2, 'weight', None)),
        'norm2_bias_shape': shape_or_none(getattr(resnet.norm2, 'bias', None)),
        'conv1_weight_shape': shape_or_none(getattr(resnet.conv1, 'weight', None)),
        'conv1_bias_shape': shape_or_none(getattr(resnet.conv1, 'bias', None)),
        'conv2_weight_shape': shape_or_none(getattr(resnet.conv2, 'weight', None)),
        'conv2_bias_shape': shape_or_none(getattr(resnet.conv2, 'bias', None)),
        'time_emb_proj_exists': time_emb_proj is not None,
        'time_emb_proj_weight_shape': shape_or_none(getattr(time_emb_proj, 'weight', None)) if time_emb_proj is not None else None,
        'time_emb_proj_bias_shape': shape_or_none(getattr(time_emb_proj, 'bias', None)) if time_emb_proj is not None else None,
        'conv_shortcut_exists': conv_shortcut is not None,
        'conv_shortcut_weight_shape': shape_or_none(getattr(conv_shortcut, 'weight', None)) if conv_shortcut is not None else None,
        'conv_shortcut_bias_shape': shape_or_none(getattr(conv_shortcut, 'bias', None)) if conv_shortcut is not None else None,
        'nin_shortcut_exists': nin_shortcut is not None,
        'nin_shortcut_weight_shape': shape_or_none(getattr(nin_shortcut, 'weight', None)) if nin_shortcut is not None else None,
        'npz_prefix': name.replace('.', '__'),
    }


def upsampler_info(upsampler, name):
    conv = getattr(upsampler, 'conv', None)
    return {
        'module_name': name,
        'class': upsampler.__class__.__name__,
        'module': upsampler.__class__.__module__,
        'children_order': module_order(upsampler),
        'channels': getattr(upsampler, 'channels', None),
        'out_channels': getattr(upsampler, 'out_channels', None),
        'use_conv': getattr(upsampler, 'use_conv', None),
        'use_conv_transpose': getattr(upsampler, 'use_conv_transpose', None),
        'interpolate': getattr(upsampler, 'interpolate', None),
        'padding': getattr(upsampler, 'padding', None),
        'expected_mode': 'nearest',
        'expected_scale_factor': 2.0,
        'conv': conv_info(conv),
        'npz_mapping': {
            'conv': f'{name.replace(".", "__")}__conv__weight',
        },
    }


def npz_rows():
    if not NPZ_PATH.exists():
        return []
    data = np.load(NPZ_PATH)
    rows = []
    for key in data.files:
        if key.startswith('decoder__up_blocks'):
            rows.append({'key': key, 'shape': list(data[key].shape), 'dtype': str(data[key].dtype), 'numel': int(data[key].size)})
    return rows


def block_summary(block, index, input_shape=None, output_shape=None):
    resnets = list(getattr(block, 'resnets', []))
    upsamplers = list(getattr(block, 'upsamplers', []) or [])
    attentions = list(getattr(block, 'attentions', []) or [])
    resnet_infos = [resnet_info(r, f'decoder.up_blocks.{index}.resnets.{i}') for i, r in enumerate(resnets)]
    upsampler_infos = [upsampler_info(u, f'decoder.up_blocks.{index}.upsamplers.{i}') for i, u in enumerate(upsamplers)]
    return {
        'block_index': index,
        'class': block.__class__.__name__,
        'module': block.__class__.__module__,
        'forward_signature': str(inspect.signature(block.forward)),
        'children_order': module_order(block),
        'input_shape': input_shape,
        'output_shape': output_shape,
        'resnet_count': len(resnets),
        'attention_count': len([x for x in attentions if x is not None]),
        'upsampler_count': len(upsamplers),
        'has_upsampler': bool(upsamplers),
        'has_attention': any(x is not None for x in attentions),
        'module_order': [f'resnets.{i}' for i in range(len(resnets))] + [f'upsamplers.{i}' for i in range(len(upsamplers))],
        'uses_temb': 'temb' in str(inspect.signature(block.forward)),
        'default_temb_is_none': True,
        'spatial_size_changes': bool(input_shape and output_shape and input_shape[-2:] != output_shape[-2:]),
        'channel_count_changes': bool(input_shape and output_shape and input_shape[1] != output_shape[1]),
        'resnet_info': resnet_infos,
        'upsampler_info': upsampler_infos,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    decoder = model.decoder
    up_blocks = decoder.up_blocks

    input_shapes: dict[int, list[int]] = {}
    output_shapes: dict[int, list[int]] = {}

    hooks = []

    def make_hook(index):
        def hook(_module, inputs, output):
            first = inputs[0]
            input_shapes[index] = list(first.shape) if hasattr(first, 'shape') else None
            output_shapes[index] = list(output.shape) if hasattr(output, 'shape') else None

        return hook

    for i, block in enumerate(up_blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        z = torch.linspace(-1.0, 1.0, steps=1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        h = decoder.mid_block(decoder.conv_in(model.post_quant_conv(z)), None)
        sequential_outputs = []
        h_all = h
        for block in up_blocks:
            h_all = block(h_all, None)
            sequential_outputs.append(h_all)

    for hnd in hooks:
        hnd.remove()

    blocks = [block_summary(block, i, input_shapes.get(i), output_shapes.get(i)) for i, block in enumerate(up_blocks)]
    def validate_upblock(
        index: int,
        expected_input: list[int] | None = None,
        expected_output: list[int] | None = None,
        expected_upsamplers: int | None = 1,
        allow_no_upsampler: bool = False,
    ) -> tuple[str, str, dict]:
        block = blocks[index] if len(blocks) > index else {}
        status = 'PASS'
        reason = ''
        upsampler_count = block.get('upsampler_count')
        expected_order = ['resnets.0', 'resnets.1', 'resnets.2'] + [f'upsamplers.{i}' for i in range(upsampler_count or 0)]
        if len(blocks) <= index:
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'decoder.up_blocks.{index} is missing'
        elif block.get('class') != 'UpDecoderBlock2D':
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f"unexpected block class {block.get('class')}"
        elif block.get('resnet_count') != 3 or block.get('attention_count') != 0:
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} topology is not 3 resnets with no attention'
        elif expected_upsamplers is not None and upsampler_count != expected_upsamplers:
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} has {upsampler_count} upsamplers, expected {expected_upsamplers}'
        elif expected_upsamplers is None and not (upsampler_count == 1 or (allow_no_upsampler and upsampler_count == 0)):
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} has unsupported upsampler count {upsampler_count}'
        elif expected_input is not None and expected_output is not None and (block.get('input_shape') != expected_input or block.get('output_shape') != expected_output):
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} input/output shape does not match expected {expected_input} -> {expected_output}'
        elif block.get('module_order') != expected_order:
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} module order is not resnets.* followed by available upsamplers.*'
        elif block.get('has_attention'):
            status = 'BLOCKED_NEEDS_REVIEW'
            reason = f'upblock{index} unexpectedly has attention'
        return status, reason, block

    up0_status, up0_reason, up0 = validate_upblock(0, [1, 512, 4, 4], [1, 512, 8, 8])
    up1_status, up1_reason, up1 = validate_upblock(1, [1, 512, 8, 8], [1, 512, 16, 16])
    up2_status, up2_reason, up2 = validate_upblock(2, [1, 512, 16, 16], [1, 256, 32, 32])
    up3_status, up3_reason, up3 = validate_upblock(
        3,
        [1, 256, 32, 32],
        [1, 128, 32, 32],
        expected_upsamplers=None,
        allow_no_upsampler=True,
    )

    result = {
        'status': 'PASS' if up0_status == 'PASS' and up1_status == 'PASS' and up2_status == 'PASS' and up3_status == 'PASS' else 'BLOCKED_NEEDS_REVIEW',
        'official_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'decoder_class': f'{decoder.__class__.__module__}.{decoder.__class__.__name__}',
        'decoder_forward_signature': str(inspect.signature(decoder.forward)),
        'decoder_forward_source_excerpt': inspect.getsource(decoder.forward)[:4000],
        'decoder_up_blocks_container_class': up_blocks.__class__.__name__,
        'decoder_up_blocks_count': len(up_blocks),
        'decoder_execution_order': 'post_quant_conv -> decoder.conv_in -> decoder.mid_block -> decoder.up_blocks -> decoder tail',
        'default_decode_passes_temb_none': True,
        'up_blocks': blocks,
        'upblock0_status': up0_status,
        'upblock0_reason': up0_reason,
        'upblock0_summary': up0,
        'upblock1_status': up1_status,
        'upblock1_reason': up1_reason,
        'upblock1_summary': up1,
        'upblock2_status': up2_status,
        'upblock2_reason': up2_reason,
        'upblock2_summary': up2,
        'upblock3_status': up3_status,
        'upblock3_reason': up3_reason,
        'upblock3_summary': up3,
        'upblock0_probe_output_shape': list(sequential_outputs[0].shape) if len(sequential_outputs) > 0 else None,
        'upblock1_probe_output_shape': list(sequential_outputs[1].shape) if len(sequential_outputs) > 1 else None,
        'upblock2_probe_output_shape': list(sequential_outputs[2].shape) if len(sequential_outputs) > 2 else None,
        'upblock3_probe_output_shape': list(sequential_outputs[3].shape) if len(sequential_outputs) > 3 else None,
        'full_up_blocks_probe_output_shape': list(h_all.shape),
        'converted_npz_keys': npz_rows(),
    }
    (OUT / 'audit_time_vae_decoder_upblocks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    up0_doc = {'status': up0_status, 'reason': up0_reason, **up0}
    up1_doc = {'status': up1_status, 'reason': up1_reason, **up1}
    up2_doc = {'status': up2_status, 'reason': up2_reason, **up2}
    up3_doc = {'status': up3_status, 'reason': up3_reason, **up3}
    (OUT / 'audit_time_vae_decoder_upblock0.json').write_text(json.dumps(up0_doc, indent=2), encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock1.json').write_text(json.dumps(up1_doc, indent=2), encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock2.json').write_text(json.dumps(up2_doc, indent=2), encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock3.json').write_text(json.dumps(up3_doc, indent=2), encoding='utf-8')

    lines = [
        '# TimeAware VAE Decoder UpBlocks Audit',
        '',
        f"Status: {result['status']}",
        f"TIME_VAE_DECODER_UPBLOCKS_AUDIT: {result['status']}",
        f"TIME_VAE_DECODER_UPBLOCK0_AUDIT: {up0_status}",
        f"TIME_VAE_DECODER_UPBLOCK1_AUDIT: {up1_status}",
        f"TIME_VAE_DECODER_UPBLOCK2_AUDIT: {up2_status}",
        f"TIME_VAE_DECODER_UPBLOCK3_AUDIT: {up3_status}",
        '',
        f"- decoder class: `{result['decoder_class']}`",
        f"- up_blocks count: `{len(up_blocks)}`",
        f"- default decoder-side temb: `None`",
        '',
        '## UpBlock Overview',
        '',
        '| Index | Class | Input | Output | ResNets | Upsamplers | Attention | Spatial change | Channel change |',
        '|---:|---|---|---|---:|---:|---:|---|---|',
    ]
    for block in blocks:
        lines.append(
            f"| {block['block_index']} | `{block['class']}` | `{block['input_shape']}` | `{block['output_shape']}` | "
            f"{block['resnet_count']} | {block['upsampler_count']} | {block['attention_count']} | {block['spatial_size_changes']} | {block['channel_count_changes']} |"
        )
    def append_block_detail(block: dict, label: str) -> None:
        lines.extend(['', f'## {label} ResNets', '', '| Module | in | out | conv1 | conv2 | shortcut | time proj | npz prefix |', '|---|---:|---:|---|---|---|---|---|'])
        for row in block.get('resnet_info', []):
            lines.append(
                f"| `{row['module_name']}` | {row.get('in_channels')} | {row.get('out_channels')} | `{row.get('conv1_weight_shape')}` | "
                f"`{row.get('conv2_weight_shape')}` | {row.get('conv_shortcut_exists')} | {row.get('time_emb_proj_exists')} | `{row.get('npz_prefix')}` |"
            )
        lines.extend(['', f'## {label} Upsamplers', '', '| Module | Class | Use conv | Interpolate | Conv shape | Padding | NPZ conv key |',
                      '|---|---|---|---|---|---|---|'])
        for row in block.get('upsampler_info', []):
            lines.append(
                f"| `{row['module_name']}` | `{row['class']}` | {row.get('use_conv')} | {row.get('interpolate')} | "
                f"`{row.get('conv', {}).get('weight_shape')}` | `{row.get('conv', {}).get('padding')}` | `{row.get('npz_mapping', {}).get('conv')}` |"
            )

    append_block_detail(up0, 'UpBlock0')
    append_block_detail(up1, 'UpBlock1')
    append_block_detail(up2, 'UpBlock2')
    append_block_detail(up3, 'UpBlock3')
    (OUT / 'audit_time_vae_decoder_upblocks.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock0.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock1.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock2.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'audit_time_vae_decoder_upblock3.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f"TIME_VAE_DECODER_UPBLOCKS_AUDIT: {result['status']}")
    print(f"TIME_VAE_DECODER_UPBLOCK0_AUDIT: {up0_status}")
    print(f"TIME_VAE_DECODER_UPBLOCK1_AUDIT: {up1_status}")
    print(f"TIME_VAE_DECODER_UPBLOCK2_AUDIT: {up2_status}")
    print(f"TIME_VAE_DECODER_UPBLOCK3_AUDIT: {up3_status}")
    return 0 if result['status'] == 'PASS' and up0_status == 'PASS' and up1_status == 'PASS' and up2_status == 'PASS' and up3_status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
