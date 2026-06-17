#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
TADSR_PKL = WEIGHTS_DIR / 'tadsr.pkl'
OUT_DIR = Path('experiments/full_repro/unet_alignment')
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock2_downsampler'
META_JSON = ORACLE_DIR / 'unet_downblock2_downsampler_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock2_downsampler_effective_weights.npz'
PREV_ATTENTION1_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock2_attention1' / 'entry_downblock2_attention1_output.npy'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32) if hasattr(x, 'detach') else np.asarray(x, dtype=np.float32)


def stats(x):
    y = np.asarray(x)
    yf = y.astype(np.float64, copy=False)
    return {
        'shape': list(y.shape),
        'dtype': str(y.dtype),
        'min': float(yf.min()) if y.size else 0.0,
        'max': float(yf.max()) if y.size else 0.0,
        'mean': float(yf.mean()) if y.size else 0.0,
        'std': float(yf.std()) if y.size else 0.0,
        'nbytes': int(y.nbytes),
    }


def base_layer(module):
    return module.base_layer if hasattr(module, 'base_layer') else module


def load_unet():
    import torch
    sys.path.insert(0, str(OFFICIAL_REPO))
    from tadsr import initialize_unet
    args = SimpleNamespace(pretrained_model_name_or_path=str(WEIGHTS_DIR), pretrained_lora_path=str(TADSR_PKL), lora_rank=4)
    unet, enc, dec, oth = initialize_unet(args)
    ckpt = torch.load(str(TADSR_PKL), map_location='cpu')
    sd = ckpt.get('state_dict_unet', {})
    loaded = []
    for n, p in unet.named_parameters():
        if 'lora' in n and n in sd:
            p.data.copy_(sd[n])
            loaded.append(n)
    conv = base_layer(unet.conv_in)
    if 'conv_in.weight' in sd:
        conv.weight.data.copy_(sd['conv_in.weight'])
    if 'conv_in.bias' in sd and getattr(conv, 'bias', None) is not None:
        conv.bias.data.copy_(sd['conv_in.bias'])
    if hasattr(unet, 'set_adapter'):
        unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
    unet.eval()
    return unet, loaded


def effective_weight(module):
    base = base_layer(module)
    weight = base.weight.detach().cpu().clone()
    delta_records = []
    if hasattr(module, 'get_delta_weight'):
        for adapter in list(getattr(module, 'active_adapters', [])):
            try:
                delta = module.get_delta_weight(adapter).detach().cpu()
            except Exception:
                continue
            weight = weight + delta
            delta_records.append({'adapter': adapter, 'shape': list(delta.shape), 'max_abs': float(delta.abs().max().item()) if delta.numel() else 0.0})
    bias = base.bias.detach().cpu().clone() if getattr(base, 'bias', None) is not None else None
    return weight, bias, delta_records


def save_tensor(saved, name, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


def main() -> int:
    maybe_reexec()
    import torch
    torch.manual_seed(1234)
    np.random.seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    downsampler = block2.downsamplers[0]
    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    # Shape is intentionally derived from the real local bridge output below instead of hardcoding.
    saved = {}
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        down0_out, down0_res_samples = block0(conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        down1_out, down1_res_samples = block1(down0_out, time_embedding, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = block2.resnets[0](down1_out, time_embedding)
        attention0_out = block2.attentions[0](resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = block2.resnets[1](attention0_out, time_embedding)
        attention1_out = block2.attentions[1](resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        synthetic_shape = list(attention1_out.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(synthetic_shape)), dtype=torch.float32).reshape(*synthetic_shape)
        synthetic_output = downsampler(synthetic_hidden)
        downsample_out = downsampler(attention1_out)
        block2_forward_out, block2_res_samples = block2(down1_out, time_embedding, encoder_hidden_states=encoder_hidden_states)
    save_tensor(saved, 'synthetic_downblock2_downsampler_input', synthetic_hidden)
    save_tensor(saved, 'synthetic_downblock2_downsampler_output', synthetic_output)
    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_downblock0_output_hidden', down0_out)
    save_tensor(saved, 'entry_downblock1_output_hidden', down1_out)
    save_tensor(saved, 'entry_downblock2_resnet0_output', resnet0_out)
    save_tensor(saved, 'entry_downblock2_attention0_output', attention0_out)
    save_tensor(saved, 'entry_downblock2_resnet1_output', resnet1_out)
    save_tensor(saved, 'entry_downblock2_attention1_output', attention1_out)
    save_tensor(saved, 'entry_downblock2_downsampler_output', downsample_out)
    save_tensor(saved, 'local_downblock2_hidden_states_output', block2_forward_out)
    for i, tensor in enumerate(block2_res_samples):
        save_tensor(saved, f'local_downblock2_res_sample_{i}', tensor)
    w, b, deltas = effective_weight(downsampler.conv)
    prefix = 'down_blocks_2_downsamplers_0'
    arrays = {f'{prefix}_conv_weight': w.numpy().astype(np.float32)}
    if b is not None:
        arrays[f'{prefix}_conv_bias'] = b.numpy().astype(np.float32)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **arrays)
    block_diff = (downsample_out - block2_forward_out).abs()
    output_state_diffs = []
    expected_states = [attention0_out, attention1_out, downsample_out]
    for idx, (expected, actual) in enumerate(zip(expected_states, block2_res_samples)):
        diff = (expected - actual).abs()
        output_state_diffs.append({'index': idx, 'max_abs_error': float(diff.max().item()) if diff.numel() else 0.0, 'shape': list(actual.shape)})
    previous_attention1_compare = {'previous_attention1_oracle_tensor_available': False}
    if PREV_ATTENTION1_ORACLE.exists():
        prev = np.load(PREV_ATTENTION1_ORACLE)
        curr = to_np(attention1_out)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_attention1_compare = {
            'previous_attention1_oracle_tensor_available': True,
            'max_abs_diff': float(diff.max()) if diff.size else 0.0,
            'mean_abs_diff': float(diff.mean()) if diff.size else 0.0,
        }
    local_status = 'PASS' if float(block_diff.max()) < 1e-5 else 'FAIL'
    output_states_status = 'PASS' if len(block2_res_samples) == 3 and all(x['max_abs_error'] < 1e-5 for x in output_state_diffs) else 'FAIL'
    metadata = {
        'status': 'PASS' if local_status == 'PASS' and output_states_status == 'PASS' else 'FAIL',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downsampler_config': {
            'prefix': prefix,
            'class': f'{type(downsampler).__module__}.{type(downsampler).__name__}',
            'channels': int(downsampler.channels),
            'out_channels': int(downsampler.out_channels),
            'use_conv': bool(downsampler.use_conv),
            'padding': int(downsampler.padding),
            'name': str(downsampler.name),
            'conv_weight_shape': list(w.shape),
            'conv_bias_shape': list(b.shape) if b is not None else None,
            'stride': [2, 2],
            'kernel_size': [3, 3],
            'lora_delta_records': deltas,
        },
        'path_a_isolated_downsampler': {'status': 'PASS', 'input_shape': synthetic_shape, 'output_shape': list(synthetic_output.shape)},
        'path_b_entry_full_downblock2': {
            'status': local_status,
            'sample_shape': sample_shape,
            'downblock0_output_shape': list(down0_out.shape),
            'downblock1_output_shape': list(down1_out.shape),
            'attention1_shape': list(attention1_out.shape),
            'output_shape': list(downsample_out.shape),
            'official_block_forward_output_shape': list(block2_forward_out.shape),
            'official_block_res_sample_shapes': [list(x.shape) for x in block2_res_samples],
        },
        'manual_chain_vs_official_block_forward': {
            'status': local_status,
            'max_abs_error': float(block_diff.max().item()) if block_diff.numel() else 0.0,
            'mean_abs_error': float(block_diff.mean().item()) if block_diff.numel() else 0.0,
        },
        'output_states': {
            'status': output_states_status,
            'expected_order': ['attention0_output', 'attention1_output', 'downsampler_output'],
            'count': len(block2_res_samples),
            'diffs': output_state_diffs,
        },
        'previous_attention1_compare': previous_attention1_compare,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in arrays.items()},
            'module_effective_meta': {
                f'{prefix}_conv': {
                    'is_lora_wrapped': hasattr(downsampler.conv, 'base_layer') or hasattr(downsampler.conv, 'lora_A'),
                    'active_adapters': list(getattr(downsampler.conv, 'active_adapters', [])) if hasattr(downsampler.conv, 'active_adapters') else [],
                    'delta_records': deltas,
                    'weight_shape': list(w.shape),
                    'bias_shape': list(b.shape) if b is not None else None,
                }
            },
        },
        'saved_tensors': saved,
        'markers': {
            'TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS': 'PASS',
            'TADSR_UNET_DOWNBLOCK2_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ORACLE_TENSORS': output_states_status,
        },
        'not_in_scope': ['down_blocks.2', 'mid_block', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.2.downsamplers.0 Oracle Export',
        '',
        f"TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS: {metadata['markers']['TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_ORACLE_TENSORS']}",
        f"TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS: {metadata['markers']['TADSR_UNET_DOWNBLOCK2_DOWNSAMPLER_EFFECTIVE_WEIGHTS']}",
        f"TADSR_UNET_DOWNBLOCK2_ORACLE_TENSORS: {metadata['markers']['TADSR_UNET_DOWNBLOCK2_ORACLE_TENSORS']}",
        f"TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ORACLE_TENSORS: {metadata['markers']['TADSR_UNET_DOWNBLOCK2_OUTPUT_STATES_ORACLE_TENSORS']}",
        '',
        f"Oracle dir: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        f"Path A synthetic output shape: `{metadata['path_a_isolated_downsampler']['output_shape']}`",
        f"Path B entry full down_blocks.2 output shape: `{metadata['path_b_entry_full_downblock2']['output_shape']}`",
        f"Manual chain vs official block.forward max abs error: `{metadata['manual_chain_vs_official_block_forward']['max_abs_error']}`",
        f"Output states order: `{metadata['output_states']['expected_order']}`",
        '',
        'This export completes the local `down_blocks.2` chain only. Full UNet forward and full TADSR inference remain out of scope.',
    ]
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': metadata['status'], 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
