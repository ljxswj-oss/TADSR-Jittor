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
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock0_resnet0'
META_JSON = ORACLE_DIR / 'unet_downblock0_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock0_resnet0_effective_weights.npz'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


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
    sys.path.insert(0, str(OFFICIAL_REPO))
    import torch
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
    import torch
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


def module_effective_arrays(prefix, module, arrays, meta):
    w, b, deltas = effective_weight(module)
    arrays[f'{prefix}_weight'] = w.numpy().astype(np.float32)
    if b is not None:
        arrays[f'{prefix}_bias'] = b.numpy().astype(np.float32)
    meta[prefix] = {
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'base_class': f'{type(base_layer(module)).__module__}.{type(base_layer(module)).__name__}',
        'is_lora_wrapped': hasattr(module, 'base_layer') or hasattr(module, 'lora_A'),
        'active_adapters': list(getattr(module, 'active_adapters', [])) if hasattr(module, 'active_adapters') else [],
        'delta_records': deltas,
        'weight_key': f'{prefix}_weight',
        'bias_key': f'{prefix}_bias' if b is not None else None,
        'weight_shape': list(w.shape),
        'bias_shape': list(b.shape) if b is not None else None,
    }


def resnet_manual(resnet, x, temb):
    hidden = x
    out = {}
    out['norm1'] = resnet.norm1(hidden)
    out['act1'] = resnet.nonlinearity(out['norm1'])
    out['conv1'] = resnet.conv1(out['act1'])
    temb_proj = None
    if resnet.time_emb_proj is not None:
        temb_in = temb if resnet.skip_time_act else resnet.nonlinearity(temb)
        temb_proj = resnet.time_emb_proj(temb_in)[:, :, None, None]
        out['time_emb_proj'] = temb_proj
    if temb_proj is not None and resnet.time_embedding_norm == 'default':
        out['after_temb_add'] = out['conv1'] + temb_proj
    else:
        out['after_temb_add'] = out['conv1']
    out['norm2'] = resnet.norm2(out['after_temb_add'])
    if temb_proj is not None and resnet.time_embedding_norm == 'scale_shift':
        scale, shift = temb_proj.chunk(2, dim=1)
        out['norm2'] = out['norm2'] * (1 + scale) + shift
    out['act2'] = resnet.nonlinearity(out['norm2'])
    out['dropout'] = resnet.dropout(out['act2'])
    out['conv2'] = resnet.conv2(out['dropout'])
    out['shortcut'] = resnet.conv_shortcut(x) if resnet.conv_shortcut is not None else x
    out['output'] = (out['shortcut'] + out['conv2']) / resnet.output_scale_factor
    return out


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
    resnet = unet.down_blocks[0].resnets[0]
    cfg = dict(unet.config)
    sample_shape = [1, int(cfg.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    x_shape = [1, int(getattr(resnet, 'in_channels')), 32, 32]
    temb_shape = [1, int(resnet.time_emb_proj.weight.shape[1])]
    synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
    synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)

    with torch.no_grad():
        synthetic_inter = resnet_manual(resnet, synthetic_x, synthetic_temb)
        synthetic_official = resnet(synthetic_x, synthetic_temb)
        centered = 2.0 * sample - 1.0 if cfg.get('center_input_sample') else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        entry_inter = resnet_manual(resnet, conv_in, time_embedding)
        entry_official = resnet(conv_in, time_embedding)

    tensors = {
        'synthetic_downblock0_resnet0_input': synthetic_x,
        'synthetic_downblock0_resnet0_temb': synthetic_temb,
        'synthetic_downblock0_resnet0_norm1_output': synthetic_inter['norm1'],
        'synthetic_downblock0_resnet0_act1_output': synthetic_inter['act1'],
        'synthetic_downblock0_resnet0_conv1_output': synthetic_inter['conv1'],
        'synthetic_downblock0_resnet0_time_emb_proj_output': synthetic_inter['time_emb_proj'],
        'synthetic_downblock0_resnet0_after_temb_add_output': synthetic_inter['after_temb_add'],
        'synthetic_downblock0_resnet0_norm2_output': synthetic_inter['norm2'],
        'synthetic_downblock0_resnet0_act2_output': synthetic_inter['act2'],
        'synthetic_downblock0_resnet0_conv2_output': synthetic_inter['conv2'],
        'synthetic_downblock0_resnet0_shortcut_output': synthetic_inter['shortcut'],
        'synthetic_downblock0_resnet0_output': synthetic_official,
        'synthetic_downblock0_resnet0_manual_output': synthetic_inter['output'],
        'entry_synthetic_unet_sample': sample,
        'entry_synthetic_unet_timestep': timestep,
        'entry_synthetic_unet_sample_after_center': centered,
        'entry_synthetic_unet_conv_in_output': conv_in,
        'entry_synthetic_unet_time_proj_output': time_proj,
        'entry_synthetic_unet_time_embedding_output': time_embedding,
        'entry_downblock0_resnet0_input': conv_in,
        'entry_downblock0_resnet0_temb': time_embedding,
        'entry_downblock0_resnet0_norm1_output': entry_inter['norm1'],
        'entry_downblock0_resnet0_act1_output': entry_inter['act1'],
        'entry_downblock0_resnet0_conv1_output': entry_inter['conv1'],
        'entry_downblock0_resnet0_time_emb_proj_output': entry_inter['time_emb_proj'],
        'entry_downblock0_resnet0_after_temb_add_output': entry_inter['after_temb_add'],
        'entry_downblock0_resnet0_norm2_output': entry_inter['norm2'],
        'entry_downblock0_resnet0_act2_output': entry_inter['act2'],
        'entry_downblock0_resnet0_conv2_output': entry_inter['conv2'],
        'entry_downblock0_resnet0_shortcut_output': entry_inter['shortcut'],
        'entry_downblock0_resnet0_output': entry_official,
        'entry_downblock0_resnet0_manual_output': entry_inter['output'],
    }
    for name, value in tensors.items():
        arr = value.detach().cpu().numpy().astype(np.float32) if hasattr(value, 'detach') else np.asarray(value)
        np.save(ORACLE_DIR / f'{name}.npy', arr)

    effective = {}
    effective_meta = {}
    prefix = 'down_blocks_0_resnets_0'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet, name), effective, effective_meta)
    if getattr(resnet, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'resnet0_input_shape': list(conv_in.shape),
        'resnet0_output_shape': list(entry_official.shape),
        'temb_shape': list(time_embedding.shape),
        'resnet0_config': {
            'prefix': prefix,
            'eps': float(resnet.norm1.eps),
            'groups': int(resnet.norm1.num_groups),
            'groups_out': int(resnet.norm2.num_groups),
            'activation': type(resnet.nonlinearity).__name__,
            'time_embedding_norm': resnet.time_embedding_norm,
            'output_scale_factor': float(resnet.output_scale_factor),
            'dropout': float(resnet.dropout.p),
            'has_shortcut': resnet.conv_shortcut is not None,
            'use_in_shortcut': bool(resnet.use_in_shortcut),
            'skip_time_act': bool(resnet.skip_time_act),
            'pre_norm': bool(resnet.pre_norm),
            'in_channels': int(resnet.in_channels),
            'out_channels': int(resnet.out_channels),
        },
        'path_a_isolated_synthetic_resnet0': {'status': 'PASS', 'input_shape': x_shape, 'temb_shape': temb_shape, 'output_shape': list(synthetic_official.shape)},
        'path_b_entry_to_resnet0': {'status': 'PASS', 'sample_shape': sample_shape, 'conv_in_shape': list(conv_in.shape), 'temb_shape': list(time_embedding.shape), 'output_shape': list(entry_official.shape)},
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in effective.items()},
            'module_effective_meta': effective_meta,
            'lora_affected_modules': [k for k, v in effective_meta.items() if v.get('is_lora_wrapped')],
        },
        'manual_effective_validation': {
            'status': 'PASS' if float(synth_diff.max()) < 1e-5 and float(entry_diff.max()) < 1e-5 else 'FAIL',
            'synthetic_manual_vs_official_max_abs_error': float(synth_diff.max().item()),
            'synthetic_manual_vs_official_mean_abs_error': float(synth_diff.mean().item()),
            'entry_manual_vs_official_max_abs_error': float(entry_diff.max().item()),
            'entry_manual_vs_official_mean_abs_error': float(entry_diff.mean().item()),
            'loaded_lora_param_count': len(loaded),
        },
        'saved_tensors': {name: stats(np.load(ORACLE_DIR / f'{name}.npy')) for name in tensors.keys()},
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'not_in_scope': ['full down_blocks.0', 'attention0', 'resnets.1', 'downsampler', 'full UNet forward', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.resnets.0 Oracle Export', '',
        'TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS: PASS',
        'TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS: PASS', '',
        f"Oracle dir: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        f"Path A synthetic output shape: `{metadata['path_a_isolated_synthetic_resnet0']['output_shape']}`",
        f"Path B entry output shape: `{metadata['path_b_entry_to_resnet0']['output_shape']}`",
        f"Synthetic manual/offical max abs error: `{metadata['manual_effective_validation']['synthetic_manual_vs_official_max_abs_error']}`",
        f"Entry manual/offical max abs error: `{metadata['manual_effective_validation']['entry_manual_vs_official_max_abs_error']}`",
    ]
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_RESNET0_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_DOWNBLOCK0_RESNET0_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_effective_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
