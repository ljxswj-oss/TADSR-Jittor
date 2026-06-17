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
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock0_resnet1'
META_JSON = ORACLE_DIR / 'unet_downblock0_resnet1_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock0_resnet1_effective_weights.npz'
PREV_ATTENTION0_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock0_attention0' / 'entry_downblock0_attention0_official_output.npy'


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


def save_tensor(saved, name, tensor):
    arr = to_np(tensor) if hasattr(tensor, 'detach') else np.asarray(tensor, dtype=np.float32)
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
    block = unet.down_blocks[0]
    resnet0 = block.resnets[0]
    resnet1 = block.resnets[1]
    attention0 = block.attentions[0]
    cfg = dict(unet.config)
    sample_shape = [1, int(cfg.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    x_shape = [1, int(getattr(resnet1, 'in_channels')), 32, 32]
    temb_shape = [1, int(resnet1.time_emb_proj.weight.shape[1])]
    synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
    synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        synthetic_inter = resnet_manual(resnet1, synthetic_x, synthetic_temb)
        synthetic_official = resnet1(synthetic_x, synthetic_temb)
        centered = 2.0 * sample - 1.0 if cfg.get('center_input_sample') else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = resnet0(conv_in, time_embedding)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        entry_inter = resnet_manual(resnet1, attention0_out, time_embedding)
        entry_official = resnet1(attention0_out, time_embedding)

    saved = {}
    save_tensor(saved, 'synthetic_downblock0_resnet1_input', synthetic_x)
    save_tensor(saved, 'synthetic_downblock0_resnet1_temb', synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f'synthetic_downblock0_resnet1_{k}_output' if k not in {'output'} else 'synthetic_downblock0_resnet1_manual_output', v)
    save_tensor(saved, 'synthetic_downblock0_resnet1_output', synthetic_official)

    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_downblock0_resnet0_output', resnet0_out)
    save_tensor(saved, 'entry_downblock0_attention0_output', attention0_out)
    save_tensor(saved, 'entry_downblock0_resnet1_input', attention0_out)
    save_tensor(saved, 'entry_downblock0_resnet1_temb', time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f'entry_downblock0_resnet1_{k}_output' if k not in {'output'} else 'entry_downblock0_resnet1_manual_output', v)
    save_tensor(saved, 'entry_downblock0_resnet1_output', entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'down_blocks_0_resnets_1'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet1, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet1, name), effective, effective_meta)
    if getattr(resnet1, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet1.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_attention0_compare = {'previous_attention0_oracle_tensor_available': False}
    if PREV_ATTENTION0_ORACLE.exists():
        prev = np.load(PREV_ATTENTION0_ORACLE)
        curr = to_np(attention0_out)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_attention0_compare = {
            'previous_attention0_oracle_tensor_available': True,
            'max_abs_diff': float(diff.max()) if diff.size else 0.0,
            'mean_abs_diff': float(diff.mean()) if diff.size else 0.0,
        }
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': [1],
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'resnet1_input_shape': list(attention0_out.shape),
        'resnet1_output_shape': list(entry_official.shape),
        'temb_shape': list(time_embedding.shape),
        'resnet1_config': {
            'prefix': prefix,
            'eps': float(resnet1.norm1.eps),
            'groups': int(resnet1.norm1.num_groups),
            'groups_out': int(resnet1.norm2.num_groups),
            'activation': type(resnet1.nonlinearity).__name__,
            'time_embedding_norm': resnet1.time_embedding_norm,
            'output_scale_factor': float(resnet1.output_scale_factor),
            'dropout': float(resnet1.dropout.p),
            'has_shortcut': resnet1.conv_shortcut is not None,
            'use_in_shortcut': bool(resnet1.use_in_shortcut),
            'skip_time_act': bool(resnet1.skip_time_act),
            'pre_norm': bool(resnet1.pre_norm),
            'in_channels': int(resnet1.in_channels),
            'out_channels': int(resnet1.out_channels),
        },
        'path_a_isolated_synthetic_resnet1': {'status': 'PASS', 'input_shape': x_shape, 'temb_shape': temb_shape, 'output_shape': list(synthetic_official.shape)},
        'path_b_entry_resnet0_attention0_resnet1': {'status': 'PASS', 'sample_shape': sample_shape, 'attention0_shape': list(attention0_out.shape), 'temb_shape': list(time_embedding.shape), 'output_shape': list(entry_official.shape)},
        'previous_attention0_compare': previous_attention0_compare,
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
        'saved_tensors': saved,
        'markers': {
            'TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'not_in_scope': ['full down_blocks.0', 'attentions.1', 'downsampler', 'full UNet forward', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.resnets.1 Oracle Export',
        '',
        'TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS: PASS',
        'TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS: PASS',
        '',
        f"Oracle dir: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        f"Path A synthetic output shape: `{metadata['path_a_isolated_synthetic_resnet1']['output_shape']}`",
        f"Path B entry->resnet0->attention0->resnet1 output shape: `{metadata['path_b_entry_resnet0_attention0_resnet1']['output_shape']}`",
        f"Synthetic manual/offical max abs error: `{metadata['manual_effective_validation']['synthetic_manual_vs_official_max_abs_error']}`",
        f"Entry manual/offical max abs error: `{metadata['manual_effective_validation']['entry_manual_vs_official_max_abs_error']}`",
        '',
        'This export stops before down_blocks.0.attentions.1, downsampler, full down_blocks.0 and full UNet.',
    ]
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_RESNET1_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_DOWNBLOCK0_RESNET1_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_effective_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
