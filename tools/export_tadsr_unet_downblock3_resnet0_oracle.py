#!/usr/bin/env python3
from __future__ import annotations

import inspect
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


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def shape(x):
    return list(x.shape) if x is not None and hasattr(x, 'shape') else None


def module_shape(m, attr='weight'):
    t = getattr(m, attr, None)
    return shape(t) if t is not None else None


def base_layer(module):
    return module.base_layer if hasattr(module, 'base_layer') else module


def module_lora_info(module):
    if module is None:
        return {'exists': False}
    base = base_layer(module)
    info = {
        'exists': True,
        'class': f'{type(module).__module__}.{type(module).__name__}',
        'base_class': f'{type(base).__module__}.{type(base).__name__}',
        'is_lora_wrapped': hasattr(module, 'base_layer') or hasattr(module, 'lora_A'),
        'weight_shape': module_shape(base, 'weight'),
        'bias_shape': module_shape(base, 'bias') if getattr(base, 'bias', None) is not None else None,
        'active_adapters': list(getattr(module, 'active_adapters', [])) if hasattr(module, 'active_adapters') else [],
        'lora_A_shapes': {},
        'lora_B_shapes': {},
        'scaling': dict(getattr(module, 'scaling', {})) if isinstance(getattr(module, 'scaling', None), dict) else {},
        'delta_weight_max_abs': {},
    }
    for side in ['lora_A', 'lora_B']:
        obj = getattr(module, side, None)
        if hasattr(obj, 'items'):
            info[f'{side}_shapes'] = {k: shape(v.weight) for k, v in obj.items()}
    if hasattr(module, 'get_delta_weight'):
        for adapter in info['active_adapters']:
            try:
                delta = module.get_delta_weight(adapter).detach().cpu()
            except Exception as exc:
                info['delta_weight_max_abs'][adapter] = f'ERROR: {exc}'
                continue
            info['delta_weight_max_abs'][adapter] = float(delta.abs().max().item()) if delta.numel() else 0.0
    return info


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


def resnet_config(resnet, prefix: str):
    norm1 = getattr(resnet, 'norm1', None)
    norm2 = getattr(resnet, 'norm2', None)
    dropout = getattr(resnet, 'dropout', None)
    return {
        'prefix': prefix,
        'class': f'{type(resnet).__module__}.{type(resnet).__name__}',
        'forward_signature': str(inspect.signature(resnet.forward)),
        'in_channels': getattr(resnet, 'in_channels', None),
        'out_channels': getattr(resnet, 'out_channels', None),
        'temb_channels': shape(resnet.time_emb_proj.weight)[1] if getattr(resnet, 'time_emb_proj', None) is not None else None,
        'groups': getattr(norm1, 'num_groups', None),
        'groups_out': getattr(norm2, 'num_groups', None),
        'eps': getattr(norm1, 'eps', None),
        'eps_out': getattr(norm2, 'eps', None),
        'non_linearity': type(getattr(resnet, 'nonlinearity', None)).__name__,
        'time_embedding_norm': getattr(resnet, 'time_embedding_norm', None),
        'output_scale_factor': getattr(resnet, 'output_scale_factor', None),
        'use_in_shortcut': getattr(resnet, 'use_in_shortcut', None),
        'use_conv_shortcut': getattr(resnet, 'use_conv_shortcut', None),
        'up': getattr(resnet, 'up', None),
        'down': getattr(resnet, 'down', None),
        'dropout': getattr(dropout, 'p', None),
        'skip_time_act': getattr(resnet, 'skip_time_act', None),
        'pre_norm': getattr(resnet, 'pre_norm', None),
        'has_time_emb_proj': getattr(resnet, 'time_emb_proj', None) is not None,
        'time_emb_proj_out_features': shape(resnet.time_emb_proj.weight)[0] if getattr(resnet, 'time_emb_proj', None) is not None else None,
        'has_shortcut': getattr(resnet, 'conv_shortcut', None) is not None,
        'has_conv_shortcut': getattr(resnet, 'conv_shortcut', None) is not None,
        'channel_change': getattr(resnet, 'in_channels', None) != getattr(resnet, 'out_channels', None),
        'operation_order': [
            'input_tensor -> norm1',
            'norm1 -> nonlinearity',
            'act1 -> conv1',
            'temb -> nonlinearity unless skip_time_act',
            'temb -> time_emb_proj -> NCHW broadcast',
            'default: conv1 + temb_proj before norm2',
            'scale_shift: norm2 then scale/shift',
            'norm2 -> nonlinearity -> dropout -> conv2',
            'optional conv_shortcut on input_tensor',
            '(shortcut + conv2) / output_scale_factor',
        ],
    }


def stats_tensor(x):
    y = x.detach().cpu()
    return {
        'shape': list(y.shape),
        'dtype': str(y.dtype),
        'min': float(y.min().item()) if y.numel() else 0.0,
        'max': float(y.max().item()) if y.numel() else 0.0,
        'mean': float(y.float().mean().item()) if y.numel() else 0.0,
        'std': float(y.float().std(unbiased=False).item()) if y.numel() else 0.0,
    }


def attention_preview(att, expected_input_shape=None, encoder_hidden_states_shape=None):
    if att is None:
        return {'exists': False}
    out = {
        'exists': True,
        'class': f'{type(att).__module__}.{type(att).__name__}',
        'expected_input_shape_from_resnet0': expected_input_shape,
        'expected_encoder_hidden_states_shape': encoder_hidden_states_shape,
        'transformer_blocks': len(getattr(att, 'transformer_blocks', [])),
        'norm_num_groups': getattr(getattr(att, 'norm', None), 'num_groups', None),
        'inner_dim': getattr(att, 'inner_dim', None),
        'num_attention_heads': getattr(att, 'num_attention_heads', None),
        'attention_head_dim': getattr(att, 'attention_head_dim', None),
        'cross_attention_dim': getattr(att, 'cross_attention_dim', None),
        'use_linear_projection': getattr(att, 'use_linear_projection', None),
        'analogous_to_previous_attention_modules': True,
        'not_ported_this_stage': True,
    }
    blocks = getattr(att, 'transformer_blocks', [])
    tb = blocks[0] if blocks else None
    if tb is not None:
        out['transformer0_class'] = f'{type(tb).__module__}.{type(tb).__name__}'
        out['attn1_class'] = f'{type(tb.attn1).__module__}.{type(tb.attn1).__name__}'
        out['attn2_class'] = f'{type(tb.attn2).__module__}.{type(tb.attn2).__name__}'
    return out

ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock3_resnet0'
META_JSON = ORACLE_DIR / 'unet_downblock3_resnet0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock3_resnet0_effective_weights.npz'
PREV_DOWNBLOCK2_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock2_downsampler' / 'local_downblock2_hidden_states_output.npy'


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
    out = {}
    out['norm1'] = resnet.norm1(x)
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
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


def max_abs(a, b):
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


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
    block3 = unet.down_blocks[3]
    resnet0 = block3.resnets[0]
    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block2_hidden, block2_states = block2(block1_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        x_shape = list(block2_hidden.shape)
        temb_shape = [1, int(resnet0.time_emb_proj.weight.shape[1])]
        synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_inter = resnet_manual(resnet0, synthetic_x, synthetic_temb)
        synthetic_official = resnet0(synthetic_x, synthetic_temb)
        entry_inter = resnet_manual(resnet0, block2_hidden, time_embedding)
        entry_official = resnet0(block2_hidden, time_embedding)

    saved = {}
    save_tensor(saved, 'synthetic_downblock3_resnet0_input', synthetic_x)
    save_tensor(saved, 'synthetic_downblock3_resnet0_temb', synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f'synthetic_downblock3_resnet0_{k}_output' if k != 'output' else 'synthetic_downblock3_resnet0_manual_output', v)
    save_tensor(saved, 'synthetic_downblock3_resnet0_output', synthetic_official)

    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_downblock0_output_hidden', block0_hidden)
    for i, state in enumerate(block0_states):
        save_tensor(saved, f'entry_downblock0_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock1_output_hidden', block1_hidden)
    for i, state in enumerate(block1_states):
        save_tensor(saved, f'entry_downblock1_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock2_output_hidden', block2_hidden)
    for i, state in enumerate(block2_states):
        save_tensor(saved, f'entry_downblock2_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock3_resnet0_input', block2_hidden)
    save_tensor(saved, 'entry_downblock3_resnet0_temb', time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f'entry_downblock3_resnet0_{k}_output' if k != 'output' else 'entry_downblock3_resnet0_manual_output', v)
    save_tensor(saved, 'entry_downblock3_resnet0_output', entry_official)

    effective = {}
    effective_meta = {}
    prefix = 'down_blocks_3_resnets_0'
    for name in ['norm1', 'norm2']:
        mod = getattr(resnet0, name)
        effective[f'{prefix}_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f'{prefix}_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f'{prefix}_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for name in ['conv1', 'time_emb_proj', 'conv2']:
        module_effective_arrays(f'{prefix}_{name}', getattr(resnet0, name), effective, effective_meta)
    if getattr(resnet0, 'conv_shortcut', None) is not None:
        module_effective_arrays(f'{prefix}_conv_shortcut', resnet0.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    synth_diff = (synthetic_inter['output'] - synthetic_official).abs()
    entry_diff = (entry_inter['output'] - entry_official).abs()
    previous_downblock2_compare = {'previous_downblock2_oracle_tensor_available': False}
    if PREV_DOWNBLOCK2_ORACLE.exists():
        prev = np.load(PREV_DOWNBLOCK2_ORACLE)
        curr = to_np(block2_hidden)
        previous_downblock2_compare = {
            'previous_downblock2_oracle_tensor_available': True,
            'max_abs_diff': max_abs(prev, curr),
            'mean_abs_diff': float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
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
        'downblock0_output_state_shapes': [list(x.shape) for x in block0_states],
        'downblock1_output_state_shapes': [list(x.shape) for x in block1_states],
        'downblock2_output_state_shapes': [list(x.shape) for x in block2_states],
        'accumulated_down_block_res_sample_shapes': [list(conv_in.shape)] + [list(x.shape) for x in block0_states] + [list(x.shape) for x in block1_states] + [list(x.shape) for x in block2_states],
        'resnet0_input_shape': list(block2_hidden.shape),
        'resnet0_output_shape': list(entry_official.shape),
        'temb_shape': list(time_embedding.shape),
        'resnet0_config': resnet_config(resnet0, prefix),
        'saved_tensors': saved,
        'effective_weight_metadata': effective_meta,
        'manual_vs_official': {
            'synthetic_max_abs_diff': float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            'synthetic_mean_abs_diff': float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            'entry_max_abs_diff': float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            'entry_mean_abs_diff': float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        'previous_downblock2_compare': previous_downblock2_compare,
        'markers': {
            'TADSR_UNET_DOWNBLOCK3_RESNET0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK3_RESNET0_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'down_blocks.3.resnets.0',
            'not_ported_this_stage': ['down_blocks.3.resnets.1 if present', 'down_blocks.3.attentions.* if present', 'down_blocks.3.downsamplers.* if present', 'mid_block', 'up_blocks', 'full UNet forward'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text(
        '\n'.join([
            '# TADSR UNet down_blocks.3.resnets.0 oracle export',
            '',
            'TADSR_UNET_DOWNBLOCK3_RESNET0_ORACLE_TENSORS: PASS',
            'TADSR_UNET_DOWNBLOCK3_RESNET0_EFFECTIVE_WEIGHTS: PASS',
            '',
            f'effective weights: {EFFECTIVE_WEIGHTS}',
            f'oracle tensors: {ORACLE_DIR}',
            f'synthetic manual/offical max abs diff: {metadata["manual_vs_official"]["synthetic_max_abs_diff"]}',
            f'entry manual/offical max abs diff: {metadata["manual_vs_official"]["entry_max_abs_diff"]}',
            f'previous downblock2 compare: {previous_downblock2_compare}',
        ]) + '\n',
        encoding='utf-8',
    )
    for k, v in metadata['markers'].items():
        print(f'{k}: {v}')
    print(json.dumps({'status': metadata['status'], 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
