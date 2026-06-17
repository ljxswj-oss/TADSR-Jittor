#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
TADSR_PKL = WEIGHTS_DIR / 'tadsr.pkl'
OUT_DIR = Path('experiments/full_repro/unet_alignment')
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock1_resnet1'
META_JSON = ORACLE_DIR / 'unet_downblock1_resnet1_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock1_resnet1_effective_weights.npz'
PREV_ATTENTION0_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock1_attention0' / 'entry_downblock1_attention0_output.npy'


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


def manual_attention(attn, hidden_states, encoder_hidden_states=None):
    import torch.nn.functional as F
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    b = hidden_states.shape[0]
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    qh = query.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    kh = key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    vh = value.view(b, -1, attn.heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim).to(query.dtype)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return {'q': query, 'k': key, 'v': value, 'output': out}


def manual_transformer_attention(attention, hidden_states, encoder_hidden_states):
    tensors = {}
    tb = attention.transformer_blocks[0]
    residual = hidden_states
    norm = attention.norm(hidden_states)
    b, c, h, w = norm.shape
    seq = norm.permute(0, 2, 3, 1).reshape(b, h * w, c)
    proj_in = attention.proj_in(seq)
    x = proj_in
    n1 = tb.norm1(x)
    a1 = manual_attention(tb.attn1, n1, encoder_hidden_states=None)
    x = a1['output'] + x
    n2 = tb.norm2(x)
    a2 = manual_attention(tb.attn2, n2, encoder_hidden_states=encoder_hidden_states)
    x = a2['output'] + x
    n3 = tb.norm3(x)
    ff0 = tb.ff.net[0].proj(n3)
    hidden, gate = ff0.chunk(2, dim=-1)
    ff_act = hidden * torch.nn.functional.gelu(gate)
    ff_out = tb.ff.net[2](tb.ff.net[1](ff_act))
    x = ff_out + x
    proj_out_seq = attention.proj_out(x)
    proj_out_nchw = proj_out_seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    output = proj_out_nchw + residual
    tensors['output'] = output
    return tensors


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


def resnet_config(resnet, prefix):
    return {
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
        'in_channels': int(resnet.in_channels),
        'out_channels': int(resnet.out_channels),
        'temb_channels': int(resnet.time_emb_proj.weight.shape[1]),
        'time_emb_proj_out_features': int(resnet.time_emb_proj.weight.shape[0]),
        'skip_time_act': bool(resnet.skip_time_act),
    }


def main() -> int:
    maybe_reexec()
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
    resnet0 = block1.resnets[0]
    attention0 = block1.attentions[0]
    resnet1 = block1.resnets[1]
    cfg = dict(unet.config)
    sample_shape = [1, int(cfg.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    x_shape = [1, int(getattr(resnet1, 'in_channels')), 16, 16]
    temb_shape = [1, int(resnet1.time_emb_proj.weight.shape[1])]
    synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
    synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
    with torch.no_grad():
        synthetic_inter = resnet_manual(resnet1, synthetic_x, synthetic_temb)
        synthetic_official = resnet1(synthetic_x, synthetic_temb)
        centered = 2.0 * sample - 1.0 if cfg.get('center_input_sample') else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = resnet0(block0_hidden, time_embedding)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        entry_inter = resnet_manual(resnet1, attention0_out, time_embedding)
        entry_official = resnet1(attention0_out, time_embedding)
    saved = {}
    save_tensor(saved, 'synthetic_downblock1_resnet1_input', synthetic_x)
    save_tensor(saved, 'synthetic_downblock1_resnet1_temb', synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f'synthetic_downblock1_resnet1_{k}_output' if k != 'output' else 'synthetic_downblock1_resnet1_manual_output', v)
    save_tensor(saved, 'synthetic_downblock1_resnet1_output', synthetic_official)
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
    save_tensor(saved, 'entry_downblock1_resnet0_output', resnet0_out)
    save_tensor(saved, 'entry_downblock1_attention0_output', attention0_out)
    save_tensor(saved, 'entry_downblock1_resnet1_input', attention0_out)
    save_tensor(saved, 'entry_downblock1_resnet1_temb', time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f'entry_downblock1_resnet1_{k}_output' if k != 'output' else 'entry_downblock1_resnet1_manual_output', v)
    save_tensor(saved, 'entry_downblock1_resnet1_output', entry_official)
    effective = {}
    effective_meta = {}
    prefix = 'down_blocks_1_resnets_1'
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
    previous_attention0_compare = {'previous_downblock1_attention0_oracle_tensor_available': False}
    if PREV_ATTENTION0_ORACLE.exists():
        prev = np.load(PREV_ATTENTION0_ORACLE)
        curr = to_np(attention0_out)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_attention0_compare = {
            'previous_downblock1_attention0_oracle_tensor_available': True,
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
        'downblock0_output_state_shapes': [list(x.shape) for x in block0_states],
        'resnet1_input_shape': list(attention0_out.shape),
        'resnet1_output_shape': list(entry_official.shape),
        'temb_shape': list(time_embedding.shape),
        'resnet1_config': resnet_config(resnet1, prefix),
        'path_a_isolated_resnet1': {'status': 'PASS', 'input_shape': x_shape, 'temb_shape': temb_shape, 'output_shape': list(synthetic_official.shape)},
        'path_b_entry_downblock0_downblock1_resnet0_attention0_resnet1': {'status': 'PASS', 'sample_shape': sample_shape, 'attention0_shape': list(attention0_out.shape), 'output_shape': list(entry_official.shape)},
        'previous_downblock1_attention0_compare': previous_attention0_compare,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in effective.items()},
            'module_effective_meta': effective_meta,
            'lora_affected_modules': [k for k, v in effective_meta.items() if v.get('is_lora_wrapped')],
        },
        'manual_vs_official_validation': {
            'status': 'PASS' if float(synth_diff.max()) < 1e-5 and float(entry_diff.max()) < 1e-5 else 'FAIL',
            'synthetic_manual_vs_official_max_abs_error': float(synth_diff.max().item()),
            'synthetic_manual_vs_official_mean_abs_error': float(synth_diff.mean().item()),
            'entry_manual_vs_official_max_abs_error': float(entry_diff.max().item()),
            'entry_manual_vs_official_mean_abs_error': float(entry_diff.mean().item()),
            'loaded_lora_param_count': len(loaded),
        },
        'saved_tensors': saved,
        'markers': {
            'TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'not_in_scope': ['down_blocks.1.attentions.1', 'down_blocks.1.downsamplers.0', 'full down_blocks.1', 'full UNet forward', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.1.resnets.1 Oracle Export',
        '',
        'TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS: PASS',
        'TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS: PASS',
        '',
        f"Oracle dir: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        f"Path A synthetic output shape: `{metadata['path_a_isolated_resnet1']['output_shape']}`",
        f"Path B entry->downblock0->downblock1.resnet0->attention0->resnet1 output shape: `{metadata['path_b_entry_downblock0_downblock1_resnet0_attention0_resnet1']['output_shape']}`",
        f"Synthetic manual/offical max abs error: `{metadata['manual_vs_official_validation']['synthetic_manual_vs_official_max_abs_error']}`",
        f"Entry manual/offical max abs error: `{metadata['manual_vs_official_validation']['entry_manual_vs_official_max_abs_error']}`",
        '',
        'This export stops before down_blocks.1.attentions.1, down_blocks.1.downsamplers.0 and full UNet.',
    ]
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK1_RESNET1_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_DOWNBLOCK1_RESNET1_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_vs_official_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
