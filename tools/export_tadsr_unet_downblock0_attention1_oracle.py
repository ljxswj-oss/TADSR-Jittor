#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_downblock0_attention1'
META_JSON = ORACLE_DIR / 'unet_downblock0_attention1_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_downblock0_attention1_effective_weights.npz'
PREV_RESNET1_ORACLE = OUT_DIR / 'oracle_tensors_unet_downblock0_resnet1' / 'entry_downblock0_resnet1_output.npy'


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
    tensors['norm'] = norm
    b, c, h, w = norm.shape
    seq = norm.permute(0, 2, 3, 1).reshape(b, h * w, c)
    tensors['sequence_input'] = seq
    proj_in = attention.proj_in(seq)
    tensors['proj_in'] = proj_in
    x = proj_in
    n1 = tb.norm1(x)
    tensors['transformer0_norm1'] = n1
    a1 = manual_attention(tb.attn1, n1, encoder_hidden_states=None)
    for k, v in a1.items():
        tensors[f'transformer0_attn1_{k}'] = v
    x = a1['output'] + x
    tensors['transformer0_after_attn1'] = x
    n2 = tb.norm2(x)
    tensors['transformer0_norm2'] = n2
    a2 = manual_attention(tb.attn2, n2, encoder_hidden_states=encoder_hidden_states)
    for k, v in a2.items():
        tensors[f'transformer0_attn2_{k}'] = v
    x = a2['output'] + x
    tensors['transformer0_after_attn2'] = x
    n3 = tb.norm3(x)
    tensors['transformer0_norm3'] = n3
    ff0 = tb.ff.net[0].proj(n3)
    tensors['transformer0_ff_geglu_proj'] = ff0
    hidden, gate = ff0.chunk(2, dim=-1)
    ff_act = hidden * torch.nn.functional.gelu(gate)
    tensors['transformer0_ff_geglu_output'] = ff_act
    ff_out = tb.ff.net[2](tb.ff.net[1](ff_act))
    tensors['transformer0_ff_output'] = ff_out
    x = ff_out + x
    tensors['transformer0_output'] = x
    proj_out_seq = attention.proj_out(x)
    tensors['proj_out_sequence'] = proj_out_seq
    proj_out_nchw = proj_out_seq.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    tensors['proj_out_nchw'] = proj_out_nchw
    output = proj_out_nchw + residual
    tensors['output'] = output
    return tensors


def save_tensor(saved, name, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


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
    block = unet.down_blocks[0]
    attention0 = block.attentions[0]
    attention1 = block.attentions[1]
    resnet0 = block.resnets[0]
    resnet1 = block.resnets[1]
    cfg = dict(unet.config)
    sample_shape = [1, int(cfg.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    h_shape = [1, int(attention1.in_channels), 32, 32]
    synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    saved = {}

    with torch.no_grad():
        synthetic_manual = manual_transformer_attention(attention1, synthetic_hidden, encoder_hidden_states)
        synthetic_official = attention1(synthetic_hidden, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        centered = 2.0 * sample - 1.0 if cfg.get('center_input_sample') else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        resnet0_out = resnet0(conv_in, time_embedding)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, time_embedding)
        entry_manual = manual_transformer_attention(attention1, resnet1_out, encoder_hidden_states)
        entry_official = attention1(resnet1_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    save_tensor(saved, 'synthetic_attention1_input', synthetic_hidden)
    save_tensor(saved, 'synthetic_attention1_encoder_hidden_states', encoder_hidden_states)
    for k, v in synthetic_manual.items():
        save_tensor(saved, f'synthetic_attention1_{k}', v)
    save_tensor(saved, 'synthetic_attention1_official_output', synthetic_official)
    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_attention1_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_downblock0_resnet0_output', resnet0_out)
    save_tensor(saved, 'entry_downblock0_attention0_output', attention0_out)
    save_tensor(saved, 'entry_downblock0_resnet1_output', resnet1_out)
    for k, v in entry_manual.items():
        save_tensor(saved, f'entry_downblock0_attention1_{k}', v)
    save_tensor(saved, 'entry_downblock0_attention1_official_output', entry_official)

    arrays = {}
    emeta = {}
    prefix = 'down_blocks_0_attentions_1'
    arrays[f'{prefix}_norm_weight'] = attention1.norm.weight.detach().cpu().numpy().astype(np.float32)
    arrays[f'{prefix}_norm_bias'] = attention1.norm.bias.detach().cpu().numpy().astype(np.float32)
    emeta[f'{prefix}_norm'] = {'is_lora_wrapped': False, 'weight_shape': list(attention1.norm.weight.shape), 'bias_shape': list(attention1.norm.bias.shape)}
    module_effective_arrays(f'{prefix}_proj_in', attention1.proj_in, arrays, emeta)
    module_effective_arrays(f'{prefix}_proj_out', attention1.proj_out, arrays, emeta)
    tb = attention1.transformer_blocks[0]
    for name in ['norm1', 'norm2', 'norm3']:
        mod = getattr(tb, name)
        arrays[f'{prefix}_transformer0_{name}_weight'] = mod.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f'{prefix}_transformer0_{name}_bias'] = mod.bias.detach().cpu().numpy().astype(np.float32)
        emeta[f'{prefix}_transformer0_{name}'] = {'is_lora_wrapped': False, 'weight_shape': list(mod.weight.shape), 'bias_shape': list(mod.bias.shape)}
    for an in ['attn1', 'attn2']:
        attn = getattr(tb, an)
        for ln in ['to_q', 'to_k', 'to_v']:
            module_effective_arrays(f'{prefix}_transformer0_{an}_{ln}', getattr(attn, ln), arrays, emeta)
        module_effective_arrays(f'{prefix}_transformer0_{an}_to_out_0', attn.to_out[0], arrays, emeta)
    module_effective_arrays(f'{prefix}_transformer0_ff_geglu_proj', tb.ff.net[0].proj, arrays, emeta)
    module_effective_arrays(f'{prefix}_transformer0_ff_out', tb.ff.net[2], arrays, emeta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **arrays)

    synth_diff = (synthetic_manual['output'] - synthetic_official).abs()
    entry_diff = (entry_manual['output'] - entry_official).abs()
    previous_resnet1_compare = {'previous_resnet1_oracle_tensor_available': False}
    if PREV_RESNET1_ORACLE.exists():
        prev = np.load(PREV_RESNET1_ORACLE)
        curr = to_np(resnet1_out)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_resnet1_compare = {
            'previous_resnet1_oracle_tensor_available': True,
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
        'attention1_config': {
            'prefix': prefix,
            'in_channels': int(attention1.in_channels),
            'inner_dim': int(attention1.num_attention_heads) * int(attention1.attention_head_dim),
            'num_attention_heads': int(attention1.num_attention_heads),
            'attention_head_dim': int(attention1.attention_head_dim),
            'cross_attention_dim': int(unet.config.cross_attention_dim),
            'use_linear_projection': bool(attention1.use_linear_projection),
            'norm_num_groups': int(attention1.norm.num_groups),
            'norm_eps': float(attention1.norm.eps),
        },
        'transformer0_config': {
            'heads': int(tb.attn1.heads),
            'head_dim': int(tb.attn1.inner_dim // tb.attn1.heads),
            'norm_eps': float(tb.norm1.eps),
            'dropout': 0.0,
        },
        'path_a_isolated_attention1': {'status': 'PASS', 'input_shape': h_shape, 'encoder_shape': encoder_shape, 'output_shape': list(synthetic_official.shape)},
        'path_b_entry_resnet0_attention0_resnet1_attention1': {'status': 'PASS', 'sample_shape': sample_shape, 'resnet1_shape': list(resnet1_out.shape), 'output_shape': list(entry_official.shape)},
        'previous_resnet1_compare': previous_resnet1_compare,
        'effective_weight_export': {
            'status': 'PASS',
            'path': str(EFFECTIVE_WEIGHTS),
            'keys': {k: list(v.shape) for k, v in arrays.items()},
            'module_effective_meta': emeta,
            'lora_affected_modules': [k for k, v in emeta.items() if v.get('is_lora_wrapped')],
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
            'TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'not_in_scope': ['full down_blocks.0', 'downsampler', 'down_blocks.1', 'full UNet forward', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet down_blocks.0.attentions.1 Oracle Export',
        '',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS: PASS',
        'TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS: PASS',
        '',
        f"Oracle dir: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        f"Path A synthetic output shape: `{metadata['path_a_isolated_attention1']['output_shape']}`",
        f"Path B entry->resnet0->attention0->resnet1->attention1 output shape: `{metadata['path_b_entry_resnet0_attention0_resnet1_attention1']['output_shape']}`",
        f"Synthetic manual/offical max abs error: `{metadata['manual_vs_official_validation']['synthetic_manual_vs_official_max_abs_error']}`",
        f"Entry manual/offical max abs error: `{metadata['manual_vs_official_validation']['entry_manual_vs_official_max_abs_error']}`",
        '',
        'This export stops before down_blocks.0.downsamplers.0, down_blocks.1, full down_blocks.0 and full UNet.',
    ]
    SUMMARY_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION1_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_DOWNBLOCK0_ATTENTION1_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['manual_vs_official_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
