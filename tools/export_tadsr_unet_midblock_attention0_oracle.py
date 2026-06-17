#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT_DIR = Path('experiments/full_repro/unet_alignment')
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_midblock_attention0'
META_JSON = ORACLE_DIR / 'unet_midblock_attention0_oracle_metadata.json'
SUMMARY_TXT = ORACLE_DIR / 'oracle_summary.txt'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_midblock_attention0_effective_weights.npz'
PREV_MID_RESNET0_ORACLE = OUT_DIR / 'oracle_tensors_unet_midblock_resnet0' / 'entry_midblock_resnet0_output.npy'


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
    return {'shape': list(y.shape), 'dtype': str(y.dtype), 'min': float(yf.min()) if y.size else 0.0, 'max': float(yf.max()) if y.size else 0.0, 'mean': float(yf.mean()) if y.size else 0.0, 'std': float(yf.std()) if y.size else 0.0, 'nbytes': int(y.nbytes)}


def save_tensor(saved, name, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f'{name}.npy', arr)
    saved[name] = stats(arr)
    return arr


def main() -> int:
    maybe_reexec()
    from tools.export_tadsr_unet_downblock2_attention1_oracle import (
        load_unet,
        manual_transformer_attention,
        module_effective_arrays,
    )
    from tools.audit_official_tadsr_unet_downblock2_attention1 import attention_config

    torch.manual_seed(1234)
    np.random.seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    attention0 = unet.mid_block.attentions[0]
    mid_resnet0 = unet.mid_block.resnets[0]
    if len(attention0.transformer_blocks) != 1:
        raise RuntimeError(f'Expected one transformer block, got {len(attention0.transformer_blocks)}')

    cfg_unet = dict(unet.config)
    sample_shape = [1, int(cfg_unet.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)
    saved = {}
    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if cfg_unet.get('center_input_sample') else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = unet.down_blocks[0](conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = unet.down_blocks[1](block0_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block2_hidden, block2_states = unet.down_blocks[2](block1_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block3_hidden, block3_states = unet.down_blocks[3](block2_hidden, time_embedding)
        mid0 = mid_resnet0(block3_hidden, time_embedding)
        h_shape = list(mid0.shape)
        synthetic_hidden = torch.linspace(-1.0, 1.0, steps=int(np.prod(h_shape)), dtype=torch.float32).reshape(*h_shape)
        synthetic_manual = manual_transformer_attention(attention0, synthetic_hidden, encoder_hidden_states)
        synthetic_official = attention0(synthetic_hidden, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        entry_manual = manual_transformer_attention(attention0, mid0, encoder_hidden_states)
        entry_official = attention0(mid0, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

    save_tensor(saved, 'synthetic_midblock_attention0_input', synthetic_hidden)
    save_tensor(saved, 'synthetic_midblock_attention0_encoder_hidden_states', encoder_hidden_states)
    for k, v in synthetic_manual.items():
        save_tensor(saved, f'synthetic_midblock_attention0_{k}', v)
    save_tensor(saved, 'synthetic_midblock_attention0_official_output', synthetic_official)

    save_tensor(saved, 'entry_synthetic_unet_sample', sample)
    save_tensor(saved, 'entry_synthetic_unet_timestep', timestep)
    save_tensor(saved, 'entry_synthetic_unet_sample_after_center', centered)
    save_tensor(saved, 'entry_synthetic_unet_conv_in_output', conv_in)
    save_tensor(saved, 'entry_synthetic_unet_time_proj_output', time_proj)
    save_tensor(saved, 'entry_synthetic_unet_time_embedding_output', time_embedding)
    save_tensor(saved, 'entry_encoder_hidden_states', encoder_hidden_states)
    save_tensor(saved, 'entry_downblock0_output_hidden', block0_hidden)
    for i, state in enumerate(block0_states):
        save_tensor(saved, f'entry_downblock0_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock1_output_hidden', block1_hidden)
    for i, state in enumerate(block1_states):
        save_tensor(saved, f'entry_downblock1_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock2_output_hidden', block2_hidden)
    for i, state in enumerate(block2_states):
        save_tensor(saved, f'entry_downblock2_output_state_{i}', state)
    save_tensor(saved, 'entry_downblock3_output_hidden', block3_hidden)
    for i, state in enumerate(block3_states):
        save_tensor(saved, f'entry_downblock3_output_state_{i}', state)
    save_tensor(saved, 'entry_midblock_resnet0_output', mid0)
    for k, v in entry_manual.items():
        save_tensor(saved, f'entry_midblock_attention0_{k}', v)
    save_tensor(saved, 'entry_midblock_attention0_official_output', entry_official)

    arrays = {}
    emeta = {}
    prefix = 'mid_block_attentions_0'
    arrays[f'{prefix}_norm_weight'] = attention0.norm.weight.detach().cpu().numpy().astype(np.float32)
    arrays[f'{prefix}_norm_bias'] = attention0.norm.bias.detach().cpu().numpy().astype(np.float32)
    emeta[f'{prefix}_norm'] = {'is_lora_wrapped': False, 'weight_shape': list(attention0.norm.weight.shape), 'bias_shape': list(attention0.norm.bias.shape)}
    module_effective_arrays(f'{prefix}_proj_in', attention0.proj_in, arrays, emeta)
    module_effective_arrays(f'{prefix}_proj_out', attention0.proj_out, arrays, emeta)
    tb = attention0.transformer_blocks[0]
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

    cfg = attention_config(attention0, prefix, 'unet.mid_block.attentions.0')
    heads = int(getattr(tb.attn1, 'heads', 0) or cfg.get('num_attention_heads') or 20)
    inner_dim = int(getattr(tb.attn1, 'inner_dim', 0) or cfg.get('in_channels') or 1280)
    head_dim = inner_dim // heads
    cfg['num_attention_heads'] = heads
    cfg['attention_head_dim'] = head_dim
    cfg['inner_dim'] = inner_dim
    cfg['cross_attention_dim'] = int(getattr(tb.attn2, 'cross_attention_dim', 0) or unet.config.cross_attention_dim)
    transformer0_config = {'heads': heads, 'head_dim': head_dim, 'inner_dim': inner_dim, 'norm_eps': float(getattr(tb.norm1, 'eps', 1e-5)), 'dropout': 0.0, 'transformer_block_count': 1}

    synth_diff = (synthetic_manual['output'] - synthetic_official).abs()
    entry_diff = (entry_manual['output'] - entry_official).abs()
    previous_mid_resnet0_compare = {'previous_midblock_resnet0_oracle_tensor_available': False}
    if PREV_MID_RESNET0_ORACLE.exists():
        prev = np.load(PREV_MID_RESNET0_ORACLE)
        curr = to_np(mid0)
        diff = np.abs(prev.astype(np.float32) - curr.astype(np.float32))
        previous_mid_resnet0_compare = {'previous_midblock_resnet0_oracle_tensor_available': True, 'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean())}

    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'selected_timestep': to_np(timestep).astype(int).tolist(),
        'entry_sample_shape': sample_shape,
        'encoder_hidden_states_shape': encoder_shape,
        'downblock0_output_state_shapes': [list(x.shape) for x in block0_states],
        'downblock1_output_state_shapes': [list(x.shape) for x in block1_states],
        'downblock2_output_state_shapes': [list(x.shape) for x in block2_states],
        'downblock3_output_state_shapes': [list(x.shape) for x in block3_states],
        'accumulated_down_block_res_sample_shapes': [[1, 320, 32, 32], *[list(x.shape) for x in block0_states], *[list(x.shape) for x in block1_states], *[list(x.shape) for x in block2_states], *[list(x.shape) for x in block3_states]],
        'accumulated_down_block_res_sample_count': 12,
        'midblock_attention0_input_shape': h_shape,
        'midblock_attention0_output_shape': list(entry_manual['output'].shape),
        'midblock_attention0_consumes': ['hidden_states', 'encoder_hidden_states'],
        'midblock_attention0_consumes_accumulated_residuals': False,
        'attention_config': cfg,
        'attention0_config': cfg,
        'transformer0_config': transformer0_config,
        'effective_weight_metadata': emeta,
        'saved_tensors': saved,
        'synthetic_manual_official_max_abs_diff': float(synth_diff.max().item()),
        'entry_manual_official_max_abs_diff': float(entry_diff.max().item()),
        'previous_midblock_resnet0_compare': previous_mid_resnet0_compare,
        'markers': {
            'TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS': 'PASS',
            'TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS': 'PASS',
        },
        'scope_boundaries': {
            'ported_this_stage': 'mid_block.attentions.0 only',
            'not_ported_this_stage': ['mid_block.resnets.1', 'up_blocks', 'full UNet forward', 'full TADSR inference'],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    SUMMARY_TXT.write_text('\n'.join([
        '# TADSR UNet mid_block.attentions.0 oracle export',
        '',
        'TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS: PASS',
        'TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS: PASS',
        '',
        f'effective weights: {EFFECTIVE_WEIGHTS}',
        f'oracle tensors: {ORACLE_DIR}',
        f'synthetic manual/offical max abs diff: {float(synth_diff.max().item())}',
        f'entry manual/offical max abs diff: {float(entry_diff.max().item())}',
        f'previous midblock resnet0 compare: {previous_mid_resnet0_compare}',
    ]) + '\n', encoding='utf-8')
    print('TADSR_UNET_MIDBLOCK_ATTENTION0_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_MIDBLOCK_ATTENTION0_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
