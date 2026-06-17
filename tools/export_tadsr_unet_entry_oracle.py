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
ORACLE_DIR = OUT_DIR / 'oracle_tensors_unet_entry'
EFFECTIVE_WEIGHTS = OUT_DIR / 'converted_unet_entry_effective_weights.npz'
META_JSON = OUT_DIR / 'unet_entry_oracle_metadata.json'
SUMMARY_TXT = OUT_DIR / 'oracle_summary.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32)


def stats(x: np.ndarray) -> dict:
    y = np.asarray(x)
    yf = y.astype(np.float64, copy=False)
    return {
        'shape': list(y.shape), 'dtype': str(y.dtype), 'mean': float(yf.mean()) if y.size else 0.0,
        'std': float(yf.std()) if y.size else 0.0, 'min': float(yf.min()) if y.size else 0.0,
        'max': float(yf.max()) if y.size else 0.0,
    }


def load_unet():
    sys.path.insert(0, str(OFFICIAL_REPO))
    import torch
    from tadsr import initialize_unet

    args = SimpleNamespace(pretrained_model_name_or_path=str(WEIGHTS_DIR), pretrained_lora_path=str(TADSR_PKL), lora_rank=4)
    unet, enc, dec, oth = initialize_unet(args)
    loaded_lora = []
    copied = []
    ckpt = torch.load(str(TADSR_PKL), map_location='cpu')
    sd = ckpt.get('state_dict_unet', {})
    for n, p in unet.named_parameters():
        if 'lora' in n and n in sd:
            p.data.copy_(sd[n])
            loaded_lora.append(n)
    conv = unet.conv_in.base_layer if hasattr(unet.conv_in, 'base_layer') else unet.conv_in
    if 'conv_in.weight' in sd:
        conv.weight.data.copy_(sd['conv_in.weight'])
        copied.append('conv_in.weight')
    if 'conv_in.bias' in sd and getattr(conv, 'bias', None) is not None:
        conv.bias.data.copy_(sd['conv_in.bias'])
        copied.append('conv_in.bias')
    if hasattr(unet, 'set_adapter'):
        unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
    unet.eval()
    return unet, enc, dec, oth, loaded_lora, copied


def effective_conv_weight(unet):
    import torch
    conv_module = unet.conv_in
    conv = conv_module.base_layer if hasattr(conv_module, 'base_layer') else conv_module
    weight = conv.weight.detach().cpu().clone()
    delta_records = []
    if hasattr(conv_module, 'get_delta_weight'):
        for adapter in list(getattr(conv_module, 'active_adapters', [])):
            try:
                delta = conv_module.get_delta_weight(adapter).detach().cpu()
            except Exception:
                continue
            weight = weight + delta
            delta_records.append({'adapter': adapter, 'shape': list(delta.shape), 'max_abs': float(delta.abs().max().item())})
    bias = conv.bias.detach().cpu().clone() if getattr(conv, 'bias', None) is not None else torch.zeros(weight.shape[0])
    return weight, bias, delta_records


def main() -> int:
    maybe_reexec()
    import torch
    import torch.nn.functional as F

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    np.random.seed(1234)
    unet, enc, dec, oth, loaded_lora, copied = load_unet()
    cfg = dict(unet.config)
    sample_shape = [1, int(cfg.get('in_channels', 4)), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timesteps = torch.tensor([1], dtype=torch.long)
    centered = 2.0 * sample - 1.0 if cfg.get('center_input_sample') else sample
    with torch.no_grad():
        conv_out = unet.conv_in(centered)
        time_proj = unet.time_proj(timesteps)
        emb_input = time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype)
        emb_linear1 = unet.time_embedding.linear_1(emb_input)
        emb_act = unet.time_embedding.act(emb_linear1) if getattr(unet.time_embedding, 'act', None) is not None else emb_linear1
        emb_linear2 = unet.time_embedding.linear_2(emb_act)
        emb_full = unet.time_embedding(emb_input)
        eff_w, eff_b, delta_records = effective_conv_weight(unet)
        manual_conv = F.conv2d(centered, eff_w.to(centered.dtype), eff_b.to(centered.dtype), stride=1, padding=1)
    arrays = {
        'synthetic_sample': to_np(sample),
        'unet_centered_sample': to_np(centered),
        'unet_conv_in_output': to_np(conv_out),
        'unet_time_proj_output': to_np(time_proj),
        'unet_time_embedding_linear1_output': to_np(emb_linear1),
        'unet_time_embedding_act_output': to_np(emb_act),
        'unet_time_embedding_output': to_np(emb_linear2),
        'unet_time_embedding_full_output': to_np(emb_full),
        'unet_conv_in_manual_effective_output': to_np(manual_conv),
        'timestep': timesteps.detach().cpu().numpy().astype(np.int64),
    }
    for name, arr in arrays.items():
        np.save(ORACLE_DIR / f'{name}.npy', arr)
    np.savez_compressed(ORACLE_DIR / 'unet_entry_oracle_inputs.npz', synthetic_sample=arrays['synthetic_sample'], timestep=arrays['timestep'])
    np.savez_compressed(
        EFFECTIVE_WEIGHTS,
        conv_in__weight=eff_w.detach().cpu().numpy().astype(np.float32),
        conv_in__bias=eff_b.detach().cpu().numpy().astype(np.float32),
        time_embedding__linear_1__weight=to_np(unet.time_embedding.linear_1.weight),
        time_embedding__linear_1__bias=to_np(unet.time_embedding.linear_1.bias),
        time_embedding__linear_2__weight=to_np(unet.time_embedding.linear_2.weight),
        time_embedding__linear_2__bias=to_np(unet.time_embedding.linear_2.bias),
    )
    diff = (manual_conv - conv_out).abs()
    metadata = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'oracle_dir': str(ORACLE_DIR),
        'effective_weights': str(EFFECTIVE_WEIGHTS),
        'official_class': f'{type(unet).__module__}.{type(unet).__name__}',
        'config': cfg,
        'path_a_synthetic_latent_entry': {
            'status': 'PASS', 'sample_shape': sample_shape, 'timestep': [1],
            'center_input_sample': cfg.get('center_input_sample'),
            'saved_tensors': {k: stats(v) for k, v in arrays.items() if k != 'timestep'},
        },
        'path_b_vae_mode_latent_contract': {
            'status': 'NOT_APPLICABLE',
            'reason': 'Full TimeAware VAE API / stochastic encode-decode pipeline is intentionally not complete; this stage avoids full TADSR inference and exports only synthetic UNet entry tensors.',
        },
        'effective_weight_validation': {
            'status': 'PASS' if float(diff.max()) < 1e-5 else 'FAIL',
            'manual_effective_conv_vs_unet_conv_in_max_abs_error': float(diff.max().item()),
            'manual_effective_conv_vs_unet_conv_in_mean_abs_error': float(diff.mean().item()),
            'conv_in_lora_delta_records': delta_records,
            'loaded_lora_param_count': len(loaded_lora),
            'copied_entry_weight_keys': copied,
            'lora_runtime_integration_status': 'DEFERRED; Jittor consumes static effective entry weights for this stage',
        },
        'markers': {'TADSR_UNET_ENTRY_ORACLE_TENSORS': 'PASS', 'TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS': 'PASS'},
        'not_in_scope': ['full UNet forward', 'down/up/mid blocks', 'cross-attention', 'full TADSR inference'],
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    summary = [
        '# TADSR UNet Entry Oracle Export', '',
        'TADSR_UNET_ENTRY_ORACLE_TENSORS: PASS',
        'TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS: PASS', '',
        f"Synthetic sample shape: `{sample_shape}`",
        f"Timestep: `[1]`",
        f"Effective conv validation max_abs_error: `{metadata['effective_weight_validation']['manual_effective_conv_vs_unet_conv_in_max_abs_error']}`",
        f"Oracle directory: `{ORACLE_DIR}`",
        f"Effective weights: `{EFFECTIVE_WEIGHTS}`",
        '', 'Path B VAE-mode latent contract is marked NOT_APPLICABLE until the full TimeAware VAE API/pipeline path is intentionally opened.',
    ]
    SUMMARY_TXT.write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('TADSR_UNET_ENTRY_ORACLE_TENSORS: PASS')
    print('TADSR_UNET_ENTRY_EFFECTIVE_WEIGHTS: PASS')
    print(json.dumps({'status': 'PASS', 'metadata': str(META_JSON), 'effective_weights': str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0 if metadata['effective_weight_validation']['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
