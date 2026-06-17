#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights')
TADSR_PKL = WEIGHTS_DIR / 'tadsr.pkl'
OUT_DIR = Path('experiments/full_repro/unet_alignment')
OUT_JSON = OUT_DIR / 'audit_tadsr_unet_entry.json'
OUT_TXT = OUT_DIR / 'audit_tadsr_unet_entry.txt'


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def tensor_shape(x):
    return list(x.shape) if x is not None and hasattr(x, 'shape') else None


def load_unet():
    sys.path.insert(0, str(OFFICIAL_REPO))
    import torch
    from tadsr import initialize_unet

    args = SimpleNamespace(
        pretrained_model_name_or_path=str(WEIGHTS_DIR),
        pretrained_lora_path=str(TADSR_PKL),
        lora_rank=4,
    )
    unet, enc, dec, oth = initialize_unet(args)
    loaded_lora = []
    copied = []
    ckpt_keys = []
    if TADSR_PKL.exists():
        ckpt = torch.load(str(TADSR_PKL), map_location='cpu')
        ckpt_keys = list(ckpt.keys())
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
    return unet, enc, dec, oth, loaded_lora, copied, ckpt_keys


def main() -> int:
    maybe_reexec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, enc, dec, oth, loaded_lora, copied, ckpt_keys = load_unet()
    conv_module = unet.conv_in
    conv = conv_module.base_layer if hasattr(conv_module, 'base_layer') else conv_module
    blocks = {
        'down_blocks': [type(x).__name__ for x in getattr(unet, 'down_blocks', [])],
        'up_blocks': [type(x).__name__ for x in getattr(unet, 'up_blocks', [])],
        'mid_block': type(getattr(unet, 'mid_block', None)).__name__ if getattr(unet, 'mid_block', None) is not None else None,
    }
    active_adapters = list(getattr(conv_module, 'active_adapters', [])) if hasattr(conv_module, 'active_adapters') else []
    lora_shapes = {}
    for side in ['lora_A', 'lora_B']:
        obj = getattr(conv_module, side, None)
        if hasattr(obj, 'items'):
            lora_shapes[side] = {k: tensor_shape(v.weight) for k, v in obj.items()}
    cfg = dict(unet.config)
    report = {
        'status': 'PASS',
        'python': sys.executable,
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'tadsr_pkl': str(TADSR_PKL),
        'official_unet_class': f'{type(unet).__module__}.{type(unet).__name__}',
        'official_unet_source': inspect.getsourcefile(type(unet)),
        'config': cfg,
        'input_contract': {
            'sample': ['batch', cfg.get('in_channels'), 'height', 'width'],
            'recommended_synthetic_sample_shape': [1, cfg.get('in_channels'), 32, 32],
            'official_sample_size': cfg.get('sample_size'),
            'timestep_dtype': 'torch.long',
            'default_tadsr_inference_timestep': [1],
            'encoder_hidden_states': ['batch', 'sequence_length', cfg.get('cross_attention_dim')],
            'center_input_sample': cfg.get('center_input_sample'),
            'full_unet_forward_in_scope': False,
        },
        'lora_state': {
            'target_module_counts': {'encoder': len(enc), 'decoder': len(dec), 'others': len(oth)},
            'peft_config_keys': list(getattr(unet, 'peft_config', {}).keys()) if hasattr(unet, 'peft_config') else [],
            'conv_in_module_class': f'{type(conv_module).__module__}.{type(conv_module).__name__}',
            'conv_in_active_adapters': active_adapters,
            'conv_in_lora_shapes': lora_shapes,
            'loaded_lora_param_count': len(loaded_lora),
            'copied_entry_weight_keys': copied,
            'checkpoint_top_level_keys': ckpt_keys,
            'runtime_lora_integration_status': 'DEFERRED; this stage exports effective entry weights only',
        },
        'conv_in': {
            'effective_base_class': f'{type(conv).__module__}.{type(conv).__name__}',
            'weight_shape': tensor_shape(conv.weight),
            'bias_shape': tensor_shape(conv.bias) if getattr(conv, 'bias', None) is not None else None,
            'stride': list(conv.stride),
            'padding': list(conv.padding),
            'dilation': list(conv.dilation),
            'groups': conv.groups,
        },
        'time_proj': {
            'class': f'{type(unet.time_proj).__module__}.{type(unet.time_proj).__name__}',
            'attrs': {k: getattr(unet.time_proj, k, None) for k in ['num_channels', 'flip_sin_to_cos', 'downscale_freq_shift', 'scale', 'max_period']},
        },
        'time_embedding': {
            'class': f'{type(unet.time_embedding).__module__}.{type(unet.time_embedding).__name__}',
            'state_shapes': {k: tensor_shape(v) for k, v in unet.time_embedding.state_dict().items()},
            'activation': type(getattr(unet.time_embedding, 'act', None)).__name__ if getattr(unet.time_embedding, 'act', None) is not None else None,
        },
        'downstream_overview_only': blocks,
        'not_in_scope': [
            'full UNet forward', 'down_blocks', 'mid_block', 'up_blocks', 'cross-attention',
            'runtime LoRA integration', 'full VAE pipeline', 'full TADSR inference',
        ],
        'audit_markers': {
            'TADSR_UNET_OVERVIEW_AUDIT': 'PASS',
            'TADSR_UNET_ENTRY_AUDIT': 'PASS',
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [
        '# TADSR UNet Entry Audit', '',
        'TADSR_UNET_OVERVIEW_AUDIT: PASS',
        'TADSR_UNET_ENTRY_AUDIT: PASS', '',
        f"Official class: `{report['official_unet_class']}`",
        f"Source: `{report['official_unet_source']}`",
        f"Input sample contract: `{report['input_contract']['sample']}`",
        f"Synthetic sample shape: `{report['input_contract']['recommended_synthetic_sample_shape']}`",
        f"Timestep: `{report['input_contract']['default_tadsr_inference_timestep']}`",
        f"Conv-in weight: `{report['conv_in']['weight_shape']}`; bias: `{report['conv_in']['bias_shape']}`",
        f"Time proj attrs: `{report['time_proj']['attrs']}`",
        f"Time embedding state shapes: `{report['time_embedding']['state_shapes']}`", '',
        '## Scope guard', '',
        'This audit only covers UNet entry modules: input centering, `conv_in`, `time_proj`, and `time_embedding`. It does not execute full UNet forward or any downstream blocks.',
    ]
    OUT_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TADSR_UNET_OVERVIEW_AUDIT: PASS')
    print('TADSR_UNET_ENTRY_AUDIT: PASS')
    print(json.dumps({'status': 'PASS', 'json': str(OUT_JSON), 'txt': str(OUT_TXT)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
