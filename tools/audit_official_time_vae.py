#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment')


def module_rows(model):
    rows = []
    for name, module in model.named_modules():
        rows.append({
            'name': name or '<root>',
            'class': module.__class__.__name__,
            'module': f'{module.__class__.__module__}.{module.__class__.__name__}',
            'child_count': len(list(module.children())),
        })
    return rows


def state_rows(model):
    rows = []
    for key, tensor in model.state_dict().items():
        rows.append({
            'key': key,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': int(tensor.numel()),
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    modules = module_rows(model)
    states = state_rows(model)
    config = dict(model.config)

    priority = [
        {'priority': 1, 'target': 'encoder.conv_in', 'reason': 'leaf Conv2d, no timestep dependency, smallest real forward alignment target'},
        {'priority': 2, 'target': 'encoder.time_proj + encoder.time_embedding', 'reason': 'required before time-aware ResNet blocks'},
        {'priority': 3, 'target': 'encoder.down_blocks.0.resnets.0', 'reason': 'first real TimeAware ResnetBlock2D with scale-shift timestep conditioning'},
        {'priority': 4, 'target': 'encoder.down_blocks.0', 'reason': 'first DownEncoderBlock2D including two resnets and downsample'},
        {'priority': 5, 'target': 'encoder.mid_block', 'reason': 'contains attention and time-aware ResNet blocks'},
        {'priority': 6, 'target': 'quant_conv / post_quant_conv', 'reason': 'latent bottleneck conversion'},
        {'priority': 7, 'target': 'decoder blocks', 'reason': 'decoder is standard VAE decoder after latent path is aligned'},
    ]

    summary = {
        'status': 'PASS',
        'official_repo': str(OFFICIAL_REPO),
        'weights_dir': str(WEIGHTS_DIR),
        'class_name': model.__class__.__name__,
        'class_module': model.__class__.__module__,
        'encoder_class': model.encoder.__class__.__name__,
        'encoder_module': model.encoder.__class__.__module__,
        'module_count': len(modules),
        'state_key_count': len(states),
        'config': config,
        'priority_port_plan': priority,
    }

    (OUT / 'official_time_vae_state_keys.json').write_text(json.dumps({'summary': summary, 'state_keys': states}, indent=2), encoding='utf-8')
    (OUT / 'official_time_vae_module_tree.json').write_text(json.dumps({'summary': summary, 'modules': modules}, indent=2), encoding='utf-8')
    (OUT / 'official_time_vae_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    module_md = ['# Official TimeAware VAE Module Tree', '', f"Status: **{summary['status']}**", '', f"Class: `{summary['class_module']}.{summary['class_name']}`", f"Encoder: `{summary['encoder_module']}.{summary['encoder_class']}`", '', '## Config', '', '```json', json.dumps(config, indent=2), '```', '', '## Modules', '', '| Name | Class | Children |', '|---|---|---:|']
    for row in modules:
        module_md.append(f"| `{row['name']}` | `{row['class']}` | {row['child_count']} |")
    (OUT / 'official_time_vae_module_tree.md').write_text('\n'.join(module_md) + '\n', encoding='utf-8')

    key_md = ['# Official TimeAware VAE State Dict Keys', '', f"Status: **{summary['status']}**", '', '| Key | Shape | DType | Numel |', '|---|---:|---|---:|']
    for row in states:
        key_md.append(f"| `{row['key']}` | `{row['shape']}` | `{row['dtype']}` | {row['numel']} |")
    (OUT / 'official_time_vae_state_keys.md').write_text('\n'.join(key_md) + '\n', encoding='utf-8')

    plan_md = ['# TimeAware VAE Port Plan', '', 'The plan is based on the real official PyTorch `TimeAwareAutoencoderKL` module tree and converted weight shapes.', '', '| Priority | Target | Reason |', '|---:|---|---|']
    for item in priority:
        plan_md.append(f"| {item['priority']} | `{item['target']}` | {item['reason']} |")
    plan_md += ['', 'Important: full TimeAware VAE forward remains **not complete** until encoder/decoder block-level alignment is finished.']
    (OUT / 'time_vae_port_plan.md').write_text('\n'.join(plan_md) + '\n', encoding='utf-8')

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
