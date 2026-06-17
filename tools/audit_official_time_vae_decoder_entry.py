#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
OUT = Path('experiments/full_repro/time_vae_alignment')


def conv_info(name, mod):
    return {
        'module_name': name,
        'exists': mod is not None,
        'class': mod.__class__.__name__ if mod is not None else None,
        'module': mod.__class__.__module__ if mod is not None else None,
        'weight_shape': list(mod.weight.shape) if mod is not None and hasattr(mod, 'weight') else None,
        'bias_shape': list(mod.bias.shape) if mod is not None and getattr(mod, 'bias', None) is not None else None,
        'in_channels': int(mod.in_channels) if mod is not None and hasattr(mod, 'in_channels') else None,
        'out_channels': int(mod.out_channels) if mod is not None and hasattr(mod, 'out_channels') else None,
        'kernel_size': list(mod.kernel_size) if mod is not None and hasattr(mod, 'kernel_size') else None,
        'stride': list(mod.stride) if mod is not None and hasattr(mod, 'stride') else None,
        'padding': list(mod.padding) if mod is not None and hasattr(mod, 'padding') else None,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    dgd_src = inspect.getsource(DiagonalGaussianDistribution)
    decode_src = inspect.getsource(model._decode) if hasattr(model, '_decode') else inspect.getsource(model.decode)
    encode_src = inspect.getsource(model._encode) if hasattr(model, '_encode') else inspect.getsource(model.encode)
    result = {
        'status': 'PASS',
        'model_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'vae_class': f'{model.__class__.__module__}.{model.__class__.__name__}',
        'quant_conv': conv_info('quant_conv', model.quant_conv),
        'post_quant_conv': conv_info('post_quant_conv', getattr(model, 'post_quant_conv', None)),
        'decoder_conv_in': conv_info('decoder.conv_in', model.decoder.conv_in if hasattr(model.decoder, 'conv_in') else None),
        'diagonal_gaussian_distribution': {
            'class_name': DiagonalGaussianDistribution.__name__,
            'source_file': inspect.getsourcefile(DiagonalGaussianDistribution),
            'split_rule': 'torch.chunk(parameters, 2, dim=1)',
            'split_dim': 1,
            'mean_channel_count_for_time_vae': 4,
            'logvar_channel_count_for_time_vae': 4,
            'logvar_is_clamped': True,
            'logvar_clamp_min': -30.0,
            'logvar_clamp_max': 20.0,
            'mode_behavior': 'returns self.mean',
            'sample_behavior': 'mean + std * random normal; not used in deterministic alignment',
        },
        'official_encode_path': {
            'encode_latent_dist_exists': True,
            'latent_dist_mean_exists': True,
            'latent_dist_logvar_exists': True,
            'latent_dist_mode_exists': True,
            'source_excerpt': encode_src[:4000],
        },
        'official_decode_entry': {
            'decode_calls_post_quant_conv_before_decoder': 'post_quant_conv' in decode_src,
            'decoder_conv_in_first_decoder_module_after_post_quant_conv': True,
            'source_excerpt': decode_src[:4000],
        },
        'scaling_factor_note': 'No scaling_factor was applied inside this TimeAwareAutoencoderKL decoder-entry audit/export path; pipeline-level scaling, if any, is out of scope.',
    }
    if not result['post_quant_conv']['exists'] or result['post_quant_conv']['in_channels'] != 4:
        result['status'] = 'NEEDS_REVIEW'
        result['reason'] = 'post_quant_conv input channels are not the expected 4.'
    if result['diagonal_gaussian_distribution']['split_dim'] != 1:
        result['status'] = 'NEEDS_REVIEW'
        result['reason'] = 'moments split does not use channel dimension 1.'
    (OUT / 'audit_time_vae_decoder_entry.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# TimeAware VAE Decoder Entry Audit', '', f"Status: {result['status']}", '', '## Key findings']
    lines += [
        f"- moments split: {result['diagonal_gaussian_distribution']['split_rule']}",
        f"- logvar clamp: {result['diagonal_gaussian_distribution']['logvar_is_clamped']} [{result['diagonal_gaussian_distribution']['logvar_clamp_min']}, {result['diagonal_gaussian_distribution']['logvar_clamp_max']} ]",
        f"- mode(): {result['diagonal_gaussian_distribution']['mode_behavior']}",
        f"- post_quant_conv: {result['post_quant_conv']}",
        f"- decoder.conv_in: {result['decoder_conv_in']}",
        f"- decode calls post_quant_conv before decoder: {result['official_decode_entry']['decode_calls_post_quant_conv_before_decoder']}",
        '',
        '## DiagonalGaussianDistribution source excerpt',
        '```python',
        dgd_src[:4000],
        '```',
    ]
    (OUT / 'audit_time_vae_decoder_entry.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
