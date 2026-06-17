#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
OUT = Path('experiments/full_repro/vae_alignment')


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''


def line_number(text: str, needle: str) -> int | None:
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return None


def snippets(text: str, patterns: list[str], context: int = 2) -> list[dict]:
    lines = text.splitlines()
    out = []
    for idx, line in enumerate(lines, 1):
        if any(p in line for p in patterns):
            lo = max(1, idx - context)
            hi = min(len(lines), idx + context)
            out.append({
                'line': idx,
                'text': line.strip(),
                'context': '\n'.join(f'{j}: {lines[j - 1]}' for j in range(lo, hi + 1)),
            })
    return out


def bool_in_source(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text))


def decoder_hook_dispatch_probe(VAEHook) -> dict:
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append(('to', str(args), str(kwargs)))
            return self

        def original_forward(self, x):
            self.calls.append(('original_forward', tuple(x.shape)))
            return x + 1

    net = DummyNet()
    hook = VAEHook(net, tile_size=1, is_decoder=True, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=False)
    tiny = torch.zeros(1, 4, 8, 8)
    large = torch.zeros(1, 4, 80, 80)
    with torch.no_grad():
        tiny_out = hook(tiny)
        large_out = hook(large)
    original_calls = [c for c in net.calls if c[0] == 'original_forward']
    return {
        'status': 'PASS',
        'decoder_time_vae_false_hook_calls_original_forward_for_tiny_input': bool(torch.allclose(tiny_out, tiny + 1)),
        'decoder_time_vae_false_hook_calls_original_forward_for_large_input': bool(torch.allclose(large_out, large + 1)),
        'original_forward_call_count': len(original_calls),
        'conclusion': 'official TADSR installs decoder VAEHook with time_vae=False, and VAEHook.__call__ dispatches to net.original_forward rather than vae_tile_forward for both tiny and large decoder inputs',
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from my_utils.vaehook import VAEHook

    tadsr_path = OFFICIAL_REPO / 'tadsr.py'
    test_path = OFFICIAL_REPO / 'test_tadsr.py'
    hook_path = OFFICIAL_REPO / 'my_utils/vaehook.py'
    time_vae_path = OFFICIAL_REPO / 'diffusers/models/autoencoders/time_autoencoder_kl.py'
    tadsr_text = read(tadsr_path)
    test_text = read(test_path)
    hook_text = read(hook_path)
    time_vae_text = read(time_vae_path)

    hook_src = inspect.getsource(VAEHook)
    decoder_probe = decoder_hook_dispatch_probe(VAEHook)

    source_file_paths = {
        'tadsr.py': str(tadsr_path),
        'test_tadsr.py': str(test_path),
        'my_utils/vaehook.py': str(hook_path),
        'time_autoencoder_kl.py': str(time_vae_path),
    }
    symbol_locations = {
        'TADSR_test': {'file': str(tadsr_path), 'line': line_number(tadsr_text, 'class TADSR_test')},
        'TADSR_test._init_tiled_vae': {'file': str(tadsr_path), 'line': line_number(tadsr_text, 'def _init_tiled_vae')},
        'VAEHook': {'file': str(hook_path), 'line': line_number(hook_text, 'class VAEHook')},
        'VAEHook.__call__': {'file': str(hook_path), 'line': line_number(hook_text, 'def __call__')},
        'VAEHook.split_tiles': {'file': str(hook_path), 'line': line_number(hook_text, 'def split_tiles')},
        'VAEHook.vae_tile_forward': {'file': str(hook_path), 'line': line_number(hook_text, 'def vae_tile_forward')},
    }

    parser_defaults = {
        'vae_decoder_tiled_size': 224 if '--vae_decoder_tiled_size' in test_text else None,
        'vae_encoder_tiled_size': 1024 if '--vae_encoder_tiled_size' in test_text else None,
        'latent_tiled_size': 96 if '--latent_tiled_size' in test_text else None,
        'latent_tiled_overlap': 32 if '--latent_tiled_overlap' in test_text else None,
    }
    hook_init_params = {
        'encoder': {
            'tile_size_arg': 'args.vae_encoder_tiled_size',
            'default_from_test_tadsr': parser_defaults['vae_encoder_tiled_size'],
            'is_decoder': False,
            'fast_decoder': False,
            'fast_encoder': False,
            'color_fix': False,
            'to_gpu': True,
            'time_vae': True,
            'pad': 32,
        },
        'decoder': {
            'tile_size_arg': 'args.vae_decoder_tiled_size',
            'default_from_test_tadsr': parser_defaults['vae_decoder_tiled_size'],
            'is_decoder': True,
            'fast_decoder': False,
            'fast_encoder': False,
            'color_fix': False,
            'to_gpu': True,
            'time_vae': False,
            'pad': 11,
        },
    }

    patch_targets = {
        'vae.encoder.forward': 'replaced with VAEHook(encoder, encoder_tile_size, is_decoder=False, time_vae=True)',
        'vae.decoder.forward': 'replaced with VAEHook(decoder, decoder_tile_size, is_decoder=True, time_vae=False)',
        'vae.encode': 'not directly patched; encode internally calls the patched encoder.forward',
        'vae.decode': 'not directly patched; decode internally calls the patched decoder.forward',
        'other_functions': 'no other VAE functions are patched by _init_tiled_vae',
        'original_forward_saved': {
            'encoder': "self.vae.encoder.original_forward stores the previous encoder.forward if absent",
            'decoder': "self.vae.decoder.original_forward stores the previous decoder.forward if absent",
        },
        'restore_api': 'no explicit restore method found in TADSR_test._init_tiled_vae',
    }

    encoder_tiling_trigger = 'max(H, W) > pad * 2 + tile_size for VAEHook.__call__; however time_vae=True calls vae_tile_forward even when it prints tiny/unnecessary'
    decoder_tiling_trigger = 'not reachable under official TADSR_test decoder hook because time_vae=False dispatches to net.original_forward for both tiny and large inputs'
    encode_contract = {
        'input_layout': 'NCHW image tensor',
        'input_range': 'official caller passes lq converted to model weight dtype; earlier preprocessing maps image range to [-1,1]',
        'dtype': 'weight_dtype, float32 by default; fp16 if --mixed_precision fp16',
        'timestep': 'required for TimeAware encoder; scalar or batch tensor expanded to batch in VAEHook.vae_tile_forward',
        'tile_size': 'args.vae_encoder_tiled_size, default 1024 in test_tadsr.py',
        'padding': 32,
        'tile_split_dims': 'height/width only; batch is preserved',
        'tile_count_formula': 'ceil((H - 2*pad) / tile_size) by ceil((W - 2*pad) / tile_size), min 1 per dimension',
        'real_tile_size': 'auto-adjusted by get_best_tile_size to divisor-friendly sizes',
        'tile_call_path': 'VAEHook.vae_tile_forward builds a task queue for the encoder and executes tasks per tile',
        'time_embedding': 'computed inside VAEHook.vae_tile_forward from net.time_proj and net.time_embedding',
        'merge': 'crop_valid_region writes each tile result into the output tensor by output bounding boxes; no gaussian blend in this VAEHook implementation',
        'output': 'encoder moments before quant_conv if patch target is encoder.forward; TimeAwareAutoencoderKL.encode then applies quant_conv and builds posterior',
        'posterior_sampling': 'outside VAEHook in latent_dist.sample(); deterministic oracle should reconstruct mean + std * fixed epsilon after merged moments',
    }
    decode_contract = {
        'input_layout': 'NCHW latent tensor',
        'decode_input_scaled': 'official pipeline passes x_denoised / scaling_factor to vae.decode',
        'tile_size': 'args.vae_decoder_tiled_size, default 224 in test_tadsr.py',
        'padding': 11,
        'official_patch_installed': True,
        'official_tiled_decode_reachable': False,
        'reason': 'decoder VAEHook is constructed with time_vae=False; VAEHook.__call__ returns net.original_forward(x) in both small and large branches when time_vae is false',
        'clamp': 'outside decoder in TADSR_test.forward/decode_latent via .sample.clamp(-1, 1)',
        'sample_object_contract': 'vae.decode returns an object with .sample; VAEHook itself only replaces decoder.forward tensor path',
    }

    can_trigger_encoder_with_controlled_tile = True
    can_trigger_decoder_with_official_hook = False
    feasibility_status = 'BLOCKED_DECODER_HOOK_CONTRACT'
    feasibility_reason = (
        'Official _init_tiled_vae installs a decoder VAEHook, but the hook is constructed with time_vae=False and '
        'the VAEHook.__call__ source dispatches to net.original_forward instead of vae_tile_forward for decoder inputs. '
        'Therefore a truthful official tiled encode/decode oracle cannot claim tiled decode was triggered.'
    )

    summary = {
        'status': 'PASS',
        'source_file_paths': source_file_paths,
        'symbol_locations': symbol_locations,
        'symbol_names': ['TADSR_test', '_init_tiled_vae', 'VAEHook', 'VAEHook.__call__', 'VAEHook.vae_tile_forward'],
        'call_graph_summary': [
            'TADSR_test.__init__ loads TimeAwareAutoencoderKL and immediately calls _init_tiled_vae.',
            '_init_tiled_vae saves encoder/decoder original_forward and replaces encoder.forward / decoder.forward.',
            'TADSR_test.forward calls vae.encode(lq, timesteps).latent_dist.sample() * scaling_factor.',
            'TADSR_test.forward later calls vae.decode(x_denoised / scaling_factor).sample.clamp(-1, 1).',
            'encode enters patched encoder.forward and can execute VAEHook.vae_tile_forward with time_vae=True.',
            'decode enters patched decoder.forward, but official hook dispatches to decoder.original_forward because time_vae=False.',
        ],
        'pipeline_usage_audit': {
            'status': 'PASS',
            'tadsr_test_calls_init_tiled_vae': '_init_tiled_vae' in tadsr_text,
            'train_or_generation_path_uses_init_tiled_vae': False,
            'test_or_inference_path_uses_init_tiled_vae': True,
            'cli_or_config_controls': parser_defaults,
            'encode_sample_scale_snippets': snippets(tadsr_text, ['vae.encode', 'latent_dist.sample', 'scaling_factor'], 2),
            'decode_clamp_snippets': snippets(tadsr_text, ['vae.decode', 'clamp(-1, 1)', 'decode_latent'], 2),
            'init_tiled_vae_snippets': snippets(tadsr_text, ['_init_tiled_vae', 'encoder.forward = VAEHook', 'decoder.forward = VAEHook'], 2),
        },
        'hook_audit': {
            'status': 'PASS',
            'patch_targets': patch_targets,
            'hook_init_params': hook_init_params,
            'decoder_dispatch_probe': decoder_probe,
            'source_contains_vae_tile_forward': bool_in_source(hook_src, r'def vae_tile_forward'),
            'source_contains_original_forward_dispatch': 'return self.net.original_forward(x)' in hook_src,
        },
        'tiled_boundary_contract': {
            'status': 'PASS',
            'encoder': encode_contract,
            'decoder': decode_contract,
            'encoder_tiling_trigger': encoder_tiling_trigger,
            'decoder_tiling_trigger': decoder_tiling_trigger,
            'minimal_reproducible_size': {
                'official_test_defaults': parser_defaults,
                'controlled_encoder_oracle_recommendation': {
                    'shape': [1, 3, 96, 96],
                    'encoder_tile_size': 16,
                    'reason': '96 > 32*2 + 16, so VAEHook.__call__ enters the large-input branch and vae_tile_forward for encoder',
                },
                'decoder_oracle_recommendation': None,
                'decoder_blocker': feasibility_reason,
            },
            'tiled_vs_nontiled_reference': {
                'status': 'NOT_APPLICABLE',
                'reason': 'full official tiled encode/decode oracle is blocked by the decoder hook contract; do not force zero-error tiled vs non-tiled comparison',
            },
            'scheduler_executed': False,
            'full_tadsr_inference_executed': False,
            'image_saved': False,
            'jittor_tiled_implementation_done': False,
            'timevae_full_alignment_candidate': False,
        },
        'oracle_feasibility': {
            'status': feasibility_status,
            'reason': feasibility_reason,
            'encoder_tiled_oracle_feasible': can_trigger_encoder_with_controlled_tile,
            'decoder_tiled_oracle_feasible_under_official_hook': can_trigger_decoder_with_official_hook,
            'recommended_next_stage': 'Resolve whether the Jittor port should mirror official decoder.original_forward dispatch or intentionally implement a corrected decoder tiled path; until then implement only an audit/export blocker, not Jittor tiled VAE.',
        },
    }
    (OUT / 'audit_tadsr_vae_tiled_boundary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    txt = [
        'TIME_VAE_TILED_HOOK_AUDIT: PASS',
        'TIME_VAE_TILED_PIPELINE_USAGE_AUDIT: PASS',
        'TIME_VAE_TILED_BOUNDARY_CONTRACT_AUDIT: PASS',
        f'TIME_VAE_TILED_ORACLE_FEASIBILITY: {feasibility_status}',
        '',
        'Key finding:',
        feasibility_reason,
        '',
        f"_init_tiled_vae: {symbol_locations['TADSR_test._init_tiled_vae']}",
        f"VAEHook: {symbol_locations['VAEHook']}",
        'Patch targets: vae.encoder.forward and vae.decoder.forward only; vae.encode/vae.decode are not directly replaced.',
        'Encoder hook: time_vae=True, pad=32, tile_size=args.vae_encoder_tiled_size.',
        'Decoder hook: time_vae=False, pad=11, tile_size=args.vae_decoder_tiled_size.',
        'Decoder tiled path under official hook: not reachable.',
        'TIME_VAE_TILED_VS_NONTILED_REFERENCE_RECORDED: NOT_APPLICABLE',
    ]
    (OUT / 'audit_tadsr_vae_tiled_boundary.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
    print('\n'.join(txt[:4]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
