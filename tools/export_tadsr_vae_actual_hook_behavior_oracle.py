#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

OFFICIAL_VENV_PYTHON = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')
OFFICIAL_VENV_ROOT = OFFICIAL_VENV_PYTHON.parents[1]
if (
    OFFICIAL_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != OFFICIAL_VENV_ROOT.resolve()
    and os.environ.get('TADSR_OFFICIAL_VENV_REEXEC') != '1'
):
    env = os.environ.copy()
    env['TADSR_OFFICIAL_VENV_REEXEC'] = '1'
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.execve(str(OFFICIAL_VENV_PYTHON), [str(OFFICIAL_VENV_PYTHON), *sys.argv], env)

import numpy as np
import torch

OFFICIAL_REPO = Path('/mnt/data/sj/projects/TADSR_official_pytorch')
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
BASE_OUT = Path('experiments/full_repro/vae_alignment')
AUDIT_PATH = BASE_OUT / 'audit_tadsr_vae_actual_hook_behavior.json'
ORACLE_OUT = BASE_OUT / 'oracle_tensors_vae_actual_hook_behavior'


def to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def stat(name: str, arr: np.ndarray, source: str) -> dict:
    a = np.asarray(arr)
    return {
        'name': name,
        'shape': list(a.shape),
        'dtype': str(a.dtype),
        'min': float(a.min()) if a.size else None,
        'max': float(a.max()) if a.size else None,
        'mean': float(a.mean()) if a.size else None,
        'std': float(a.std()) if a.size else None,
        'sha256': sha256_array(a),
        'source': source,
        'nbytes': int(a.nbytes),
    }


def save(name: str, value, source: str, stats: list[dict]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    np.save(ORACLE_OUT / f'{name}.npy', arr)
    stats.append(stat(name, arr, source))
    return arr


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    diff = aa - bb
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return {
        'shape_match': list(np.asarray(a).shape) == list(np.asarray(b).shape),
        'max_abs_error': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'mean_abs_error': float(np.mean(np.abs(diff))) if diff.size else 0.0,
        'cosine_similarity': float(np.dot(aa, bb) / denom) if denom else 1.0,
    }


def main() -> int:
    if not AUDIT_PATH.exists():
        raise SystemExit(f'Missing actual hook behavior audit metadata: {AUDIT_PATH}')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    if audit.get('status') != 'PASS':
        raise SystemExit(f"Actual hook behavior audit is not PASS: {audit.get('status')}")

    ORACLE_OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL
    from my_utils.vaehook import VAEHook, build_task_queue
    import my_utils.devices as devices

    torch.manual_seed(123)
    np.random.seed(123)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # The Jittor migration checks run the alignment bridge on CPU/NumPy. The
    # official tiled encoder accumulates many Conv/GroupNorm/Attention steps,
    # and CUDA vs CPU arithmetic drifts enough to obscure implementation bugs.
    # Export this oracle on the official CPU path so the tensor contract is
    # deterministic and directly comparable to the Jittor-side checker.
    device = torch.device('cpu')
    devices.device = device
    devices.get_optimal_device = lambda: device
    devices.torch_gc = lambda: None
    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    model = model.to(device)

    if not hasattr(model.encoder, 'original_forward'):
        model.encoder.original_forward = model.encoder.forward
    if not hasattr(model.decoder, 'original_forward'):
        model.decoder.original_forward = model.decoder.forward

    raw_decoder_forward = model.decoder.original_forward
    decoder_counter = {'original_forward_calls': 0, 'vae_tile_forward_calls': 0}

    def counted_decoder_original_forward(*args, **kwargs):
        decoder_counter['original_forward_calls'] += 1
        return raw_decoder_forward(*args, **kwargs)

    model.decoder.original_forward = counted_decoder_original_forward

    encoder_counter = {'vae_tile_forward_calls': 0}
    encoder_hook = VAEHook(model.encoder, 16, is_decoder=False, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=(device.type == 'cuda'), time_vae=True)
    encoder_tile_forward = encoder_hook.vae_tile_forward

    def counted_encoder_tile_forward(*args, **kwargs):
        encoder_counter['vae_tile_forward_calls'] += 1
        return encoder_tile_forward(*args, **kwargs)

    encoder_hook.vae_tile_forward = counted_encoder_tile_forward
    model.encoder.forward = encoder_hook

    decoder_hook = VAEHook(model.decoder, 16, is_decoder=True, fast_decoder=False, fast_encoder=False, color_fix=False, to_gpu=(device.type == 'cuda'), time_vae=False)
    decoder_tile_forward = decoder_hook.vae_tile_forward

    def counted_decoder_tile_forward(*args, **kwargs):
        decoder_counter['vae_tile_forward_calls'] += 1
        return decoder_tile_forward(*args, **kwargs)

    decoder_hook.vae_tile_forward = counted_decoder_tile_forward
    model.decoder.forward = decoder_hook

    stats: list[dict] = []
    scaling_factor = float(getattr(model.config, 'scaling_factor', 0.18215))
    sample = torch.linspace(-1.0, 1.0, 1 * 3 * 96 * 96, dtype=torch.float32, device=device).reshape(1, 3, 96, 96)
    timestep = torch.tensor([50], dtype=torch.long, device=device)
    tile_input_bboxes, tile_output_bboxes = encoder_hook.split_tiles(sample.shape[2], sample.shape[3])
    tile_records = []
    for tile_index, (input_bbox, output_bbox) in enumerate(zip(tile_input_bboxes, tile_output_bboxes)):
        padded_bbox = [v // 8 for v in input_bbox]
        crop_margin = [output_bbox[i] - padded_bbox[i] for i in range(4)]
        tile_records.append({
            'tile_index': tile_index,
            'input_bbox': input_bbox,
            'output_bbox': output_bbox,
            'padded_output_bbox': padded_bbox,
            'crop_margin': crop_margin,
            'write_bbox': output_bbox,
        })
        tile = sample[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]]
        save(f'actual_encoder_input_tile_{tile_index}', to_np(tile), f'official encoder VAEHook input tile {tile_index}', stats)

    task_queue = build_task_queue(model.encoder, is_decoder=False, time_vae=True)
    task_queue_summary = []
    for index, task in enumerate(task_queue):
        payload = task[1] if len(task) > 1 else None
        task_queue_summary.append({
            'index': index,
            'task': task[0],
            'payload_class': payload.__class__.__name__ if payload is not None else None,
        })

    with torch.no_grad():
        encoded = model.encode(sample, timestep)
        actual_encoder_raw = encoder_tile_forward(sample, timestep)
        actual_encoder_moments_from_raw = model.quant_conv(actual_encoder_raw)
        mean = encoded.latent_dist.mean
        logvar = encoded.latent_dist.logvar
        std = encoded.latent_dist.std
        epsilon = torch.linspace(-0.5, 0.5, mean.numel(), dtype=mean.dtype, device=device).reshape_as(mean)
        posterior_sample = mean + std * epsilon
        scaled_latent = posterior_sample * scaling_factor
        decode_input = scaled_latent / scaling_factor
        decoded = model.decode(decode_input).sample
        final = decoded.clamp(-1, 1)

        model.decoder.forward = raw_decoder_forward
        model.decoder.original_forward = raw_decoder_forward
        nontiled_decoded = model.decode(decode_input).sample
        nontiled_final = nontiled_decoded.clamp(-1, 1)

    np.save(ORACLE_OUT / 'actual_hook_timestep.npy', np.array([50], dtype=np.int64))
    stats.append(stat('actual_hook_timestep', np.array([50], dtype=np.int64), 'deterministic timestep'))
    save('actual_hook_input', to_np(sample), 'deterministic input image tensor in [-1,1]', stats)
    raw_np = save('actual_encoder_tiled_raw_output', to_np(actual_encoder_raw), 'official actual encoder VAEHook tiled raw output before quant_conv', stats)
    recomputed_moments_np = save('actual_encoder_tiled_recomputed_moments', to_np(actual_encoder_moments_from_raw), 'quant_conv(actual_encoder_tiled_raw_output)', stats)
    save('actual_encoder_moments', to_np(encoded.latent_dist.parameters), 'official actual encoder hook path: VAEHook(time_vae=True) + quant_conv posterior parameters', stats)
    save('actual_posterior_mean', to_np(mean), 'latent_dist.mean after official actual encoder hook path', stats)
    save('actual_posterior_logvar', to_np(logvar), 'latent_dist.logvar after official actual encoder hook path', stats)
    save('actual_posterior_std', to_np(std), 'latent_dist.std after official actual encoder hook path', stats)
    save('actual_sample_epsilon', to_np(epsilon), 'fixed deterministic epsilon for posterior sample reproduction', stats)
    save('actual_posterior_sample', to_np(posterior_sample), 'mean + std * fixed epsilon', stats)
    save('actual_scaled_latent', to_np(scaled_latent), 'posterior_sample * scaling_factor', stats)
    save('actual_decode_input', to_np(decode_input), 'scaled_latent / scaling_factor', stats)
    decoded_np = save('actual_decoded_output', to_np(decoded), 'official actual decoder hook path: hook installed but original_forward dispatch', stats)
    final_np = save('actual_final_clamped_output', to_np(final), 'actual decoded output clamped to [-1,1]', stats)
    nontiled_decoded_np = save('nontiled_reference_decoded_output', to_np(nontiled_decoded), 'decoder reference with hook removed', stats)
    nontiled_final_np = save('nontiled_reference_final_clamped_output', to_np(nontiled_final), 'decoder reference final output with hook removed', stats)

    enc_status = 'PASS' if encoder_counter['vae_tile_forward_calls'] > 0 else 'NOT_TRIGGERED'
    dec_original_status = 'PASS' if decoder_counter['original_forward_calls'] > 0 and decoder_counter['vae_tile_forward_calls'] == 0 else 'FAIL'
    cmp_decoded = metrics(decoded_np, nontiled_decoded_np)
    cmp_final = metrics(final_np, nontiled_final_np)
    cmp_raw_quant = metrics(recomputed_moments_np, to_np(encoded.latent_dist.parameters))
    status = 'PASS' if dec_original_status == 'PASS' else 'FAIL'

    metadata = {
        'status': status,
        'oracle_type': 'official_actual_vaehook_behavior',
        'not_ideal_tiled_decode': True,
        'source_audit_file': str(AUDIT_PATH),
        'policy_decision': 'mirror_official_actual_behavior',
        'input_shape': list(sample.shape),
        'input_dtype': 'float32',
        'oracle_device': str(device),
        'oracle_device_policy': 'official_cpu_deterministic_export_for_jittor_numpy_alignment',
        'input_range': [-1.0, 1.0],
        'layout': 'NCHW',
        'timestep_shape': [1],
        'timestep_dtype': 'int64',
        'timestep_value': 50,
        'latent_shape': list(mean.shape),
        'output_shape': list(final.shape),
        'scaling_factor': scaling_factor,
        'deterministic_fixed_epsilon_policy': 'posterior_sample = mean + std * exported epsilon',
        'encoder_hook_installed': True,
        'encoder_tiled_path_triggered': encoder_counter['vae_tile_forward_calls'] > 0,
        'encoder_tile_count': audit.get('encoder_hook', {}).get('total_tile_count', 4),
        'encoder_tile_size': 16,
        'encoder_pad': 32,
        'encoder_tile_input_bboxes': tile_input_bboxes,
        'encoder_tile_output_bboxes': tile_output_bboxes,
        'encoder_tile_records': tile_records,
        'encoder_write_back_rule': 'assignment_no_blending',
        'encoder_gaussian_blend_used': False,
        'encoder_task_queue_summary': task_queue_summary,
        'encoder_task_queue_counts': {name: sum(1 for row in task_queue_summary if row['task'] == name) for name in sorted({row['task'] for row in task_queue_summary})},
        'encoder_groupnorm_contract': {
            'class': 'GroupNormParam',
            'num_groups': 32,
            'eps': 1e-6,
            'var_unbiased': False,
            'aggregation': 'per-tile var/mean weighted by tile pixel count normalized by max(pixel_list), then normalized by sum',
            'pause_task': 'pre_norm',
            'resume_task': 'apply_norm',
        },
        'actual_encoder_tiled_raw_output_exported': True,
        'actual_encoder_tiled_recomputed_moments_vs_encoded_parameters': cmp_raw_quant,
        'decoder_hook_installed': True,
        'decoder_original_forward_used': decoder_counter['original_forward_calls'] > 0,
        'decoder_original_forward_call_count': decoder_counter['original_forward_calls'],
        'decoder_tiled_path_triggered': False,
        'decoder_tiled_path_status': 'BLOCKED_DECODER_HOOK_CONTRACT',
        'actual_hook_vs_nontiled_decode': cmp_decoded,
        'actual_hook_vs_nontiled_final_output': cmp_final,
        'scheduler_executed': False,
        'full_tadsr_inference_executed': False,
        'image_saved': False,
        'jittor_tiled_implementation_done': False,
        'timevae_full_alignment_candidate': False,
        'recommended_next_stage': 'implement Jittor official-actual VAEHook behavior wrapper/tester, not corrected tiled decoder',
        'tensor_stats': stats,
    }
    (ORACLE_OUT / 'vae_actual_hook_behavior_oracle_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = [
        'TIME_VAE_ACTUAL_VAEHOOK_ORACLE_TENSORS: PASS',
        f'TIME_VAE_ACTUAL_ENCODER_TILED_ORACLE_TENSORS: {enc_status}',
        f'TIME_VAE_ACTUAL_DECODER_HOOK_ORIGINAL_FORWARD_ORACLE: {dec_original_status}',
        'TIME_VAE_TILED_ORACLE_TENSORS: BLOCKED_DECODER_HOOK_CONTRACT',
        '',
        f"encoder_tiled_path_triggered: {metadata['encoder_tiled_path_triggered']}",
        f"decoder_original_forward_used: {metadata['decoder_original_forward_used']}",
        f"decoder_tiled_path_triggered: {metadata['decoder_tiled_path_triggered']}",
        f"actual_encoder_raw_quant_vs_encoded_max_abs: {cmp_raw_quant['max_abs_error']}",
        f"actual_hook_vs_nontiled_decode_max_abs: {cmp_decoded['max_abs_error']}",
        f"actual_hook_vs_nontiled_final_max_abs: {cmp_final['max_abs_error']}",
    ]
    (ORACLE_OUT / 'oracle_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:4]))
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
