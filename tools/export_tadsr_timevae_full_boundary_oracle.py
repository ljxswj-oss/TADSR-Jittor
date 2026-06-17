#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

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
WEIGHTS_DIR = Path('/mnt/data/sj/checkpoints/TADSR/preset/weights/time_vae')
BASE_OUT = Path('experiments/full_repro/vae_alignment')
ORACLE_OUT = BASE_OUT / 'oracle_tensors_timevae_full_boundary'
AUDIT_PATH = BASE_OUT / 'audit_tadsr_timevae_full_boundary.json'


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


def save(name: str, value, source: str, arrays: dict[str, np.ndarray], stats: list[dict]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arrays[name] = arr
    np.save(ORACLE_OUT / f'{name}.npy', arr)
    stats.append(stat(name, arr, source))
    return arr


def main() -> int:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    ORACLE_OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(OFFICIAL_REPO))
    from diffusers.models.autoencoders.time_autoencoder_kl import TimeAwareAutoencoderKL

    if not AUDIT_PATH.exists():
        raise SystemExit('Missing audit metadata; run tools/audit_official_tadsr_timevae_full_boundary.py first')
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    if audit.get('full_boundary_contract', {}).get('status') != 'PASS':
        raise SystemExit('Boundary contract audit is not PASS')

    torch.manual_seed(42)
    np.random.seed(42)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    model = TimeAwareAutoencoderKL.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    model.eval()
    scaling_factor = float(getattr(model.config, 'scaling_factor', 0.18215))
    arrays: dict[str, np.ndarray] = {}
    stats: list[dict] = []

    # The small 32x32 input avoids official VAEHook tiled paths and full inference.
    sample_np = np.linspace(-1.0, 1.0, num=1 * 3 * 32 * 32, dtype=np.float32).reshape(1, 3, 32, 32)
    timestep_np = np.array([50], dtype=np.int64)
    epsilon_np = np.linspace(-0.75, 0.75, num=1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)

    with torch.no_grad():
        sample = torch.from_numpy(sample_np)
        timesteps = torch.from_numpy(timestep_np).long()
        epsilon = torch.from_numpy(epsilon_np)
        time_proj = model.encoder.time_proj(timesteps)
        encoder_temb = model.encoder.time_embedding(time_proj.to(dtype=list(model.encoder.time_embedding.parameters())[0].dtype))
        encoded = model.encode(sample, timesteps)
        posterior = encoded.latent_dist
        mean = posterior.mean
        logvar = posterior.logvar
        std = posterior.std
        posterior_sample = mean + std * epsilon
        scaled_latent = posterior_sample * scaling_factor
        decode_input = scaled_latent / scaling_factor
        decoded = model.decode(decode_input).sample
        final = decoded.clamp(-1, 1)

    np.save(ORACLE_OUT / 'timevae_full_boundary_timestep.npy', timestep_np)
    stats.append(stat('timevae_full_boundary_timestep', timestep_np, 'deterministic timestep passed to encode'))
    save('timevae_full_boundary_input', sample_np, 'deterministic image tensor in [-1,1]', arrays, stats)
    save('timevae_full_boundary_encoder_temb', to_np(encoder_temb), 'encoder.time_proj + encoder.time_embedding(timestep)', arrays, stats)
    save('timevae_full_boundary_moments', to_np(posterior.parameters), 'TimeAwareAutoencoderKL.encode(...).latent_dist.parameters', arrays, stats)
    save('timevae_full_boundary_posterior_mean', to_np(mean), 'latent_dist.mean', arrays, stats)
    save('timevae_full_boundary_posterior_logvar', to_np(logvar), 'latent_dist.logvar after official clamp', arrays, stats)
    save('timevae_full_boundary_posterior_std', to_np(std), 'latent_dist.std', arrays, stats)
    save('timevae_full_boundary_sample_epsilon', epsilon_np, 'exported deterministic epsilon for sample reproduction', arrays, stats)
    save('timevae_full_boundary_posterior_sample', to_np(posterior_sample), 'mean + std * exported epsilon', arrays, stats)
    save('timevae_full_boundary_scaled_latent', to_np(scaled_latent), 'posterior_sample * scaling_factor', arrays, stats)
    save('timevae_full_boundary_decode_input', to_np(decode_input), 'scaled_latent / scaling_factor, passed to decode', arrays, stats)
    save('timevae_full_boundary_decoded_output', to_np(decoded), 'TimeAwareAutoencoderKL.decode(...).sample', arrays, stats)
    save('timevae_full_boundary_final_clamped_output', to_np(final), 'decoded sample clamped to [-1,1]', arrays, stats)

    metadata = {
        'status': 'PASS',
        'boundary_type': audit['full_boundary_contract']['boundary_type'],
        'official_vae_class': audit['official_vae_class'],
        'weights_dir': str(WEIGHTS_DIR),
        'input_shape': list(sample_np.shape),
        'input_dtype': 'float32',
        'input_range': [-1.0, 1.0],
        'timestep': int(timestep_np[0]),
        'latent_shape': list(arrays['timevae_full_boundary_posterior_sample'].shape),
        'latent_dtype': 'float32',
        'scaling_factor': scaling_factor,
        'output_shape': list(arrays['timevae_full_boundary_final_clamped_output'].shape),
        'output_range': [-1.0, 1.0],
        'layout': 'NCHW',
        'deterministic_policy': audit['full_boundary_contract']['deterministic_policy'],
        'stochastic_sampling_avoided_or_fixed': 'fixed exported epsilon tensor',
        'existing_full_decoder_oracle_reused': False,
        'this_boundary_is_sufficient_for_non_tiled_api': True,
        'this_boundary_is_sufficient_for_full_tadsr_pipeline': False,
        'scheduler_executed': False,
        'full_tadsr_inference_executed': False,
        'image_saved': False,
        'timevae_full_alignment_candidate': False,
        'remaining_timevae_gaps': audit['full_boundary_contract']['remaining_timevae_gaps'],
        'recommended_next_stage': audit['full_boundary_contract']['recommended_next_stage'],
        'tensor_stats': stats,
    }
    (ORACLE_OUT / 'timevae_full_boundary_oracle_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary = [
        'TIME_VAE_FULL_BOUNDARY_ORACLE_TENSORS: PASS',
        '',
        f"Boundary type: {metadata['boundary_type']}",
        f"Input shape: {metadata['input_shape']}",
        f"Latent shape: {metadata['latent_shape']}",
        f"Output shape: {metadata['output_shape']}",
        f"Scaling factor: {metadata['scaling_factor']}",
        f"Full TADSR inference executed: {metadata['full_tadsr_inference_executed']}",
        f"TimeVAE full alignment candidate: {metadata['timevae_full_alignment_candidate']}",
    ]
    (ORACLE_OUT / 'oracle_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(summary[0])
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
