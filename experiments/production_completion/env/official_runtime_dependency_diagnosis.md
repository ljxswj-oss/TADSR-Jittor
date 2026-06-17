# Official runtime dependency diagnosis

`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_DIAGNOSIS: PASS`

- `official_repo`: `/mnt/data/sj/projects/TADSR_official_pytorch`
- `official_weights`: `/mnt/data/sj/checkpoints/TADSR/preset/weights`
- `official_python`: `/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python`
- `pythonpath_overlay`: `/mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers`
- `strict_env_modified`: `False`
`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_OVERLAY_ACTIVE: PASS`

## Python imports

| Module | Status | Version / Error |
|---|---|---|
| `torch` | `PASS` | 2.0.1+cu118 |
| `diffusers` | `PASS` | 0.30.3 |
| `peft` | `PASS` | 0.9.0 |
| `transformers` | `PASS` | 4.28.1 |
| `accelerate` | `PASS` | 1.13.0 |
| `safetensors` | `PASS` | 0.7.0 |
| `PIL` | `PASS` | 9.5.0 |
| `numpy` | `PASS` | 1.24.3 |
| `cv2` | `PASS` | 4.11.0 |

## Official repo evidence

- `tadsr.py`
- `test_tadsr.py`
- `train_tadsr.py`
- `__pycache__/tadsr.cpython-310.pyc`
- `__pycache__/vae_my.cpython-310.pyc`
- `__pycache__/test_tadsr.cpython-310.pyc`
- `__pycache__/train_tadsr.cpython-310.pyc`
- `diffusers/pipelines`
- `diffusers/models/autoencoders`
- `diffusers/models/vae_flax.py`
- `diffusers/models/__pycache__/vae_flax.cpython-310.pyc`
- `diffusers/models/autoencoders/__init__.py`
- `diffusers/models/autoencoders/__pycache__`
- `diffusers/models/autoencoders/autoencoder_asym_kl.py`
- `diffusers/models/autoencoders/autoencoder_kl.py`
- `diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py`
- `diffusers/models/autoencoders/autoencoder_tiny.py`
- `diffusers/models/autoencoders/consistency_decoder_vae.py`
- `diffusers/models/autoencoders/time_autoencoder_kl.py`
- `diffusers/models/autoencoders/vae.py`
- `diffusers/models/autoencoders/__pycache__/__init__.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/autoencoder_asym_kl.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/autoencoder_kl.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/autoencoder_kl_temporal_decoder.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/autoencoder_tiny.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/consistency_decoder_vae.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/time_autoencoder_kl.cpython-310.pyc`
- `diffusers/models/autoencoders/__pycache__/vae.cpython-310.pyc`
- `diffusers/pipelines/__init__.py`
- `diffusers/pipelines/__pycache__`

## Weight candidates

- `scheduler`: 1 candidate(s)
  - `scheduler`
- `vae`: 2 candidate(s)
  - `time_vae`
  - `vae`
- `unet`: 1 candidate(s)
  - `unet`
- `lora`: 0 candidate(s)
