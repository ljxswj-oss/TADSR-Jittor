# Official production environment resolution

`TADSR_PRODUCTION_OFFICIAL_ENV_RESOLUTION: PASS`

| Item | Status | Detail |
|---|---|---|
| official_repo_exists | PASS | /mnt/data/sj/projects/TADSR_official_pytorch (cli) |
| official_weights_exists | PASS | /mnt/data/sj/checkpoints/TADSR/preset/weights (cli) |
| official_python_exists | PASS | /mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python (cli) |
| official_repo_has_expected_files | PASS | all expected files present |
| official_weights_has_expected_files | PASS | all expected weights present |
| official_python_version | PASS | 3.10.12 |
| official_python_can_import_torch | PASS | 2.0.1+cu118 |
| official_python_can_import_diffusers | PARTIAL | Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'diffusers' |
| scheduler_config_exists | PASS | scheduler/scheduler_config.json |
| model_index_or_config_candidates | PASS | scheduler/scheduler_config.json, time_vae/config.json, unet/config.json |
| timevae_weight_candidates | PASS | time_vae/diffusion_pytorch_model.bin |
| lora_weight_candidates | PARTIAL | no LoRA weight candidates found |

## Next action

Proceed with live metadata-only TimeVAE oracle export.
