# Official runtime dependency overlay

`TADSR_OFFICIAL_DIFFUSERS_OVERLAY_READY: PASS`
`TADSR_OFFICIAL_RUNTIME_DEPENDENCY_REPAIR_READINESS: PASS`

- `official_python`: `/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python`
- `overlay_dir`: `/mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers`
- `strict_env_modified`: `False`
- `execute`: `True`
- `package`: `diffusers`
- `version`: `0.30.3`
- `version_source`: `cli`
- `overlay_required`: `True`
- `official_import_diffusers_before`: `MISSING`
- `official_import_diffusers_after_with_overlay`: `PASS`
- `safe_to_use_overlay_for_timevae_metadata`: `True`

## Planned / executed command

```text
/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python -m pip install --target /mnt/data/sj/tmp/tadsr_official_dependency_overlay_diffusers --no-deps diffusers==0.30.3
```

## pip output tail

```text
Collecting diffusers==0.30.3
  Using cached diffusers-0.30.3-py3-none-any.whl.metadata (18 kB)
Using cached diffusers-0.30.3-py3-none-any.whl (2.7 MB)
Installing collected packages: diffusers
Successfully installed diffusers-0.30.3
```
