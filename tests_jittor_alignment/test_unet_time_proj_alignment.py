#!/usr/bin/env python3
from __future__ import annotations
from unet_entry_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_TIME_PROJ_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_time_proj_alignment', 'TADSR UNet time_proj Alignment', blocked)
        return 1
    meta = load_json(META)
    cfg = meta.get('config', {})
    tester = UNetTimestepsTester(
        num_channels=int(cfg.get('block_out_channels', [320])[0]),
        flip_sin_to_cos=bool(cfg.get('flip_sin_to_cos', True)),
        downscale_freq_shift=float(cfg.get('freq_shift', 0)),
    )
    out = tester.run_time_proj(oracle['timestep'])
    metrics = add_metrics(out, oracle['unet_time_proj_output'], tolerance=1e-5)
    result = {'status': status_from_metrics(metrics), 'target': 'UNet Timesteps positional projection', 'metrics': metrics, 'note': 'Diffusers Timesteps NumPy bridge only; no UNet forward.'}
    write_report('jittor_unet_time_proj_alignment', 'TADSR UNet time_proj Alignment', result)
    print(f"TADSR_UNET_TIME_PROJ_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
