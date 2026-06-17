#!/usr/bin/env python3
from __future__ import annotations
from unet_entry_common import *


def main() -> int:
    meta = load_json(META)
    path_b = meta.get('path_b_vae_mode_latent_contract', {})
    result = {
        'status': path_b.get('status', 'NOT_APPLICABLE'),
        'target': 'optional VAE-mode latent -> UNet entry bridge',
        'reason': path_b.get('reason', 'Path B was not exported.'),
        'note': 'Optional check only. Full TimeAware VAE API and full TADSR inference remain intentionally unopened.',
    }
    write_report('jittor_unet_entry_vae_mode_alignment', 'TADSR UNet Entry VAE-mode Alignment', result)
    print(f"TADSR_UNET_ENTRY_VAE_MODE_ALIGNMENT: {result['status']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
