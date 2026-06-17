#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_resnet1_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_RESNET1_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_resnet1_alignment', 'TADSR UNet down_blocks.0.resnets.1 Alignment', blocked)
        return 1
    t = resnet1_tester()
    out = t.run(oracle['entry_downblock0_resnet1_input'], oracle['entry_downblock0_resnet1_temb'], return_intermediates=True)
    mapping = [
        ('norm1', 'entry_downblock0_resnet1_norm1_output', 'TADSR_UNET_DOWNBLOCK0_RESNET1_NORM1_ALIGNMENT'),
        ('conv1', 'entry_downblock0_resnet1_conv1_output', 'TADSR_UNET_DOWNBLOCK0_RESNET1_CONV1_ALIGNMENT'),
        ('time_emb_proj', 'entry_downblock0_resnet1_time_emb_proj_output', 'TADSR_UNET_DOWNBLOCK0_RESNET1_TEMB_PROJ_ALIGNMENT'),
        ('conv2', 'entry_downblock0_resnet1_conv2_output', 'TADSR_UNET_DOWNBLOCK0_RESNET1_CONV2_ALIGNMENT'),
        ('output', 'entry_downblock0_resnet1_output', 'TADSR_UNET_DOWNBLOCK0_RESNET1_ALIGNMENT'),
    ]
    diagnostics = {}
    for stage, key, _ in mapping:
        expected = squeeze_expected_if_needed(out[stage], oracle[key])
        diagnostics[stage] = add_metrics(out[stage], expected, tolerance=1e-4)
    meta = load_json(META)
    shortcut_status = 'PASS' if meta.get('resnet1_config', {}).get('has_shortcut') else 'NOT_APPLICABLE'
    shortcut = {'status': shortcut_status, 'target': 'down_blocks.0.resnets.1 shortcut', 'note': 'Identity shortcut is expected when no conv_shortcut exists.'}
    if shortcut_status == 'PASS':
        shortcut['metrics'] = add_metrics(out['shortcut'], oracle['entry_downblock0_resnet1_shortcut_output'], tolerance=1e-4)
        shortcut['status'] = status_from_metrics(shortcut['metrics'])
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'entry->resnet0->attention0 hidden/temb -> down_blocks.0.resnets.1 only',
        'note': 'Does not enter attentions.1, downsampler, full down_blocks.0, or full UNet.',
        'diagnostics': diagnostics,
        'weights': str(RESNET1_WEIGHTS),
        'shortcut_status': shortcut['status'],
    }
    write_report('jittor_unet_downblock0_resnet1_alignment', 'TADSR UNet down_blocks.0.resnets.1 Alignment', result)
    write_report('jittor_unet_downblock0_resnet1_shortcut_alignment', 'TADSR UNet down_blocks.0.resnets.1 Shortcut Alignment', shortcut)
    for stage, key, marker in mapping:
        print(f"{marker}: {status_from_metrics(diagnostics[stage])}")
    print(f"TADSR_UNET_DOWNBLOCK0_RESNET1_SHORTCUT_ALIGNMENT: {shortcut['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} and shortcut['status'] in {'PASS', 'NOT_APPLICABLE'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
