#!/usr/bin/env python3
from __future__ import annotations
from unet_downblock0_resnet0_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_DOWNBLOCK0_RESNET0_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_downblock0_resnet0_alignment', 'TADSR UNet down_blocks.0.resnets.0 Alignment', blocked)
        return 1
    t = tester()
    out = t.run_downblock0_resnet0(oracle['entry_downblock0_resnet0_input'], oracle['entry_downblock0_resnet0_temb'], return_intermediates=True)
    mapping = [
        ('norm1', 'entry_downblock0_resnet0_norm1_output', 'TADSR_UNET_DOWNBLOCK0_RESNET0_NORM1_ALIGNMENT'),
        ('conv1', 'entry_downblock0_resnet0_conv1_output', 'TADSR_UNET_DOWNBLOCK0_RESNET0_CONV1_ALIGNMENT'),
        ('time_emb_proj', 'entry_downblock0_resnet0_time_emb_proj_output', 'TADSR_UNET_DOWNBLOCK0_RESNET0_TEMB_PROJ_ALIGNMENT'),
        ('conv2', 'entry_downblock0_resnet0_conv2_output', 'TADSR_UNET_DOWNBLOCK0_RESNET0_CONV2_ALIGNMENT'),
        ('output', 'entry_downblock0_resnet0_output', 'TADSR_UNET_DOWNBLOCK0_RESNET0_ALIGNMENT'),
    ]
    diagnostics = {}
    for stage, key, _ in mapping:
        actual = out[stage]
        expected = oracle[key]
        if stage == 'time_emb_proj' and getattr(actual, 'ndim', 0) == 2 and getattr(expected, 'ndim', 0) == 4 and expected.shape[2:] == (1, 1):
            expected = expected[:, :, 0, 0]
        diagnostics[stage] = add_metrics(actual, expected, tolerance=1e-4)
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'entry conv_in/time_embedding hidden -> down_blocks.0.resnets.0 only',
        'note': 'Does not enter attentions.0, resnets.1, downsampler, full down_blocks.0, or full UNet.',
        'diagnostics': diagnostics,
        'weights': str(RESNET_WEIGHTS),
    }
    write_report('jittor_unet_downblock0_resnet0_alignment', 'TADSR UNet down_blocks.0.resnets.0 Alignment', result)
    meta = load_json(META)
    shortcut_status = 'PASS' if meta.get('resnet0_config', {}).get('has_shortcut') else 'NOT_APPLICABLE'
    shortcut = {'status': shortcut_status, 'target': 'down_blocks.0.resnets.0 shortcut', 'note': 'Identity shortcut is expected when no conv_shortcut exists.'}
    if shortcut_status == 'PASS':
        shortcut['metrics'] = add_metrics(out['shortcut'], oracle['entry_downblock0_resnet0_shortcut_output'], tolerance=1e-4)
        shortcut['status'] = status_from_metrics(shortcut['metrics'])
    write_report('jittor_unet_downblock0_resnet0_shortcut_alignment', 'TADSR UNet down_blocks.0.resnets.0 Shortcut Alignment', shortcut)
    for stage, key, marker in mapping:
        print(f"{marker}: {status_from_metrics(diagnostics[stage])}")
    print(f"TADSR_UNET_DOWNBLOCK0_RESNET0_SHORTCUT_ALIGNMENT: {shortcut['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} and shortcut['status'] in {'PASS', 'NOT_APPLICABLE'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
