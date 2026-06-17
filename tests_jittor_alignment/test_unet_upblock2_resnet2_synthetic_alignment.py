#!/usr/bin/env python3
from __future__ import annotations

from unet_upblock2_resnet2_common import (
    add_metrics,
    blocked_result,
    bridge_tester,
    load_oracle,
    status_from_diagnostics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_upblock2_resnet2_synthetic_alignment'
    title = 'TADSR UNet up_blocks.2.resnets.2 Synthetic Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        print('TADSR_UNET_UPBLOCK2_RESNET2_SYNTHETIC_ALIGNMENT:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = bridge_tester()
    got = tester.run_upblock2_resnet2(
        tensors['synthetic_upblock2_resnet2_hidden_input'],
        tensors['synthetic_upblock2_resnet2_res_hidden'],
        tensors['synthetic_upblock2_resnet2_temb'],
        return_intermediates=True,
    )
    diag = {
        'concat_input': add_metrics(got['concat_input'], tensors['synthetic_upblock2_resnet2_concat_input'], tolerance=1e-6),
        'output': add_metrics(got['output'], tensors['synthetic_upblock2_resnet2_output'], tolerance=2e-4),
    }
    status = status_from_diagnostics(diag)
    result = {
        'status': status,
        'diagnostics': diag,
        'note': (
            'Synthetic isolated check for only up_blocks.2.resnets.2. '
            'It stops before up_blocks.2.attentions.2, full up_blocks.2, '
            'full UNet forward, runtime LoRA, and full TADSR inference.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_UPBLOCK2_RESNET2_SYNTHETIC_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
