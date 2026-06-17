#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from unet_full_forward_common import (
    add_metrics,
    blocked_result,
    full_forward_tester,
    load_oracle,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)


def main() -> int:
    tensors, blocked = load_oracle()
    name = 'jittor_unet_full_forward_alignment'
    title = 'TADSR UNet Full Forward Alignment'
    if blocked:
        result = blocked_result(name, title, blocked)
        for marker in [
            'TADSR_UNET_JITTOR_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT',
            'TADSR_UNET_JITTOR_VS_MANUAL_WRAPPER_ALIGNMENT',
            'TADSR_UNET_FULL_FORWARD_RETURN_CONTRACT_ALIGNMENT',
            'TADSR_UNET_FULL_FORWARD_ALIGNMENT',
        ]:
            print(f'{marker}:', result['status'])
        return 0 if result['status'] == 'PASS' else 1

    tester = full_forward_tester()
    got = tester.run_full_forward_for_alignment(
        tensors['full_forward_sample'],
        tensors['full_forward_timestep'].astype('int64'),
        tensors['full_forward_encoder_hidden_states'],
        return_dict=False,
        return_intermediates=True,
    )
    final = got['full_forward_alignment_output']
    dict_out = tester.run_full_forward_for_alignment(
        tensors['full_forward_sample'],
        tensors['full_forward_timestep'].astype('int64'),
        tensors['full_forward_encoder_hidden_states'],
        return_dict=True,
        return_intermediates=False,
    )
    tensor_out = tester.run_full_forward_for_alignment(
        tensors['full_forward_sample'],
        tensors['full_forward_timestep'].astype('int64'),
        tensors['full_forward_encoder_hidden_states'],
        return_dict=False,
        return_intermediates=False,
    )
    diag = {
        'jittor_vs_official': add_metrics(final, tensors['official_full_forward_output'], tolerance=2e-3),
        'jittor_vs_manual': add_metrics(final, tensors['manual_wrapper_output'], tolerance=2e-3),
        'tensor_return_vs_intermediate': add_metrics(tensor_out, final, tolerance=0.0),
    }
    return_contract_ok = isinstance(dict_out, dict) and 'sample' in dict_out and np.asarray(dict_out['sample']).shape == final.shape
    diag['dict_return_vs_intermediate'] = add_metrics(dict_out['sample'] if return_contract_ok else np.zeros_like(final), final, tolerance=0.0)
    official_status = status_from_metrics(diag['jittor_vs_official'])
    manual_status = status_from_metrics(diag['jittor_vs_manual'])
    return_status = 'PASS' if return_contract_ok and status_from_metrics(diag['tensor_return_vs_intermediate']) == 'PASS' and status_from_metrics(diag['dict_return_vs_intermediate']) == 'PASS' else 'FAIL'
    status = status_from_diagnostics({
        'jittor_vs_official': diag['jittor_vs_official'],
        'jittor_vs_manual': diag['jittor_vs_manual'],
        'tensor_return_vs_intermediate': diag['tensor_return_vs_intermediate'],
        'dict_return_vs_intermediate': diag['dict_return_vs_intermediate'],
    }) if return_status == 'PASS' else 'FAIL'
    result = {
        'status': status,
        'diagnostics': diag,
        'jittor_vs_official_status': official_status,
        'jittor_vs_manual_status': manual_status,
        'return_contract_status': return_status,
        'note': (
            'Alignment-only full forward wrapper matches the official PyTorch UNet.forward oracle. '
            'This is UNet numerical alignment only; it does not run scheduler, VAE, full TADSR inference, '
            'image generation, or generic runtime LoRA.'
        ),
    }
    write_report(name, title, result)
    print('TADSR_UNET_JITTOR_VS_OFFICIAL_FULL_FORWARD_ALIGNMENT:', official_status)
    print('TADSR_UNET_JITTOR_VS_MANUAL_WRAPPER_ALIGNMENT:', manual_status)
    print('TADSR_UNET_FULL_FORWARD_RETURN_CONTRACT_ALIGNMENT:', return_status)
    print('TADSR_UNET_FULL_FORWARD_ALIGNMENT:', status)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
