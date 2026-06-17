#!/usr/bin/env python3
from __future__ import annotations
from unet_entry_common import *


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        print(f"TADSR_UNET_TIME_EMBED_ALIGNMENT: {blocked['status']}")
        blocked_result('jittor_unet_time_embedding_alignment', 'TADSR UNet time_embedding Alignment', blocked)
        return 1
    tester = UNetTimestepEmbeddingTester(WEIGHTS)
    time_proj = oracle['unet_time_proj_output']
    linear1 = tester.run_linear1(time_proj)
    act = tester.run_act(linear1)
    full = tester.run_linear2(act)
    diagnostics = {
        'linear1': add_metrics(linear1, oracle['unet_time_embedding_linear1_output'], tolerance=1e-4),
        'act': add_metrics(act, oracle['unet_time_embedding_act_output'], tolerance=1e-4),
        'full': add_metrics(full, oracle['unet_time_embedding_output'], tolerance=1e-3),
    }
    result = {
        'status': status_from_diagnostics(diagnostics),
        'target': 'UNet TimestepEmbedding linear_1 -> SiLU -> linear_2',
        'note': 'Tests only the timestep MLP; no down/up/mid/cross-attention blocks are executed.',
        'diagnostics': diagnostics,
        'weights': str(WEIGHTS),
    }
    write_report('jittor_unet_time_embedding_alignment', 'TADSR UNet time_embedding Alignment', result)
    print(f"TADSR_UNET_TIME_EMBED_LINEAR1_ALIGNMENT: {status_from_metrics(diagnostics['linear1'])}")
    print(f"TADSR_UNET_TIME_EMBED_ACT_ALIGNMENT: {status_from_metrics(diagnostics['act'])}")
    print(f"TADSR_UNET_TIME_EMBED_ALIGNMENT: {result['status']}")
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
