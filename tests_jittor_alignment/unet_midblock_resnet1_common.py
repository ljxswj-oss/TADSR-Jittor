#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import UNetResnetBlock2DTester, TADSRUNetMidBlockResnet1Tester
from unet_midblock_attention0_common import (
    OUT,
    ENTRY_WEIGHTS,
    DOWN0_RESNET0_WEIGHTS, DOWN0_ATTENTION0_WEIGHTS, DOWN0_RESNET1_WEIGHTS, DOWN0_ATTENTION1_WEIGHTS, DOWN0_DOWNSAMPLER_WEIGHTS,
    DOWN1_RESNET0_WEIGHTS, DOWN1_ATTENTION0_WEIGHTS, DOWN1_RESNET1_WEIGHTS, DOWN1_ATTENTION1_WEIGHTS, DOWN1_DOWNSAMPLER_WEIGHTS,
    DOWN2_RESNET0_WEIGHTS, DOWN2_ATTENTION0_WEIGHTS, DOWN2_RESNET1_WEIGHTS, DOWN2_ATTENTION1_WEIGHTS, DOWN2_DOWNSAMPLER_WEIGHTS,
    DOWN3_RESNET0_WEIGHTS, DOWN3_RESNET1_WEIGHTS,
    MID_RESNET0_WEIGHTS, MID_ATTENTION0_WEIGHTS,
    add_metrics, blocked_result, cosine, load_json, status_from_diagnostics, status_from_metrics, write_report,
)
from unet_midblock_resnet0_common import squeeze_expected_if_needed


ORACLE = OUT / 'oracle_tensors_unet_midblock_resnet1'
META = ORACLE / 'unet_midblock_resnet1_oracle_metadata.json'
MID_RESNET1_WEIGHTS = OUT / 'converted_unet_midblock_resnet1_effective_weights.npz'


def load_oracle():
    required = [
        ORACLE, META, ENTRY_WEIGHTS,
        DOWN0_RESNET0_WEIGHTS, DOWN0_ATTENTION0_WEIGHTS, DOWN0_RESNET1_WEIGHTS, DOWN0_ATTENTION1_WEIGHTS, DOWN0_DOWNSAMPLER_WEIGHTS,
        DOWN1_RESNET0_WEIGHTS, DOWN1_ATTENTION0_WEIGHTS, DOWN1_RESNET1_WEIGHTS, DOWN1_ATTENTION1_WEIGHTS, DOWN1_DOWNSAMPLER_WEIGHTS,
        DOWN2_RESNET0_WEIGHTS, DOWN2_ATTENTION0_WEIGHTS, DOWN2_RESNET1_WEIGHTS, DOWN2_ATTENTION1_WEIGHTS, DOWN2_DOWNSAMPLER_WEIGHTS,
        DOWN3_RESNET0_WEIGHTS, DOWN3_RESNET1_WEIGHTS, MID_RESNET0_WEIGHTS, MID_ATTENTION0_WEIGHTS, MID_RESNET1_WEIGHTS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py first.', 'missing': missing}
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_midblock_resnet1_oracle.py first.'}
    return tensors, None


def metadata():
    from unet_midblock_attention0_common import metadata as attention0_metadata
    meta = attention0_metadata()
    meta['midblock_resnet1_metadata'] = load_json(META)
    return meta


def resnet1_tester():
    meta = load_json(META)
    return UNetResnetBlock2DTester(MID_RESNET1_WEIGHTS, meta.get('resnet1_config', {}).get('prefix', 'mid_block_resnets_1'), meta)


def bridge_tester():
    return TADSRUNetMidBlockResnet1Tester(
        ENTRY_WEIGHTS,
        DOWN0_RESNET0_WEIGHTS,
        DOWN0_ATTENTION0_WEIGHTS,
        DOWN0_RESNET1_WEIGHTS,
        DOWN0_ATTENTION1_WEIGHTS,
        DOWN0_DOWNSAMPLER_WEIGHTS,
        DOWN1_RESNET0_WEIGHTS,
        DOWN1_ATTENTION0_WEIGHTS,
        DOWN1_RESNET1_WEIGHTS,
        DOWN1_ATTENTION1_WEIGHTS,
        DOWN1_DOWNSAMPLER_WEIGHTS,
        DOWN2_RESNET0_WEIGHTS,
        DOWN2_ATTENTION0_WEIGHTS,
        DOWN2_RESNET1_WEIGHTS,
        DOWN2_ATTENTION1_WEIGHTS,
        DOWN2_DOWNSAMPLER_WEIGHTS,
        DOWN3_RESNET0_WEIGHTS,
        DOWN3_RESNET1_WEIGHTS,
        MID_RESNET0_WEIGHTS,
        MID_ATTENTION0_WEIGHTS,
        MID_RESNET1_WEIGHTS,
        metadata=metadata(),
    )
