#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import UNetResnetBlock2DTester, TADSRUNetUpBlock0Resnet0Tester
from unet_midblock_resnet1_common import (
    OUT,
    ENTRY_WEIGHTS,
    DOWN0_RESNET0_WEIGHTS, DOWN0_ATTENTION0_WEIGHTS, DOWN0_RESNET1_WEIGHTS, DOWN0_ATTENTION1_WEIGHTS, DOWN0_DOWNSAMPLER_WEIGHTS,
    DOWN1_RESNET0_WEIGHTS, DOWN1_ATTENTION0_WEIGHTS, DOWN1_RESNET1_WEIGHTS, DOWN1_ATTENTION1_WEIGHTS, DOWN1_DOWNSAMPLER_WEIGHTS,
    DOWN2_RESNET0_WEIGHTS, DOWN2_ATTENTION0_WEIGHTS, DOWN2_RESNET1_WEIGHTS, DOWN2_ATTENTION1_WEIGHTS, DOWN2_DOWNSAMPLER_WEIGHTS,
    DOWN3_RESNET0_WEIGHTS, DOWN3_RESNET1_WEIGHTS,
    MID_RESNET0_WEIGHTS, MID_ATTENTION0_WEIGHTS, MID_RESNET1_WEIGHTS,
    add_metrics, blocked_result, cosine, load_json, status_from_diagnostics, status_from_metrics, write_report,
)
from unet_midblock_resnet0_common import squeeze_expected_if_needed


ORACLE = OUT / 'oracle_tensors_unet_upblock0_resnet0'
META = ORACLE / 'unet_upblock0_resnet0_oracle_metadata.json'
UP0_RESNET0_WEIGHTS = OUT / 'converted_unet_upblock0_resnet0_effective_weights.npz'


def load_oracle():
    required = [
        ORACLE, META, ENTRY_WEIGHTS,
        DOWN0_RESNET0_WEIGHTS, DOWN0_ATTENTION0_WEIGHTS, DOWN0_RESNET1_WEIGHTS, DOWN0_ATTENTION1_WEIGHTS, DOWN0_DOWNSAMPLER_WEIGHTS,
        DOWN1_RESNET0_WEIGHTS, DOWN1_ATTENTION0_WEIGHTS, DOWN1_RESNET1_WEIGHTS, DOWN1_ATTENTION1_WEIGHTS, DOWN1_DOWNSAMPLER_WEIGHTS,
        DOWN2_RESNET0_WEIGHTS, DOWN2_ATTENTION0_WEIGHTS, DOWN2_RESNET1_WEIGHTS, DOWN2_ATTENTION1_WEIGHTS, DOWN2_DOWNSAMPLER_WEIGHTS,
        DOWN3_RESNET0_WEIGHTS, DOWN3_RESNET1_WEIGHTS, MID_RESNET0_WEIGHTS, MID_ATTENTION0_WEIGHTS, MID_RESNET1_WEIGHTS, UP0_RESNET0_WEIGHTS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py first.', 'missing': missing}
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock0_resnet0_oracle.py first.'}
    return tensors, None


def metadata():
    from unet_midblock_resnet1_common import metadata as midblock_metadata
    meta = midblock_metadata()
    meta['upblock0_resnet0_metadata'] = load_json(META)
    return meta


def resnet0_tester():
    meta = load_json(META)
    return UNetResnetBlock2DTester(UP0_RESNET0_WEIGHTS, meta.get('resnet0_config', {}).get('prefix', 'up_blocks_0_resnets_0'), meta)


def bridge_tester():
    return TADSRUNetUpBlock0Resnet0Tester(
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
        UP0_RESNET0_WEIGHTS,
        metadata=metadata(),
    )
