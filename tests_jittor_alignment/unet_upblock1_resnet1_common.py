#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import UNetResnetBlock2DTester, TADSRUNetUpBlock1Resnet1Tester
from unet_upblock1_attention0_common import (
    OUT,
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
    UP0_RESNET1_WEIGHTS,
    UP0_RESNET2_WEIGHTS,
    UP0_UPSAMPLER_WEIGHTS,
    UP1_RESNET0_WEIGHTS,
    UP1_ATTENTION0_WEIGHTS,
    add_metrics,
    blocked_result,
    load_json,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)
from unet_upblock1_attention0_common import metadata as upblock1_attention0_metadata
from unet_upblock1_resnet0_common import squeeze_expected_if_needed

ORACLE = OUT / 'oracle_tensors_unet_upblock1_resnet1'
META = ORACLE / 'unet_upblock1_resnet1_oracle_metadata.json'
UP1_RESNET1_WEIGHTS = OUT / 'converted_unet_upblock1_resnet1_effective_weights.npz'


def load_oracle():
    required = [
        ORACLE,
        META,
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
        UP0_RESNET1_WEIGHTS,
        UP0_RESNET2_WEIGHTS,
        UP0_UPSAMPLER_WEIGHTS,
        UP1_RESNET0_WEIGHTS,
        UP1_ATTENTION0_WEIGHTS,
        UP1_RESNET1_WEIGHTS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {
            'status': 'BLOCKED',
            'reason': 'Run python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py first.',
            'missing': missing,
        }
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock1_resnet1_oracle.py first.'}
    return tensors, None


def metadata():
    meta = upblock1_attention0_metadata()
    meta['upblock1_resnet1_metadata'] = load_json(META)
    return meta


def resnet1_tester():
    meta = load_json(META)
    prefix = meta.get('resnet1_config', {}).get('prefix', 'up_blocks_1_resnets_1')
    return UNetResnetBlock2DTester(UP1_RESNET1_WEIGHTS, prefix, meta)


def bridge_tester():
    return TADSRUNetUpBlock1Resnet1Tester(
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
        upblock0_resnet1_weights_npz_path=UP0_RESNET1_WEIGHTS,
        upblock0_resnet2_weights_npz_path=UP0_RESNET2_WEIGHTS,
        upblock0_upsampler_weights_npz_path=UP0_UPSAMPLER_WEIGHTS,
        upblock1_resnet0_weights_npz_path=UP1_RESNET0_WEIGHTS,
        upblock1_attention0_weights_npz_path=UP1_ATTENTION0_WEIGHTS,
        upblock1_resnet1_weights_npz_path=UP1_RESNET1_WEIGHTS,
        metadata=metadata(),
    )
