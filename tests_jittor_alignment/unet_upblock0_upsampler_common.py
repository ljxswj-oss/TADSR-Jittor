#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import UNetUpsample2DTester, TADSRUNetUpBlock0UpsamplerTester
from unet_upblock0_resnet2_common import (
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
    add_metrics,
    blocked_result,
    cosine,
    load_json,
    squeeze_expected_if_needed,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)
from unet_upblock0_resnet2_common import metadata as resnet2_metadata

ORACLE = OUT / 'oracle_tensors_unet_upblock0_upsampler'
META = ORACLE / 'unet_upblock0_upsampler_oracle_metadata.json'
UP0_UPSAMPLER_WEIGHTS = OUT / 'converted_unet_upblock0_upsampler_effective_weights.npz'


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
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py first.', 'missing': missing}
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock0_upsampler_oracle.py first.'}
    return tensors, None


def metadata():
    meta = resnet2_metadata()
    meta['upblock0_upsampler_metadata'] = load_json(META)
    return meta


def upsampler_tester():
    meta = load_json(META)
    return UNetUpsample2DTester(UP0_UPSAMPLER_WEIGHTS, meta.get('upsampler_config', {}).get('prefix', 'up_blocks_0_upsamplers_0'), meta)


def bridge_tester():
    return TADSRUNetUpBlock0UpsamplerTester(
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
        metadata=metadata(),
    )
