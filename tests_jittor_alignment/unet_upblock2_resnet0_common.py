#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import TADSRUNetUpBlock2Resnet0Tester
from unet_upblock1_local_common import (
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
    UP1_RESNET1_WEIGHTS,
    UP1_ATTENTION1_WEIGHTS,
    UP1_RESNET2_WEIGHTS,
    UP1_ATTENTION2_WEIGHTS,
    UP1_UPSAMPLER_WEIGHTS,
    add_metrics,
    blocked_result,
    load_json,
    status_from_diagnostics,
    status_from_metrics,
    write_report,
)
from unet_upblock1_upsampler_common import metadata as upblock1_upsampler_metadata

ORACLE = OUT / 'oracle_tensors_unet_upblock2_resnet0'
META = ORACLE / 'unet_upblock2_resnet0_oracle_metadata.json'
UP2_RESNET0_WEIGHTS = OUT / 'converted_unet_upblock2_resnet0_effective_weights.npz'
UP1_LOCAL_META = OUT / 'oracle_tensors_unet_upblock1_local' / 'unet_upblock1_local_oracle_metadata.json'


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
        UP1_ATTENTION1_WEIGHTS,
        UP1_RESNET2_WEIGHTS,
        UP1_ATTENTION2_WEIGHTS,
        UP1_UPSAMPLER_WEIGHTS,
        UP2_RESNET0_WEIGHTS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {
            'status': 'BLOCKED',
            'reason': 'Run python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py first.',
            'missing': missing,
        }
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {'status': 'BLOCKED', 'reason': 'Run python3 tools/export_tadsr_unet_upblock2_resnet0_oracle.py first.'}
    return tensors, None


def metadata():
    meta = upblock1_upsampler_metadata()
    if UP1_LOCAL_META.exists():
        meta['upblock1_local_metadata'] = load_json(UP1_LOCAL_META)
    meta['upblock2_resnet0_metadata'] = load_json(META)
    return meta


def bridge_tester():
    return TADSRUNetUpBlock2Resnet0Tester(
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
        upblock1_attention1_weights_npz_path=UP1_ATTENTION1_WEIGHTS,
        upblock1_resnet2_weights_npz_path=UP1_RESNET2_WEIGHTS,
        upblock1_attention2_weights_npz_path=UP1_ATTENTION2_WEIGHTS,
        upblock1_upsampler_weights_npz_path=UP1_UPSAMPLER_WEIGHTS,
        upblock2_resnet0_weights_npz_path=UP2_RESNET0_WEIGHTS,
        metadata=metadata(),
    )
