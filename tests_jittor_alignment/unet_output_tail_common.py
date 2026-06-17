#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jittor_tadsr_full.unet_2d_condition import TADSRUNetOutputTailTester
from unet_upblock3_local_common import *  # noqa: F401,F403
from unet_upblock3_local_common import load_oracle as _load_upblock3_local_oracle
from unet_upblock3_local_common import metadata as _upblock3_local_metadata

ORACLE = OUT / 'oracle_tensors_unet_output_tail'
META = ORACLE / 'unet_output_tail_oracle_metadata.json'
TAIL_WEIGHTS = OUT / 'converted_unet_output_tail_effective_weights.npz'


def load_oracle():
    _prev, blocked = _load_upblock3_local_oracle()
    if blocked:
        return None, blocked
    required = [ORACLE, META, TAIL_WEIGHTS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, {
            'status': 'BLOCKED',
            'reason': 'Run python3 tools/export_tadsr_unet_output_tail_oracle.py first.',
            'missing': missing,
        }
    tensors = {p.stem: np.load(p).astype(np.float32) for p in ORACLE.glob('*.npy')}
    if not tensors:
        return None, {
            'status': 'BLOCKED',
            'reason': 'Run python3 tools/export_tadsr_unet_output_tail_oracle.py first.',
        }
    return tensors, None


def metadata():
    meta = _upblock3_local_metadata()
    meta['output_tail_metadata'] = load_json(META)
    return meta


def output_tail_tester():
    return TADSRUNetOutputTailTester(
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
        upblock2_attention0_weights_npz_path=UP2_ATTENTION0_WEIGHTS,
        upblock2_resnet1_weights_npz_path=UP2_RESNET1_WEIGHTS,
        upblock2_attention1_weights_npz_path=UP2_ATTENTION1_WEIGHTS,
        upblock2_resnet2_weights_npz_path=UP2_RESNET2_WEIGHTS,
        upblock2_attention2_weights_npz_path=UP2_ATTENTION2_WEIGHTS,
        upblock2_upsampler_weights_npz_path=UP2_UPSAMPLER_WEIGHTS,
        upblock3_resnet0_weights_npz_path=UP3_RESNET0_WEIGHTS,
        upblock3_attention0_weights_npz_path=UP3_ATTENTION0_WEIGHTS,
        upblock3_resnet1_weights_npz_path=UP3_RESNET1_WEIGHTS,
        upblock3_attention1_weights_npz_path=UP3_ATTENTION1_WEIGHTS,
        upblock3_resnet2_weights_npz_path=UP3_RESNET2_WEIGHTS,
        upblock3_attention2_weights_npz_path=UP3_ATTENTION2_WEIGHTS,
        output_tail_weights_npz_path=TAIL_WEIGHTS,
        metadata=metadata(),
    )
