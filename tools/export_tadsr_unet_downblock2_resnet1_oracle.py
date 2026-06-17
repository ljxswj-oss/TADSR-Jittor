#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.export_tadsr_unet_downblock2_resnet0_oracle import (
    OFFICIAL_REPO,
    OUT_DIR,
    WEIGHTS_DIR,
    load_unet,
    max_abs,
    module_effective_arrays,
    resnet_config,
    resnet_manual,
    stats,
    to_np,
)

ORACLE_DIR = OUT_DIR / "oracle_tensors_unet_downblock2_resnet1"
META_JSON = ORACLE_DIR / "unet_downblock2_resnet1_oracle_metadata.json"
SUMMARY_TXT = ORACLE_DIR / "oracle_summary.txt"
EFFECTIVE_WEIGHTS = OUT_DIR / "converted_unet_downblock2_resnet1_effective_weights.npz"
PREV_ATTENTION0_ORACLE = OUT_DIR / "oracle_tensors_unet_downblock2_attention0" / "entry_downblock2_attention0_output.npy"

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def save_tensor(saved, name, tensor):
    arr = to_np(tensor)
    np.save(ORACLE_DIR / f"{name}.npy", arr)
    saved[name] = stats(arr)
    return arr


def main() -> int:
    maybe_reexec()
    import torch

    torch.manual_seed(1234)
    np.random.seed(1234)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    resnet0 = block2.resnets[0]
    attention0 = block2.attentions[0]
    resnet1 = block2.resnets[1]

    sample_shape = [1, int(unet.config.in_channels), 32, 32]
    sample = torch.linspace(-1.0, 1.0, steps=int(np.prod(sample_shape)), dtype=torch.float32).reshape(*sample_shape)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(-0.5, 0.5, steps=int(np.prod(encoder_shape)), dtype=torch.float32).reshape(*encoder_shape)

    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        time_embedding = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, time_embedding, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, time_embedding, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = resnet0(block1_hidden, time_embedding)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

        x_shape = list(attention0_out.shape)
        temb_shape = [1, int(resnet1.time_emb_proj.weight.shape[1])]
        synthetic_x = torch.linspace(-1.0, 1.0, steps=int(np.prod(x_shape)), dtype=torch.float32).reshape(*x_shape)
        synthetic_temb = torch.linspace(-0.5, 0.5, steps=int(np.prod(temb_shape)), dtype=torch.float32).reshape(*temb_shape)
        synthetic_inter = resnet_manual(resnet1, synthetic_x, synthetic_temb)
        synthetic_official = resnet1(synthetic_x, synthetic_temb)
        entry_inter = resnet_manual(resnet1, attention0_out, time_embedding)
        entry_official = resnet1(attention0_out, time_embedding)

    saved = {}
    save_tensor(saved, "synthetic_downblock2_resnet1_input", synthetic_x)
    save_tensor(saved, "synthetic_downblock2_resnet1_temb", synthetic_temb)
    for k, v in synthetic_inter.items():
        save_tensor(saved, f"synthetic_downblock2_resnet1_{k}_output" if k != "output" else "synthetic_downblock2_resnet1_manual_output", v)
    save_tensor(saved, "synthetic_downblock2_resnet1_output", synthetic_official)

    save_tensor(saved, "entry_synthetic_unet_sample", sample)
    save_tensor(saved, "entry_synthetic_unet_timestep", timestep)
    save_tensor(saved, "entry_encoder_hidden_states", encoder_hidden_states)
    save_tensor(saved, "entry_synthetic_unet_sample_after_center", centered)
    save_tensor(saved, "entry_synthetic_unet_conv_in_output", conv_in)
    save_tensor(saved, "entry_synthetic_unet_time_proj_output", time_proj)
    save_tensor(saved, "entry_synthetic_unet_time_embedding_output", time_embedding)
    save_tensor(saved, "entry_downblock0_output_hidden", block0_hidden)
    for i, state in enumerate(block0_states):
        save_tensor(saved, f"entry_downblock0_output_state_{i}", state)
    save_tensor(saved, "entry_downblock1_output_hidden", block1_hidden)
    for i, state in enumerate(block1_states):
        save_tensor(saved, f"entry_downblock1_output_state_{i}", state)
    save_tensor(saved, "entry_downblock2_resnet0_output", resnet0_out)
    save_tensor(saved, "entry_downblock2_attention0_output", attention0_out)
    save_tensor(saved, "entry_downblock2_resnet1_input", attention0_out)
    save_tensor(saved, "entry_downblock2_resnet1_temb", time_embedding)
    for k, v in entry_inter.items():
        save_tensor(saved, f"entry_downblock2_resnet1_{k}_output" if k != "output" else "entry_downblock2_resnet1_manual_output", v)
    save_tensor(saved, "entry_downblock2_resnet1_output", entry_official)

    effective = {}
    effective_meta = {}
    prefix = "down_blocks_2_resnets_1"
    for name in ["norm1", "norm2"]:
        mod = getattr(resnet1, name)
        effective[f"{prefix}_{name}_weight"] = mod.weight.detach().cpu().numpy().astype(np.float32)
        effective[f"{prefix}_{name}_bias"] = mod.bias.detach().cpu().numpy().astype(np.float32)
        effective_meta[f"{prefix}_{name}"] = {"is_lora_wrapped": False, "weight_shape": list(mod.weight.shape), "bias_shape": list(mod.bias.shape)}
    for name in ["conv1", "time_emb_proj", "conv2"]:
        module_effective_arrays(f"{prefix}_{name}", getattr(resnet1, name), effective, effective_meta)
    if getattr(resnet1, "conv_shortcut", None) is not None:
        module_effective_arrays(f"{prefix}_conv_shortcut", resnet1.conv_shortcut, effective, effective_meta)
    np.savez_compressed(EFFECTIVE_WEIGHTS, **effective)

    synth_diff = (synthetic_inter["output"] - synthetic_official).abs()
    entry_diff = (entry_inter["output"] - entry_official).abs()
    previous_attention0_compare = {"previous_downblock2_attention0_oracle_tensor_available": False}
    if PREV_ATTENTION0_ORACLE.exists():
        prev = np.load(PREV_ATTENTION0_ORACLE)
        curr = to_np(attention0_out)
        previous_attention0_compare = {
            "previous_downblock2_attention0_oracle_tensor_available": True,
            "max_abs_diff": max_abs(prev, curr),
            "mean_abs_diff": float(np.mean(np.abs(prev.astype(np.float32) - curr.astype(np.float32)))),
        }
    metadata = {
        "status": "PASS",
        "python": sys.executable,
        "official_repo": str(OFFICIAL_REPO),
        "weights_dir": str(WEIGHTS_DIR),
        "oracle_dir": str(ORACLE_DIR),
        "effective_weights": str(EFFECTIVE_WEIGHTS),
        "selected_timestep": [1],
        "entry_sample_shape": sample_shape,
        "encoder_hidden_states_shape": encoder_shape,
        "downblock0_output_state_shapes": [list(x.shape) for x in block0_states],
        "downblock1_output_state_shapes": [list(x.shape) for x in block1_states],
        "accumulated_down_block_res_sample_shapes": [list(conv_in.shape)] + [list(x.shape) for x in block0_states] + [list(x.shape) for x in block1_states],
        "resnet1_input_shape": list(attention0_out.shape),
        "resnet1_output_shape": list(entry_official.shape),
        "temb_shape": list(time_embedding.shape),
        "resnet1_config": resnet_config(resnet1, prefix),
        "saved_tensors": saved,
        "effective_weight_metadata": effective_meta,
        "manual_vs_official": {
            "synthetic_max_abs_diff": float(synth_diff.max().item()) if synth_diff.numel() else 0.0,
            "synthetic_mean_abs_diff": float(synth_diff.mean().item()) if synth_diff.numel() else 0.0,
            "entry_max_abs_diff": float(entry_diff.max().item()) if entry_diff.numel() else 0.0,
            "entry_mean_abs_diff": float(entry_diff.mean().item()) if entry_diff.numel() else 0.0,
        },
        "previous_downblock2_attention0_compare": previous_attention0_compare,
        "residual_preview": {
            "resnet1_output_is_immediate_downblock2_residual_sample": False,
            "reason": "CrossAttnDownBlock2D appends residual after attention1, not immediately after resnet1.",
        },
        "markers": {
            "TADSR_UNET_DOWNBLOCK2_RESNET1_ORACLE_TENSORS": "PASS",
            "TADSR_UNET_DOWNBLOCK2_RESNET1_EFFECTIVE_WEIGHTS": "PASS",
        },
        "scope_boundaries": {
            "ported_this_stage": "down_blocks.2.resnets.1",
            "not_ported_this_stage": ["down_blocks.2.attentions.1", "down_blocks.2.downsamplers.0", "mid_block", "up_blocks", "full UNet forward"],
        },
    }
    META_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    SUMMARY_TXT.write_text(
        "\n".join([
            "# TADSR UNet down_blocks.2.resnets.1 oracle export",
            "",
            "TADSR_UNET_DOWNBLOCK2_RESNET1_ORACLE_TENSORS: PASS",
            "TADSR_UNET_DOWNBLOCK2_RESNET1_EFFECTIVE_WEIGHTS: PASS",
            "",
            f"effective weights: {EFFECTIVE_WEIGHTS}",
            f"oracle tensors: {ORACLE_DIR}",
            f"synthetic manual/offical max abs diff: {metadata['manual_vs_official']['synthetic_max_abs_diff']}",
            f"entry manual/offical max abs diff: {metadata['manual_vs_official']['entry_max_abs_diff']}",
            f"previous downblock2 attention0 compare: {previous_attention0_compare}",
        ]) + "\n",
        encoding="utf-8",
    )
    for k, v in metadata["markers"].items():
        print(f"{k}: {v}")
    print(json.dumps({"status": metadata["status"], "metadata": str(META_JSON), "effective_weights": str(EFFECTIVE_WEIGHTS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
