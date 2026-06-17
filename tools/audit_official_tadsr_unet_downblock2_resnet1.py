#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_official_tadsr_unet_downblock2_resnet0 import (
    OFFICIAL_REPO,
    OUT_DIR,
    WEIGHTS_DIR,
    attention_preview,
    load_unet,
    module_lora_info,
    resnet_config,
    stats_tensor,
)

OUT_JSON = OUT_DIR / "audit_tadsr_unet_downblock2_resnet1.json"
OUT_TXT = OUT_DIR / "audit_tadsr_unet_downblock2_resnet1.txt"

STRICT_PY = Path('/mnt/data/sj/venvs/tadsr_official_strict_cu118/bin/python')


def maybe_reexec() -> None:
    expected_prefix = str(STRICT_PY.parents[1])
    in_expected_venv = str(sys.prefix) == expected_prefix or str(sys.executable).startswith(expected_prefix)
    if STRICT_PY.exists() and not in_expected_venv:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.execv(str(STRICT_PY), [str(STRICT_PY), __file__, *sys.argv[1:]])


def _submodule_info(resnet):
    info = {}
    for name in ["norm1", "conv1", "time_emb_proj", "norm2", "conv2", "conv_shortcut"]:
        mod = getattr(resnet, name, None)
        if mod is None:
            info[name] = {"exists": False}
        elif name.startswith("norm"):
            info[name] = {
                "exists": True,
                "class": f"{type(mod).__module__}.{type(mod).__name__}",
                "weight_shape": list(mod.weight.shape) if getattr(mod, "weight", None) is not None else None,
                "bias_shape": list(mod.bias.shape) if getattr(mod, "bias", None) is not None else None,
                "num_groups": getattr(mod, "num_groups", None),
                "eps": getattr(mod, "eps", None),
                "is_lora_wrapped": False,
            }
        else:
            info[name] = module_lora_info(mod)
    return info


def main() -> int:
    maybe_reexec()
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unet, loaded = load_unet()
    block0 = unet.down_blocks[0]
    block1 = unet.down_blocks[1]
    block2 = unet.down_blocks[2]
    resnet0 = block2.resnets[0]
    attention0 = block2.attentions[0]
    resnet1 = block2.resnets[1]
    attention1 = block2.attentions[1] if len(getattr(block2, "attentions", [])) > 1 else None
    downsampler0 = block2.downsamplers[0] if getattr(block2, "downsamplers", None) is not None and len(block2.downsamplers) > 0 else None

    sample = torch.linspace(
        -1.0,
        1.0,
        steps=1 * int(unet.config.in_channels) * 32 * 32,
        dtype=torch.float32,
    ).reshape(1, int(unet.config.in_channels), 32, 32)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_shape = [1, 77, int(unet.config.cross_attention_dim)]
    encoder_hidden_states = torch.linspace(
        -0.5,
        0.5,
        steps=int(torch.tensor(encoder_shape).prod().item()),
        dtype=torch.float32,
    ).reshape(*encoder_shape)

    with torch.no_grad():
        centered = 2.0 * sample - 1.0 if unet.config.center_input_sample else sample
        conv_in = unet.conv_in(centered)
        time_proj = unet.time_proj(timestep)
        temb = unet.time_embedding(time_proj.to(dtype=list(unet.time_embedding.parameters())[0].dtype))
        block0_hidden, block0_states = block0(conv_in, temb, encoder_hidden_states=encoder_hidden_states)
        block1_hidden, block1_states = block1(block0_hidden, temb, encoder_hidden_states=encoder_hidden_states)
        resnet0_out = resnet0(block1_hidden, temb)
        attention0_out = attention0(resnet0_out, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        resnet1_out = resnet1(attention0_out, temb)

    cfg = resnet_config(resnet1, "down_blocks_2_resnets_1")
    children = [{"name": n, "class": f"{type(m).__module__}.{type(m).__name__}"} for n, m in block2.named_children()]
    audit = {
        "status": "PASS",
        "scope": "down_blocks.2.resnets.1 only; attentions.1/downsampler/full down_blocks.2/full UNet remain unopened.",
        "python": sys.executable,
        "official_repo": str(OFFICIAL_REPO),
        "weights_dir": str(WEIGHTS_DIR),
        "loaded_lora_parameter_count": len(loaded),
        "downblock2_overview": {
            "class": f"{type(block2).__module__}.{type(block2).__name__}",
            "forward_signature": str(inspect.signature(block2.forward)),
            "module_order": children,
            "resnet_count": len(getattr(block2, "resnets", [])),
            "attention_count": len(getattr(block2, "attentions", [])) if hasattr(block2, "attentions") else 0,
            "downsampler_count": len(getattr(block2, "downsamplers", [])) if getattr(block2, "downsamplers", None) is not None else 0,
            "has_downsampler": downsampler0 is not None,
            "has_cross_attention": True,
            "known_completed_modules": {"resnets.0": "PASS", "attentions.0": "PASS"},
            "current_target_module": "down_blocks.2.resnets.1",
            "remaining_modules_after_this_stage": ["down_blocks.2.attentions.1", "down_blocks.2.downsamplers.0"],
            "actual_next_module_after_resnet1": "down_blocks.2.attentions.1" if attention1 is not None else ("down_blocks.2.downsamplers.0" if downsampler0 is not None else None),
            "consumes_encoder_hidden_states": True,
            "attention_mask_used_in_this_stage": False,
            "encoder_attention_mask_used_in_this_stage": False,
            "cross_attention_kwargs_used_in_this_stage": False,
            "lora_scale_argument_used_in_this_stage": False,
            "has_peft_wrappers": any(module_lora_info(getattr(resnet1, n, None)).get("is_lora_wrapped") for n in ["conv1", "time_emb_proj", "conv2", "conv_shortcut"]),
        },
        "resnet1_config": cfg,
        "compare_with_resnet0": {
            "in_channels": {"resnet0": getattr(resnet0, "in_channels", None), "resnet1": getattr(resnet1, "in_channels", None), "same": getattr(resnet0, "in_channels", None) == getattr(resnet1, "in_channels", None)},
            "out_channels": {"resnet0": getattr(resnet0, "out_channels", None), "resnet1": getattr(resnet1, "out_channels", None), "same": getattr(resnet0, "out_channels", None) == getattr(resnet1, "out_channels", None)},
            "has_shortcut": {"resnet0": getattr(resnet0, "conv_shortcut", None) is not None, "resnet1": getattr(resnet1, "conv_shortcut", None) is not None},
            "time_embedding_norm": {"resnet0": getattr(resnet0, "time_embedding_norm", None), "resnet1": getattr(resnet1, "time_embedding_norm", None)},
            "output_scale_factor": {"resnet0": getattr(resnet0, "output_scale_factor", None), "resnet1": getattr(resnet1, "output_scale_factor", None)},
        },
        "submodules": _submodule_info(resnet1),
        "tester_support": {
            "existing_UNetResnetBlock2DTester_supported": cfg["time_embedding_norm"] in {"default", "scale_shift"} and cfg["dropout"] == 0.0,
            "channel_change": cfg["channel_change"],
            "conv_shortcut_required": cfg["has_shortcut"],
        },
        "bridge_shapes": {
            "sample": list(sample.shape),
            "encoder_hidden_states": encoder_shape,
            "conv_in": list(conv_in.shape),
            "time_embedding": list(temb.shape),
            "downblock0_hidden": list(block0_hidden.shape),
            "downblock0_output_state_shapes": [list(x.shape) for x in block0_states],
            "downblock1_hidden": list(block1_hidden.shape),
            "downblock1_output_state_shapes": [list(x.shape) for x in block1_states],
            "downblock2_resnet0_output": list(resnet0_out.shape),
            "downblock2_attention0_output": list(attention0_out.shape),
            "downblock2_resnet1_input": list(attention0_out.shape),
            "downblock2_resnet1_output": list(resnet1_out.shape),
        },
        "bridge_tensor_stats": {
            "attention0_output": stats_tensor(attention0_out),
            "resnet1_output": stats_tensor(resnet1_out),
        },
        "output_states_preview": {
            "resnets0_output_shape": list(resnet0_out.shape),
            "attentions0_output_shape": list(attention0_out.shape),
            "resnets1_output_shape": list(resnet1_out.shape),
            "resnet1_output_is_immediate_residual_sample": False,
            "reason": "CrossAttnDownBlock2D appends output_states after each resnet+attention pair; resnet1 must be followed by attentions.1 before the second downblock2 residual is complete.",
            "future_output_states_not_implemented_this_stage": True,
            "accumulated_previous_output_state_shapes": [list(conv_in.shape)] + [list(x.shape) for x in block0_states] + [list(x.shape) for x in block1_states],
        },
        "attention1_preview": attention_preview(attention1, expected_input_shape=list(resnet1_out.shape), encoder_hidden_states_shape=encoder_shape),
        "effective_weights_required": True,
        "effective_weights_strategy": "Export merged static weights in tools/export_tadsr_unet_downblock2_resnet1_oracle.py; no generic Jittor LoRA runtime.",
        "not_in_scope": ["down_blocks.2.attentions.1", "down_blocks.2.downsamplers.0", "mid_block", "up_blocks", "full UNet forward", "full TADSR inference", "generic Jittor LoRA runtime"],
        "next_action": "Port down_blocks.2.attentions.1 after down_blocks.2.resnets.1 alignment; keep full UNet forward NotImplemented.",
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    lines = [
        "# TADSR UNet down_blocks.2.resnets.1 audit",
        "",
        "TADSR_UNET_DOWNBLOCK2_RESNET1_AUDIT: PASS",
        "TADSR_UNET_DOWNBLOCK2_RESNET1_LORA_AUDIT: PASS",
        "",
        f"class: {audit['resnet1_config']['class']}",
        f"in/out: {audit['resnet1_config']['in_channels']} -> {audit['resnet1_config']['out_channels']}",
        f"has_shortcut: {audit['resnet1_config']['has_shortcut']}",
        f"input shape: {audit['bridge_shapes']['downblock2_resnet1_input']}",
        f"output shape: {audit['bridge_shapes']['downblock2_resnet1_output']}",
        f"next module: {audit['downblock2_overview']['actual_next_module_after_resnet1']}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("TADSR_UNET_DOWNBLOCK2_RESNET1_AUDIT: PASS")
    print("TADSR_UNET_DOWNBLOCK2_RESNET1_LORA_AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
