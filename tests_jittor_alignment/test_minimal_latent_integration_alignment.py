#!/usr/bin/env python3
from __future__ import annotations

from minimal_integration_common import (
    add_metrics,
    load_oracle,
    minimal_tester,
    status_from_diagnostics,
    write_report,
)


def main() -> int:
    oracle, blocked = load_oracle()
    if blocked:
        result = {
            "status": blocked["status"],
            "reason": blocked.get("reason"),
            "diagnostics": {},
            "policy": {"full_tadsr_inference_executed": False},
        }
        write_report("jittor_minimal_latent_integration_alignment", "Minimal Latent Integration Alignment", result)
        for marker in [
            "TADSR_MINIMAL_VAE_ENCODE_ALIGNMENT",
            "TADSR_MINIMAL_UNET_MODEL_PRED_ALIGNMENT",
            "TADSR_GET_X0_FROM_RES_ALIGNMENT",
            "TADSR_MINIMAL_LATENT_ONE_STEP_ALIGNMENT",
            "TADSR_MINIMAL_DECODE_INPUT_ALIGNMENT",
            "TADSR_MINIMAL_DECODED_OUTPUT_ALIGNMENT",
            "TADSR_MINIMAL_FINAL_CLAMPED_OUTPUT_ALIGNMENT",
            "TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT",
            "TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN",
            "TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN",
        ]:
            print(f"{marker}: {blocked['status']}")
        return 1

    arrays = oracle["arrays"]
    meta = oracle["metadata"]
    tester = minimal_tester(float(meta.get("scaling_factor", 0.18215)))
    got = tester.run_minimal_latent_one_step_for_alignment(
        arrays["minimal_input"].astype("float32"),
        arrays["minimal_timestep"].astype("int64"),
        arrays["minimal_encoder_hidden_states"].astype("float32"),
        arrays["minimal_sample_epsilon"].astype("float32"),
        return_intermediates=True,
        decode_boundary=True,
    )

    tol = 2e-3
    diagnostics = {
        "posterior_sample": add_metrics(got["posterior_sample"], arrays["minimal_posterior_sample"], tolerance=tol),
        "scaled_latent": add_metrics(got["scaled_latent"], arrays["minimal_scaled_latent"], tolerance=tol),
        "unet_model_pred": add_metrics(got["unet_model_pred"], arrays["minimal_unet_model_pred"], tolerance=tol),
        "alpha_prod_t": add_metrics(got["alpha_prod_t"], arrays["minimal_alpha_prod_t"], tolerance=1e-8),
        "sqrt_alpha_prod_t": add_metrics(got["sqrt_alpha_prod_t"], arrays["minimal_sqrt_alpha_prod_t"], tolerance=1e-8),
        "x0_from_res": add_metrics(got["x0_from_res"], arrays["minimal_x0_from_res"], tolerance=tol),
        "decode_input": add_metrics(got["decode_input"], arrays["minimal_decode_input"], tolerance=tol),
        "decoded_output": add_metrics(got["decoded_output"], arrays["minimal_decoded_output"], tolerance=tol),
        "final_clamped_output": add_metrics(got["final_clamped_output"], arrays["minimal_final_clamped_output"], tolerance=tol),
    }

    vae_status = "PASS" if diagnostics["posterior_sample"]["status"] == "PASS" and diagnostics["scaled_latent"]["status"] == "PASS" else "FAIL"
    unet_status = diagnostics["unet_model_pred"]["status"]
    x0_status = "PASS" if diagnostics["alpha_prod_t"]["status"] == diagnostics["sqrt_alpha_prod_t"]["status"] == diagnostics["x0_from_res"]["status"] == diagnostics["decode_input"]["status"] == "PASS" else "FAIL"
    one_step_diagnostics = {k: diagnostics[k] for k in ["posterior_sample", "scaled_latent", "unet_model_pred", "alpha_prod_t", "sqrt_alpha_prod_t", "x0_from_res", "decode_input"]}
    one_step_status = status_from_diagnostics(one_step_diagnostics)
    decode_input_status = diagnostics["decode_input"]["status"]
    decoded_status = diagnostics["decoded_output"]["status"]
    clamped_status = diagnostics["final_clamped_output"]["status"]
    decode_status = "PASS" if decode_input_status == decoded_status == clamped_status == "PASS" else "FAIL"
    dry_run_status = "PASS" if vae_status == "PASS" and unet_status == "PASS" and x0_status == "PASS" and one_step_status == "PASS" else "FAIL"
    one_step_decode_status = "PASS" if dry_run_status == "PASS" and decode_status == "PASS" else "FAIL"

    result = {
        "status": one_step_decode_status,
        "tadsr_minimal_vae_encode_alignment": vae_status,
        "tadsr_minimal_unet_model_pred_alignment": unet_status,
        "tadsr_get_x0_from_res_alignment": x0_status,
        "tadsr_minimal_latent_one_step_alignment": one_step_status,
        "tadsr_minimal_decode_input_alignment": decode_input_status,
        "tadsr_minimal_decoded_output_alignment": decoded_status,
        "tadsr_minimal_final_clamped_output_alignment": clamped_status,
        "tadsr_minimal_decode_boundary_alignment": decode_status,
        "tadsr_minimal_latent_integration_dry_run": dry_run_status,
        "tadsr_minimal_one_step_decode_dry_run": one_step_decode_status,
        "diagnostics": diagnostics,
        "oracle_metadata": meta,
        "policy": {
            "latent_only": True,
            "decode_boundary": "PASS_TENSOR_ONLY_DECODE_CLAMP_BOUNDARY",
            "scheduler_step_executed": False,
            "full_denoising_loop_executed": False,
            "vae_decode_executed": True,
            "image_saved": False,
            "full_tadsr_inference_executed": False,
            "production_cli_used": False,
            "runtime_lora_dynamic_loading": False,
        },
    }
    write_report("jittor_minimal_latent_integration_alignment", "Minimal Latent Integration Alignment", result)

    print(f"TADSR_MINIMAL_VAE_ENCODE_ALIGNMENT: {vae_status}")
    print(f"TADSR_MINIMAL_UNET_MODEL_PRED_ALIGNMENT: {unet_status}")
    print(f"TADSR_GET_X0_FROM_RES_ALIGNMENT: {x0_status}")
    print(f"TADSR_MINIMAL_LATENT_ONE_STEP_ALIGNMENT: {one_step_status}")
    print(f"TADSR_MINIMAL_DECODE_INPUT_ALIGNMENT: {decode_input_status}")
    print(f"TADSR_MINIMAL_DECODED_OUTPUT_ALIGNMENT: {decoded_status}")
    print(f"TADSR_MINIMAL_FINAL_CLAMPED_OUTPUT_ALIGNMENT: {clamped_status}")
    print(f"TADSR_MINIMAL_DECODE_BOUNDARY_ALIGNMENT: {decode_status}")
    print(f"TADSR_MINIMAL_LATENT_INTEGRATION_DRY_RUN: {dry_run_status}")
    print(f"TADSR_MINIMAL_ONE_STEP_DECODE_DRY_RUN: {one_step_decode_status}")
    return 0 if one_step_decode_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
