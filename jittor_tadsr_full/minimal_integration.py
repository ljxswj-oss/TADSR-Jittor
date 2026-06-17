from __future__ import annotations

import numpy as np


class TADSRMinimalLatentIntegrationTester:
    """Alignment-only minimal latent integration tester.

    This helper composes already audited boundary testers:

    - TimeVAE actual VAEHook encoder boundary;
    - UNet full-forward alignment wrapper;
    - scheduler alpha contract for the manual TADSR get_x0_from_res formula.

    It deliberately does not implement a production TADSR pipeline, a full
    denoising loop, image saving, generic runtime LoRA, or the full inference
    CLI.
    """

    def __init__(self, timevae_tester, unet_tester, scheduler_tester, scaling_factor: float = 0.18215):
        self.timevae_tester = timevae_tester
        self.unet_tester = unet_tester
        self.scheduler_tester = scheduler_tester
        self.scaling_factor = np.float32(scaling_factor)

    def alpha_prod_t_for_alignment(self, timestep) -> np.ndarray:
        t = int(np.asarray(timestep).reshape(-1)[0])
        return np.asarray([self.scheduler_tester.alphas_cumprod[t]], dtype=np.float32)

    def get_x0_from_res_for_alignment(self, latent_lq: np.ndarray, model_pred: np.ndarray, timestep) -> dict:
        latent = np.asarray(latent_lq, dtype=np.float32)
        pred = np.asarray(model_pred, dtype=np.float32)
        alpha_prod_t = self.alpha_prod_t_for_alignment(timestep)
        sqrt_alpha = np.sqrt(alpha_prod_t).astype(np.float32)
        x0 = (latent / sqrt_alpha.reshape(-1, 1, 1, 1) - pred).astype(np.float32)
        return {
            "alpha_prod_t": alpha_prod_t,
            "sqrt_alpha_prod_t": sqrt_alpha,
            "x0": x0,
        }

    def run_minimal_latent_one_step_for_alignment(
        self,
        input_tensor: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        epsilon: np.ndarray,
        return_intermediates: bool = False,
        decode_boundary: bool = False,
    ):
        enc = self.timevae_tester.run_actual_encoder_hook_boundary_for_alignment(
            np.asarray(input_tensor, dtype=np.float32),
            np.asarray(timestep).astype(np.int64),
            return_intermediates=True,
        )
        moments = enc["moments"]
        posterior_sample = self.timevae_tester.run_posterior_sample_with_epsilon(
            moments,
            np.asarray(epsilon, dtype=np.float32),
        )
        scaled_latent = self.timevae_tester.scale_latent(posterior_sample, float(self.scaling_factor))
        model_pred = self.unet_tester.run_full_forward_for_alignment(
            scaled_latent,
            np.asarray(timestep).astype(np.int64),
            np.asarray(encoder_hidden_states, dtype=np.float32),
            return_dict=False,
            return_intermediates=False,
        ).astype(np.float32)
        x0_pack = self.get_x0_from_res_for_alignment(scaled_latent, model_pred, timestep)
        decode_input = (x0_pack["x0"] / self.scaling_factor).astype(np.float32)
        result = {
            "posterior_sample": posterior_sample.astype(np.float32),
            "scaled_latent": scaled_latent.astype(np.float32),
            "unet_model_pred": model_pred,
            "alpha_prod_t": x0_pack["alpha_prod_t"],
            "sqrt_alpha_prod_t": x0_pack["sqrt_alpha_prod_t"],
            "x0_from_res": x0_pack["x0"],
            "decode_input": decode_input,
            "full_denoising_loop_executed": False,
            "full_tadsr_inference_executed": False,
            "image_saved": False,
            "runtime_lora_dynamic_loading": False,
        }
        if decode_boundary:
            dec = self.timevae_tester.run_actual_decoder_hook_boundary_for_alignment(decode_input, return_intermediates=True)
            result["decoded_output"] = dec["decoded_output"].astype(np.float32)
            result["final_clamped_output"] = dec["final_clamped_output"].astype(np.float32)
        if return_intermediates:
            result["encoder_moments"] = moments.astype(np.float32)
            result["encoder_policy"] = enc.get("encoder_policy", {})
            result["tile_metadata"] = enc.get("tile_metadata", {})
            return result
        return result["x0_from_res"]
