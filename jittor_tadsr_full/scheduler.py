from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class MinimalDDPMScheduler:
    """Scheduler math used for alignment tests; this is not a full diffusers scheduler port."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012, beta_schedule: str = "scaled_linear"):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        if beta_schedule == "scaled_linear":
            self.betas = (np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps, dtype=np.float64) ** 2).astype("float32")
        else:
            self.betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype="float32")
        self.alphas = (1.0 - self.betas).astype("float32")
        self.alphas_cumprod = np.cumprod(self.alphas).astype("float32")
        self.timesteps = np.arange(self.num_train_timesteps - 1, -1, -1, dtype="int64")

    @classmethod
    def from_config(cls, path: str | Path):
        cfg = json.loads(Path(path).read_text())
        return cls(num_train_timesteps=cfg.get("num_train_timesteps", 1000), beta_start=cfg.get("beta_start", 0.00085), beta_end=cfg.get("beta_end", 0.012), beta_schedule=cfg.get("beta_schedule", "scaled_linear"))

    def alpha_cumprod(self, timestep: int) -> float:
        return float(self.alphas_cumprod[int(timestep)])

    def add_noise(self, sample: np.ndarray, noise: np.ndarray, timestep: int) -> np.ndarray:
        a = self.alpha_cumprod(int(timestep))
        return (a ** 0.5) * sample + ((1.0 - a) ** 0.5) * noise

    def one_step_reconstruct(self, noisy: np.ndarray, pred_noise: np.ndarray, timestep: int) -> np.ndarray:
        a = self.alpha_cumprod(int(timestep))
        return (noisy - ((1.0 - a) ** 0.5) * pred_noise) / max(a ** 0.5, 1e-12)

    def state_dict(self):
        return {"betas": self.betas, "alphas": self.alphas, "alphas_cumprod": self.alphas_cumprod, "timesteps": self.timesteps}


class TADSRSchedulerBoundaryTester(MinimalDDPMScheduler):
    """Alignment-only mirror of the official TADSR DDPMScheduler boundary.

    The official inference code instantiates ``DDPMScheduler`` and uses a fixed
    one-step setup, but it does not run a full denoising loop. This tester only
    mirrors the audited one-step boundary for oracle comparison; it is not a
    production scheduler API.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas=None,
        variance_type: str = "fixed_small",
        clip_sample: bool = False,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 1,
        rescale_betas_zero_snr: bool = False,
        **_unused,
    ):
        if trained_betas is not None:
            self.betas = np.asarray(trained_betas, dtype="float32")
            self.num_train_timesteps = int(self.betas.shape[0])
            self.beta_start = float(beta_start)
            self.beta_end = float(beta_end)
            self.beta_schedule = beta_schedule
        else:
            super().__init__(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
            )
        self.variance_type = variance_type
        self.clip_sample = bool(clip_sample)
        self.prediction_type = prediction_type
        self.thresholding = bool(thresholding)
        self.dynamic_thresholding_ratio = float(dynamic_thresholding_ratio)
        self.clip_sample_range = float(clip_sample_range)
        self.sample_max_value = float(sample_max_value)
        self.timestep_spacing = timestep_spacing
        self.steps_offset = int(steps_offset)
        self.rescale_betas_zero_snr = bool(rescale_betas_zero_snr)
        self.one = np.array(1.0, dtype="float32")
        self.init_noise_sigma = 1.0
        self.custom_timesteps = False
        self.num_inference_steps = None

    @classmethod
    def from_config(cls, path: str | Path):
        cfg = json.loads(Path(path).read_text())
        return cls(**cfg)

    def set_timesteps_for_alignment(self, num_inference_steps: int):
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")
        self.num_inference_steps = num_inference_steps
        self.custom_timesteps = False
        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype("int64")
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype("int64")
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype("int64")
            timesteps -= 1
        else:
            raise ValueError(f"unsupported timestep_spacing: {self.timestep_spacing}")
        self.timesteps = timesteps.astype("int64")
        return self.timesteps

    def scale_model_input_for_alignment(self, sample: np.ndarray, timestep: int | np.ndarray | None = None) -> np.ndarray:
        return np.asarray(sample, dtype="float32")

    def previous_timestep_for_alignment(self, timestep: int) -> int:
        if self.custom_timesteps:
            idx = np.nonzero(self.timesteps == int(timestep))[0]
            if idx.size == 0:
                raise ValueError(f"timestep {timestep} not found in custom timesteps")
            return -1 if int(idx[0]) == len(self.timesteps) - 1 else int(self.timesteps[int(idx[0]) + 1])
        num_inference_steps = self.num_inference_steps or self.num_train_timesteps
        return int(timestep) - self.num_train_timesteps // int(num_inference_steps)

    def _get_variance_for_alignment(self, t: int, predicted_variance=None) -> np.ndarray:
        prev_t = self.previous_timestep_for_alignment(int(t))
        alpha_prod_t = self.alphas_cumprod[int(t)]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev
        variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
        variance = np.maximum(variance, np.array(1e-20, dtype="float32"))
        if self.variance_type == "fixed_small":
            return variance
        if self.variance_type == "fixed_small_log":
            return np.exp(0.5 * np.log(variance))
        if self.variance_type == "fixed_large":
            return current_beta_t
        if self.variance_type == "fixed_large_log":
            return np.log(current_beta_t)
        if self.variance_type == "learned":
            return predicted_variance
        if self.variance_type == "learned_range":
            min_log = np.log(variance)
            max_log = np.log(current_beta_t)
            frac = (predicted_variance + 1.0) / 2.0
            return frac * max_log + (1.0 - frac) * min_log
        return variance

    def step_for_alignment(
        self,
        model_output: np.ndarray,
        timestep: int | np.ndarray,
        sample: np.ndarray,
        variance_noise: np.ndarray | None = None,
    ) -> dict:
        model_output = np.asarray(model_output, dtype="float32")
        sample = np.asarray(sample, dtype="float32")
        t = int(np.asarray(timestep).reshape(-1)[0])
        prev_t = self.previous_timestep_for_alignment(t)

        predicted_variance = None
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in {"learned", "learned_range"}:
            model_output, predicted_variance = np.split(model_output, 2, axis=1)

        alpha_prod_t = self.alphas_cumprod[t].astype("float32")
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].astype("float32") if prev_t >= 0 else self.one
        beta_prod_t = np.array(1.0, dtype="float32") - alpha_prod_t
        beta_prod_t_prev = np.array(1.0, dtype="float32") - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = np.array(1.0, dtype="float32") - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
        else:
            raise ValueError(f"unsupported prediction_type: {self.prediction_type}")

        if self.thresholding:
            raise NotImplementedError("thresholding is not used by the audited TADSR scheduler config")
        if self.clip_sample:
            pred_original_sample = np.clip(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = (np.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        if t > 0:
            if variance_noise is None:
                variance_noise = np.zeros_like(model_output, dtype="float32")
            variance = self._get_variance_for_alignment(t, predicted_variance=predicted_variance)
            if self.variance_type == "fixed_small_log":
                pred_prev_sample = pred_prev_sample + variance * variance_noise
            elif self.variance_type == "learned_range":
                pred_prev_sample = pred_prev_sample + np.exp(0.5 * variance) * variance_noise
            else:
                pred_prev_sample = pred_prev_sample + np.sqrt(variance) * variance_noise

        return {
            "prev_sample": pred_prev_sample.astype("float32"),
            "pred_original_sample": pred_original_sample.astype("float32"),
        }
