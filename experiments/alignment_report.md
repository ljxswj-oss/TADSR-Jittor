# Jittor vs PyTorch Tiny Alignment Report

| framework   |    psnr |      ssim |   latency_ms_per_image |   params |   checkpoint_size_mb |
|:------------|--------:|----------:|-----------------------:|---------:|---------------------:|
| Jittor      | 9.39296 | 0.062834  |                39.3715 |   256003 |             0.979259 |
| PyTorch     | 9.37534 | 0.0575725 |                20.5328 |   246755 |             0.955689 |

结论：这里比较的是 tiny faithful reproduction 的训练趋势、输出形态和指标量级，不声称达到官方 full Stable Diffusion TADSR 的 SOTA 性能。