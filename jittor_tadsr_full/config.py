from dataclasses import dataclass

@dataclass
class FullTADSRConfig:
    upscale: int = 4
    process_size: int = 512
    align_method: str = "wavelet"
    timestep: int = 999
