from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_TIME_VAE_NPZ = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/time_vae_weights.npz')


def time_vae_key_groups(npz_path: str | Path):
    path = Path(npz_path)
    if not path.exists():
        return {'status': 'MISSING', 'path': str(path), 'groups': {}}
    data = np.load(path)
    groups = {'encoder': [], 'decoder': [], 'mid_block': [], 'down_blocks': [], 'up_blocks': [], 'time_embedding': [], 'other': []}
    for key in data.files:
        low = key.lower()
        group = 'other'
        for candidate in ['time_embedding', 'down_blocks', 'up_blocks', 'mid_block', 'encoder', 'decoder']:
            if candidate in low:
                group = candidate
                break
        groups[group].append({'key': key, 'shape': list(data[key].shape), 'dtype': str(data[key].dtype)})
    return {'status': 'PARTIAL', 'path': str(path), 'groups': {k: v[:100] for k, v in groups.items()}, 'counts': {k: len(v) for k, v in groups.items()}, 'note': 'Weight loading plus leaf/block helpers; full TimeAware VAE forward is not implemented.'}


def npz_prefix(prefix: str) -> str:
    return prefix.replace('.', '__')


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (1.0 + np.exp(-x))


def conv2d_nchw(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, stride: int = 1, padding: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError(f'conv2d_nchw expects x and weight to be 4D, got {x.shape} and {weight.shape}')
    _, c, _, _ = x.shape
    _, in_c, kh, kw = weight.shape
    if c != in_c:
        raise ValueError(f'input channels mismatch: x has {c}, weight expects {in_c}; weight shape={weight.shape}')
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    windows = sliding_window_view(x, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    # Float64 accumulation keeps the pure NumPy bridge closer to PyTorch's
    # convolution results after many decoder blocks; outputs remain float32.
    out = np.einsum(
        'nchwkl,ockl->nohw',
        windows.astype(np.float64),
        weight.astype(np.float64),
        optimize=True,
    ).astype(np.float32)
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32)[None, :, None, None]
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
    if bias is not None:
        y += np.asarray(bias, dtype=np.float32)
    return y.astype(np.float32)


def timestep_embedding_np(
    timesteps: np.ndarray,
    embedding_dim: int = 128,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    max_period: int = 10000,
) -> np.ndarray:
    """NumPy equivalent of diffusers' Timesteps/get_timestep_embedding.

    The official TimeAwareEncoder uses Timesteps(128, flip_sin_to_cos=True,
    downscale_freq_shift=0) followed by the two-layer TimestepEmbedding MLP.
    Keeping this helper here lets the actual-VAEHook alignment wrapper consume
    the exported raw timestep instead of requiring a pre-exported encoder_temb.
    """

    t = np.asarray(timesteps).reshape(-1).astype(np.float64)
    half_dim = embedding_dim // 2
    if half_dim <= 0:
        raise ValueError(f'embedding_dim must be positive, got {embedding_dim}')
    denom = half_dim - downscale_freq_shift
    if denom <= 0:
        raise ValueError(f'invalid downscale_freq_shift={downscale_freq_shift} for embedding_dim={embedding_dim}')
    exponent = -np.log(float(max_period)) * np.arange(half_dim, dtype=np.float64) / denom
    emb = t[:, None] * np.exp(exponent)[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2:
        emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    return emb.astype(np.float32)


def group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, num_groups: int = 32, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n, c, h, w = x.shape
    if c % num_groups != 0:
        raise ValueError(f'channels {c} not divisible by num_groups {num_groups}')
    y = x.reshape(n, num_groups, c // num_groups, h, w)
    mean = y.mean(axis=(2, 3, 4), keepdims=True)
    var = y.var(axis=(2, 3, 4), keepdims=True)
    y = (y - mean) / np.sqrt(var + eps)
    y = y.reshape(n, c, h, w)
    y = y * np.asarray(weight, dtype=np.float32)[None, :, None, None] + np.asarray(bias, dtype=np.float32)[None, :, None, None]
    return y.astype(np.float32)


def group_norm_sequence(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, num_groups: int = 32, eps: float = 1e-6) -> np.ndarray:
    """GroupNorm for tensors shaped [batch, channels, sequence]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f'group_norm_sequence expects 3D [N,C,S], got {x.shape}')
    n, c, s = x.shape
    if c % num_groups != 0:
        raise ValueError(f'channels {c} not divisible by num_groups {num_groups}')
    y = x.reshape(n, num_groups, c // num_groups, s)
    mean = y.mean(axis=(2, 3), keepdims=True)
    var = y.var(axis=(2, 3), keepdims=True)
    y = (y - mean) / np.sqrt(var + eps)
    y = y.reshape(n, c, s)
    y = y * np.asarray(weight, dtype=np.float32)[None, :, None] + np.asarray(bias, dtype=np.float32)[None, :, None]
    return y.astype(np.float32)


def official_split_tiles(h: int, w: int, tile_size: int = 16, pad: int = 32, is_decoder: bool = False) -> tuple[list[list[int]], list[list[int]], dict]:
    """NumPy mirror of official ``VAEHook.split_tiles``.

    Bounding boxes use the official order ``[x1, x2, y1, y2]``. Encoder output
    boxes are divided by 8, decoder boxes would be multiplied by 8. This helper
    is used only by alignment checks for the official actual encoder hook.
    """

    def get_best_tile_size(lowerbound: int, upperbound: int) -> int:
        divider = 32
        while divider >= 2:
            remainder = lowerbound % divider
            if remainder == 0:
                return lowerbound
            candidate = lowerbound - remainder + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    h = int(h)
    w = int(w)
    tile_size = int(tile_size)
    pad = int(pad)
    num_height_tiles = max(int(np.ceil((h - 2 * pad) / tile_size)), 1)
    num_width_tiles = max(int(np.ceil((w - 2 * pad) / tile_size)), 1)
    real_tile_height = int(np.ceil((h - 2 * pad) / num_height_tiles))
    real_tile_width = int(np.ceil((w - 2 * pad) / num_width_tiles))
    real_tile_height = get_best_tile_size(real_tile_height, tile_size)
    real_tile_width = get_best_tile_size(real_tile_width, tile_size)

    tile_input_bboxes: list[list[int]] = []
    tile_output_bboxes: list[list[int]] = []
    for i in range(num_height_tiles):
        for j in range(num_width_tiles):
            input_bbox = [
                pad + j * real_tile_width,
                min(pad + (j + 1) * real_tile_width, w),
                pad + i * real_tile_height,
                min(pad + (i + 1) * real_tile_height, h),
            ]
            output_bbox = [
                input_bbox[0] if input_bbox[0] > pad else 0,
                input_bbox[1] if input_bbox[1] < w - pad else w,
                input_bbox[2] if input_bbox[2] > pad else 0,
                input_bbox[3] if input_bbox[3] < h - pad else h,
            ]
            output_bbox = [x * 8 if is_decoder else x // 8 for x in output_bbox]
            tile_output_bboxes.append(output_bbox)
            tile_input_bboxes.append([
                max(0, input_bbox[0] - pad),
                min(w, input_bbox[1] + pad),
                max(0, input_bbox[2] - pad),
                min(h, input_bbox[3] + pad),
            ])

    metadata = {
        'tile_size': tile_size,
        'pad': pad,
        'is_decoder': bool(is_decoder),
        'num_height_tiles': num_height_tiles,
        'num_width_tiles': num_width_tiles,
        'real_tile_height': real_tile_height,
        'real_tile_width': real_tile_width,
        'tile_count': len(tile_input_bboxes),
    }
    return tile_input_bboxes, tile_output_bboxes, metadata


def official_crop_valid_region(x: np.ndarray, input_bbox: list[int], target_bbox: list[int], is_decoder: bool = False) -> np.ndarray:
    """NumPy mirror of official ``crop_valid_region``."""

    x = np.asarray(x, dtype=np.float32)
    padded_bbox = [i * 8 if is_decoder else i // 8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.shape[2] + margin[3], margin[0]:x.shape[3] + margin[1]].astype(np.float32)


def official_get_var_mean(x: np.ndarray, num_groups: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mirror official ``get_var_mean`` used by ``GroupNormParam``."""

    x = np.asarray(x, dtype=np.float32)
    b, c = x.shape[0], x.shape[1]
    if c % num_groups != 0:
        raise ValueError(f'channels {c} not divisible by num_groups {num_groups}')
    channel_in_group = c // num_groups
    reshaped = x.reshape(1, b * num_groups, channel_in_group, *x.shape[2:])
    var = reshaped.var(axis=(0, 2, 3, 4)).astype(np.float32)
    mean = reshaped.mean(axis=(0, 2, 3, 4)).astype(np.float32)
    return var, mean


def official_custom_group_norm(
    x: np.ndarray,
    num_groups: int,
    mean: np.ndarray,
    var: np.ndarray,
    weight: np.ndarray | None,
    bias: np.ndarray | None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Mirror official fixed-stat ``custom_group_norm``."""

    x = np.asarray(x, dtype=np.float32)
    b, c = x.shape[0], x.shape[1]
    channel_in_group = c // num_groups
    reshaped = x.reshape(1, b * num_groups, channel_in_group, *x.shape[2:])
    y = (reshaped - mean[None, :, None, None, None]) / np.sqrt(var[None, :, None, None, None] + np.float32(eps))
    y = y.reshape(b, c, *x.shape[2:])
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32)[None, :, None, None]
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)[None, :, None, None]
    return y.astype(np.float32)


class OfficialGroupNormParam:
    """Small NumPy clone of official VAEHook ``GroupNormParam``."""

    def __init__(self):
        self.var_list: list[np.ndarray] = []
        self.mean_list: list[np.ndarray] = []
        self.pixel_list: list[int] = []
        self.weight: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.labels: list[str] = []

    def add_tile(self, tile: np.ndarray, norm_spec: dict) -> None:
        var, mean = official_get_var_mean(tile, 32)
        self.var_list.append(var.astype(np.float32))
        self.mean_list.append(mean.astype(np.float32))
        self.pixel_list.append(int(tile.shape[2] * tile.shape[3]))
        self.weight = None if norm_spec.get('weight') is None else np.asarray(norm_spec['weight'], dtype=np.float32)
        self.bias = None if norm_spec.get('bias') is None else np.asarray(norm_spec['bias'], dtype=np.float32)
        self.labels.append(str(norm_spec.get('label', 'pre_norm')))

    def summary(self):
        if not self.var_list:
            return None, {}
        var = np.stack(self.var_list, axis=0).astype(np.float32)
        mean = np.stack(self.mean_list, axis=0).astype(np.float32)
        pixels = np.asarray(self.pixel_list, dtype=np.float32)
        pixels = (pixels / pixels.max())[:, None]
        pixels = pixels / pixels.sum()
        merged_var = (var * pixels).sum(axis=0).astype(np.float32)
        merged_mean = (mean * pixels).sum(axis=0).astype(np.float32)
        label = self.labels[0] if self.labels else 'pre_norm'

        def apply(x, mean=merged_mean, var=merged_var, weight=self.weight, bias=self.bias):
            return official_custom_group_norm(x, 32, mean, var, weight, bias, eps=1e-6)

        stats = {
            'label': label,
            'num_tiles': len(self.var_list),
            'pixel_list': [int(x) for x in self.pixel_list],
            'mean_shape': list(merged_mean.shape),
            'var_shape': list(merged_var.shape),
            'mean_values': merged_mean.astype(float).tolist(),
            'var_values': merged_var.astype(float).tolist(),
        }
        return apply, stats


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    x64 = x64 - x64.max(axis=axis, keepdims=True)
    e = np.exp(x64)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)



class TimeAwareResnetBlock:
    def __init__(self, weights, prefix: str, groups: int = 32, eps: float = 1e-6, time_scale_shift: str = 'scale_shift'):
        self.weights = weights
        self.prefix = npz_prefix(prefix)
        self.groups = groups
        self.eps = eps
        self.time_scale_shift = time_scale_shift
        self.output_scale_factor = 1.0
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}__{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware ResnetBlock key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'norm1__weight', 'norm1__bias', 'conv1__weight', 'conv1__bias',
            'norm2__weight', 'norm2__bias', 'conv2__weight', 'conv2__bias',
        ]
        missing = [self.key(s) for s in required if not self.has(s)]
        has_time_w = self.has('time_emb_proj__weight')
        has_time_b = self.has('time_emb_proj__bias')
        if has_time_w != has_time_b:
            missing.append(self.key('time_emb_proj__weight' if not has_time_w else 'time_emb_proj__bias'))
        if missing:
            nearby = [k for k in self.weights.files if self.prefix in k or 'mid_block' in k or 'resnets' in k][:40]
            raise KeyError(f'Missing keys for {self.prefix}: {missing}; nearby keys={nearby}')

    def __call__(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        hidden = group_norm(x, self.get('norm1__weight'), self.get('norm1__bias'), self.groups, self.eps)
        hidden = silu(hidden)
        hidden = conv2d_nchw(hidden, self.get('conv1__weight'), self.get('conv1__bias'), stride=1, padding=1)

        hidden = group_norm(hidden, self.get('norm2__weight'), self.get('norm2__bias'), self.groups, self.eps)
        if self.has('time_emb_proj__weight'):
            if temb is None:
                raise ValueError(f'{self.prefix} requires temb because time_emb_proj weights exist')
            temb_proj = linear(silu(temb), self.get('time_emb_proj__weight'), self.get('time_emb_proj__bias'))[:, :, None, None]
            if self.time_scale_shift == 'scale_shift':
                scale, shift = np.split(temb_proj, 2, axis=1)
                hidden = hidden * (1.0 + scale) + shift
            elif self.time_scale_shift == 'default':
                hidden = hidden + temb_proj
            else:
                raise NotImplementedError(f'Unsupported time_scale_shift: {self.time_scale_shift}')
        hidden = silu(hidden)
        hidden = conv2d_nchw(hidden, self.get('conv2__weight'), self.get('conv2__bias'), stride=1, padding=1)

        if self.has('conv_shortcut__weight'):
            residual = conv2d_nchw(x, self.get('conv_shortcut__weight'), self.get('conv_shortcut__bias'), stride=1, padding=0)
        elif self.has('nin_shortcut__weight'):
            residual = conv2d_nchw(x, self.get('nin_shortcut__weight'), self.get('nin_shortcut__bias'), stride=1, padding=0)
        else:
            residual = x
        if residual.shape != hidden.shape:
            raise ValueError(
                f'Residual shape {residual.shape} does not match hidden shape {hidden.shape} for {self.prefix}; '
                'likely missing conv_shortcut/nin_shortcut for a channel-changing ResNet block.'
            )
        return ((residual + hidden) / self.output_scale_factor).astype(np.float32)


class TimeAwareDownsample2D:
    def __init__(self, weights, prefix: str, padding: int = 0, stride: int = 2):
        self.weights = weights
        self.prefix = npz_prefix(prefix)
        self.padding = padding
        self.stride = stride
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}__{suffix}'

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware Downsample2D key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(s) for s in ['conv__weight', 'conv__bias'] if self.key(s) not in self.weights.files]
        if missing:
            raise KeyError(f'Missing keys for {self.prefix}: {missing}')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.padding == 0:
            # Matches diffusers Downsample2D: F.pad(hidden_states, (0, 1, 0, 1)) before stride-2 conv.
            x = np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), mode='constant')
            padding = 0
        else:
            padding = self.padding
        return conv2d_nchw(x, self.get('conv__weight'), self.get('conv__bias'), stride=self.stride, padding=padding)


class TimeVAEDownBlockTester:
    def __init__(self, block_index: int, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ, downsample_padding: int = 0):
        self.block_index = int(block_index)
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        base = f'encoder.down_blocks.{self.block_index}'
        self.resnets = [
            TimeAwareResnetBlock(self.weights, f'{base}.resnets.0'),
            TimeAwareResnetBlock(self.weights, f'{base}.resnets.1'),
        ]
        self.downsampler0 = None
        if f'encoder__down_blocks__{self.block_index}__downsamplers__0__conv__weight' in self.weights.files:
            self.downsampler0 = TimeAwareDownsample2D(self.weights, f'{base}.downsamplers.0', padding=downsample_padding, stride=2)

    def run_resnet(self, index: int, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.resnets[index](x, temb)

    def run_resnet0(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_resnet(0, x, temb)

    def run_resnet1(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_resnet(1, x, temb)

    def run_downsampler0(self, x: np.ndarray) -> np.ndarray:
        if self.downsampler0 is None:
            return np.asarray(x, dtype=np.float32)
        return self.downsampler0(x)

    def run_downblock(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        x = self.run_resnet0(x, temb)
        x = self.run_resnet1(x, temb)
        x = self.run_downsampler0(x)
        return x.astype(np.float32)


class TimeVAEDownBlock0Tester(TimeVAEDownBlockTester):
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        super().__init__(block_index=0, npz_path=npz_path, downsample_padding=0)

    def run_downblock0(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_downblock(x, temb)


class TimeVAEDownBlock1Tester(TimeVAEDownBlockTester):
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        super().__init__(block_index=1, npz_path=npz_path, downsample_padding=0)

    def run_downblock1(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_downblock(x, temb)


class TimeVAEDownBlock2Tester(TimeVAEDownBlockTester):
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        super().__init__(block_index=2, npz_path=npz_path, downsample_padding=0)

    def run_downblock2(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_downblock(x, temb)


class TimeVAEDownBlock3Tester(TimeVAEDownBlockTester):
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        super().__init__(block_index=3, npz_path=npz_path, downsample_padding=0)

    def run_downblock3(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_downblock(x, temb)


class TimeVAEMidBlockAttention:
    def __init__(self, weights, prefix: str = 'encoder.mid_block.attentions.0', groups: int = 32, eps: float = 1e-6, heads: int = 1, rescale_output_factor: float = 1.0, residual_connection: bool = True):
        self.weights = weights
        self.prefix = npz_prefix(prefix)
        self.groups = groups
        self.eps = eps
        self.heads = int(heads)
        self.rescale_output_factor = float(rescale_output_factor)
        self.residual_connection = bool(residual_connection)
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}__{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware MidBlock attention key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'group_norm__weight', 'group_norm__bias',
            'query__weight', 'query__bias',
            'key__weight', 'key__bias',
            'value__weight', 'value__bias',
            'proj_attn__weight', 'proj_attn__bias',
        ]
        missing = [self.key(s) for s in required if not self.has(s)]
        if missing:
            raise KeyError(f'Missing keys for {self.prefix}: {missing}')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        residual = np.asarray(x, dtype=np.float32)
        if residual.ndim != 4:
            raise ValueError(f'MidBlock attention expects NCHW input, got {residual.shape}')
        batch, channel, height, width = residual.shape
        hidden = residual.reshape(batch, channel, height * width).transpose(0, 2, 1)
        hidden = group_norm_sequence(
            hidden.transpose(0, 2, 1),
            self.get('group_norm__weight'),
            self.get('group_norm__bias'),
            self.groups,
            self.eps,
        ).transpose(0, 2, 1)

        query = linear(hidden, self.get('query__weight'), self.get('query__bias'))
        key = linear(hidden, self.get('key__weight'), self.get('key__bias'))
        value = linear(hidden, self.get('value__weight'), self.get('value__bias'))
        inner_dim = key.shape[-1]
        if inner_dim % self.heads != 0:
            raise ValueError(f'attention inner_dim {inner_dim} not divisible by heads {self.heads}')
        head_dim = inner_dim // self.heads
        query = query.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        scores = np.matmul(query.astype(np.float32), key.astype(np.float32).transpose(0, 1, 3, 2)) / np.sqrt(float(head_dim))
        probs = softmax(scores, axis=-1)
        hidden = np.matmul(probs, value.astype(np.float32))
        hidden = hidden.transpose(0, 2, 1, 3).reshape(batch, -1, self.heads * head_dim)
        hidden = linear(hidden, self.get('proj_attn__weight'), self.get('proj_attn__bias'))
        hidden = hidden.transpose(0, 2, 1).reshape(batch, channel, height, width)
        if self.residual_connection:
            hidden = hidden + residual
        return (hidden / self.rescale_output_factor).astype(np.float32)

    def run_attention_from_normalized_nchw(self, hidden_nchw: np.ndarray) -> np.ndarray:
        """Attention body after official task-queue ``pre_norm``.

        Official ``attn2task`` stores the residual, pauses at
        ``net.group_norm`` for cross-tile statistics, applies that norm, then
        calls ``attn_forward_new``. This method mirrors only the latter body.
        The residual addition remains an ``add_res`` task in the queue.
        """

        hidden_nchw = np.asarray(hidden_nchw, dtype=np.float32)
        if hidden_nchw.ndim != 4:
            raise ValueError(f'MidBlock attention expects NCHW input, got {hidden_nchw.shape}')
        batch, channel, height, width = hidden_nchw.shape
        hidden = hidden_nchw.reshape(batch, channel, height * width).transpose(0, 2, 1)
        query = linear(hidden, self.get('query__weight'), self.get('query__bias'))
        key = linear(hidden, self.get('key__weight'), self.get('key__bias'))
        value = linear(hidden, self.get('value__weight'), self.get('value__bias'))
        inner_dim = key.shape[-1]
        if inner_dim % self.heads != 0:
            raise ValueError(f'attention inner_dim {inner_dim} not divisible by heads {self.heads}')
        head_dim = inner_dim // self.heads
        query = query.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        value = value.reshape(batch, -1, self.heads, head_dim).transpose(0, 2, 1, 3)
        scores = np.matmul(query.astype(np.float32), key.astype(np.float32).transpose(0, 1, 3, 2)) / np.sqrt(float(head_dim))
        probs = softmax(scores, axis=-1)
        hidden = np.matmul(probs, value.astype(np.float32))
        hidden = hidden.transpose(0, 2, 1, 3).reshape(batch, -1, self.heads * head_dim)
        hidden = linear(hidden, self.get('proj_attn__weight'), self.get('proj_attn__bias'))
        return hidden.transpose(0, 2, 1).reshape(batch, channel, height, width).astype(np.float32)


class TimeVAEMidBlockTester:
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ, prefix: str = 'encoder.mid_block'):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self.prefix = prefix
        p = npz_prefix(prefix)
        self.resnet0 = TimeAwareResnetBlock(self.weights, f'{prefix}.resnets.0')
        self.attention0 = None
        if f'{p}__attentions__0__query__weight' in self.weights.files:
            self.attention0 = TimeVAEMidBlockAttention(self.weights, f'{prefix}.attentions.0')
        self.resnet1 = None
        if f'{p}__resnets__1__conv1__weight' in self.weights.files:
            self.resnet1 = TimeAwareResnetBlock(self.weights, f'{prefix}.resnets.1')

    def run_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.resnet0(x, temb)

    def run_attention0(self, x: np.ndarray) -> np.ndarray:
        if self.attention0 is None:
            return np.asarray(x, dtype=np.float32)
        return self.attention0(x)

    def run_attention(self, x: np.ndarray) -> np.ndarray:
        return self.run_attention0(x)

    def run_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        if self.resnet1 is None:
            return np.asarray(x, dtype=np.float32)
        return self.resnet1(x, temb)

    def run_midblock(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        resnet0 = self.run_resnet0(x, temb)
        attention = self.run_attention0(resnet0)
        resnet1 = self.run_resnet1(attention, temb)
        out = resnet1.astype(np.float32)
        if return_intermediates:
            return {
                'resnet0': resnet0.astype(np.float32),
                'attention': attention.astype(np.float32),
                'resnet1': out,
                'output': out,
            }
        return out


class TimeVAEDecoderMidBlockTester(TimeVAEMidBlockTester):
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        try:
            super().__init__(npz_path=npz_path, prefix='decoder.mid_block')
        except KeyError as exc:
            weights = np.load(npz_path)
            nearby = [k for k in weights.files if 'decoder__mid_block' in k or 'mid_block' in k or 'resnets' in k][:80]
            raise KeyError(f'Missing decoder.mid_block weights; nearby keys={nearby}') from exc


def upsample_nearest2d(x: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f'upsample_nearest2d expects NCHW 4D input, got {x.shape}')
    if int(scale_factor) != scale_factor or scale_factor <= 0:
        raise ValueError(f'Only positive integer nearest scale factors are supported, got {scale_factor}')
    factor = int(scale_factor)
    return np.repeat(np.repeat(x, factor, axis=2), factor, axis=3).astype(np.float32)


class TimeVAEUpsample2D:
    def __init__(self, weights, prefix: str, scale_factor: int = 2, padding: int = 1):
        self.weights = weights
        self.prefix = npz_prefix(prefix)
        self.scale_factor = int(scale_factor)
        self.padding = int(padding)
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}__{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:80]
            raise KeyError(f'Missing TimeVAEUpsample2D key: {key}; nearby keys={nearby}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(s) for s in ['conv__weight', 'conv__bias'] if not self.has(s)]
        if missing:
            nearby = [k for k in self.weights.files if self.prefix in k][:80]
            raise KeyError(f'Missing keys for {self.prefix}: {missing}; nearby keys={nearby}')

    def run_upsample_only(self, x: np.ndarray) -> np.ndarray:
        return upsample_nearest2d(x, self.scale_factor)

    def run_conv_after_upsample(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(x, self.get('conv__weight'), self.get('conv__bias'), stride=1, padding=self.padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.run_conv_after_upsample(self.run_upsample_only(x)).astype(np.float32)


class TimeVAEUpDecoderBlockTester:
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ, block_index: int = 0):
        if int(block_index) not in {0, 1, 2, 3}:
            raise NotImplementedError('Only decoder.up_blocks.0, decoder.up_blocks.1, decoder.up_blocks.2, and decoder.up_blocks.3 are ported in this migration stage.')
        self.block_index = int(block_index)
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self.base = f'decoder.up_blocks.{self.block_index}'
        self.npz_base = npz_prefix(self.base)
        self.resnets = []
        index = 0
        while f'{self.npz_base}__resnets__{index}__conv1__weight' in self.weights.files:
            self.resnets.append(TimeAwareResnetBlock(self.weights, f'{self.base}.resnets.{index}'))
            index += 1
        if not self.resnets:
            nearby = [k for k in self.weights.files if self.npz_base in k][:80]
            raise KeyError(f'Missing decoder.up_blocks.{self.block_index} resnet weights; nearby keys={nearby}')
        self.upsampler0 = None
        if f'{self.npz_base}__upsamplers__0__conv__weight' in self.weights.files:
            self.upsampler0 = TimeVAEUpsample2D(self.weights, f'{self.base}.upsamplers.0', scale_factor=2, padding=1)

    def run_resnet(self, index: int, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        if index < 0 or index >= len(self.resnets):
            raise IndexError(f'decoder.up_blocks.{self.block_index}.resnets.{index} is not available; count={len(self.resnets)}')
        return self.resnets[index](x, temb)

    def run_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.run_resnet(0, x, temb)

    def run_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.run_resnet(1, x, temb)

    def run_resnet2(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.run_resnet(2, x, temb)

    def run_upsampler0(self, x: np.ndarray) -> np.ndarray:
        if self.upsampler0 is None:
            return np.asarray(x, dtype=np.float32)
        return self.upsampler0(x)

    def run_upblock(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        hidden = np.asarray(x, dtype=np.float32)
        outputs = {}
        for index, resnet in enumerate(self.resnets):
            hidden = resnet(hidden, temb)
            outputs[f'resnet{index}'] = hidden.astype(np.float32)
        pre_upsampler = hidden.astype(np.float32)
        outputs['pre_upsampler'] = pre_upsampler
        if self.upsampler0 is not None:
            hidden = self.upsampler0(pre_upsampler)
            outputs['upsampler0'] = hidden.astype(np.float32)
        outputs['output'] = hidden.astype(np.float32)
        return outputs if return_intermediates else outputs['output']

class TimeVAEEncoderTailTester:
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ, groups: int = 32, eps: float = 1e-6):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self.groups = groups
        self.eps = eps
        self._validate()

    def has(self, key: str) -> bool:
        return key in self.weights.files

    def get(self, key: str) -> np.ndarray:
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware VAE encoder tail key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'encoder__conv_norm_out__weight', 'encoder__conv_norm_out__bias',
            'encoder__conv_out__weight', 'encoder__conv_out__bias',
            'quant_conv__weight', 'quant_conv__bias',
        ]
        missing = [k for k in required if not self.has(k)]
        if missing:
            raise KeyError(f'Missing encoder tail / quant_conv keys: {missing}')

    def run_conv_norm_out(self, x: np.ndarray) -> np.ndarray:
        return group_norm(
            x,
            self.get('encoder__conv_norm_out__weight'),
            self.get('encoder__conv_norm_out__bias'),
            self.groups,
            self.eps,
        )

    def run_conv_act(self, x: np.ndarray) -> np.ndarray:
        return silu(x)

    def run_conv_out(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            x,
            self.get('encoder__conv_out__weight'),
            self.get('encoder__conv_out__bias'),
            stride=1,
            padding=1,
        )

    def run_encoder_tail(self, x: np.ndarray) -> np.ndarray:
        x = self.run_conv_norm_out(x)
        x = self.run_conv_act(x)
        x = self.run_conv_out(x)
        return x.astype(np.float32)

    def run_quant_conv(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            x,
            self.get('quant_conv__weight'),
            self.get('quant_conv__bias'),
            stride=1,
            padding=0,
        )



class TimeVAEDiagonalGaussianHelper:
    """Deterministic helper for the official DiagonalGaussianDistribution entry.

    The official TimeAwareAutoencoderKL posterior splits quant_conv moments along
    channel dimension into mean/logvar, clamps logvar to [-30, 20], and `mode()`
    returns the mean. Sampling is intentionally not ported for deterministic
    alignment.
    """

    logvar_min = -30.0
    logvar_max = 20.0

    @classmethod
    def split_moments(cls, moments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        moments = np.asarray(moments, dtype=np.float32)
        if moments.ndim != 4:
            raise ValueError(f'posterior moments must be NCHW, got {moments.shape}')
        channels = moments.shape[1]
        if channels % 2 != 0:
            raise ValueError(f'posterior moments channel count must be even, got {channels}')
        mean, logvar = np.split(moments, 2, axis=1)
        return mean.astype(np.float32), logvar.astype(np.float32)

    @classmethod
    def clamp_logvar(cls, logvar: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(logvar, dtype=np.float32), cls.logvar_min, cls.logvar_max).astype(np.float32)

    @classmethod
    def mode(cls, moments: np.ndarray) -> np.ndarray:
        mean, _ = cls.split_moments(moments)
        return mean.astype(np.float32)

    @classmethod
    def std(cls, moments: np.ndarray) -> np.ndarray:
        _, raw_logvar = cls.split_moments(moments)
        logvar = cls.clamp_logvar(raw_logvar)
        return np.exp(0.5 * logvar).astype(np.float32)

    @classmethod
    def sample_with_epsilon(cls, moments: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        mean = cls.mode(moments)
        std = cls.std(moments)
        epsilon = np.asarray(epsilon, dtype=np.float32)
        if epsilon.shape != mean.shape:
            raise ValueError(f'epsilon shape {epsilon.shape} does not match posterior mean shape {mean.shape}')
        return (mean + std * epsilon).astype(np.float32)

    @classmethod
    def split_moments_with_clamp(cls, moments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, raw_logvar = cls.split_moments(moments)
        return mean, raw_logvar, cls.clamp_logvar(raw_logvar)

    @classmethod
    def sample(cls, *args, **kwargs):
        raise NotImplementedError('Deterministic Jittor alignment uses posterior mode only; sampling is not ported yet.')


class TimeVAEDecoderEntryTester:
    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self._validate()

    def has(self, key: str) -> bool:
        return key in self.weights.files

    def get(self, key: str) -> np.ndarray:
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware VAE decoder-entry key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'post_quant_conv__weight', 'post_quant_conv__bias',
            'decoder__conv_in__weight', 'decoder__conv_in__bias',
        ]
        missing = [k for k in required if not self.has(k)]
        if missing:
            raise KeyError(f'Missing decoder-entry keys: {missing}')
        post_w = self.get('post_quant_conv__weight')
        dec_w = self.get('decoder__conv_in__weight')
        if post_w.shape != (4, 4, 1, 1):
            raise ValueError(f'Unexpected post_quant_conv weight shape: {post_w.shape}')
        if dec_w.shape[1:] != (4, 3, 3):
            raise ValueError(f'Unexpected decoder.conv_in weight shape: {dec_w.shape}')

    def run_moments_split(self, moments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return TimeVAEDiagonalGaussianHelper.split_moments_with_clamp(moments)

    def run_posterior_mode(self, moments: np.ndarray) -> np.ndarray:
        return TimeVAEDiagonalGaussianHelper.mode(moments)

    def run_post_quant_conv(self, z: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            z,
            self.get('post_quant_conv__weight'),
            self.get('post_quant_conv__bias'),
            stride=1,
            padding=0,
        )

    def run_decoder_conv_in(self, z: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            z,
            self.get('decoder__conv_in__weight'),
            self.get('decoder__conv_in__bias'),
            stride=1,
            padding=1,
        )

    def run_decoder_entry(self, z: np.ndarray) -> np.ndarray:
        return self.run_decoder_conv_in(self.run_post_quant_conv(z)).astype(np.float32)

    def run_quant_to_decoder_entry(self, moments: np.ndarray) -> np.ndarray:
        return self.run_decoder_entry(self.run_posterior_mode(moments)).astype(np.float32)


class TimeVAEDecoderTailTester:
    """Deterministic decoder tail helper: conv_norm_out -> conv_act -> conv_out.

    This is a module-level alignment helper only. It intentionally does not
    apply AutoencoderKL scaling_factor, tanh/clamp, image rescaling, stochastic
    sampling, or any full inference post-processing.
    """

    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ, groups: int = 32, eps: float = 1e-6):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self.groups = int(groups)
        self.eps = float(eps)
        self._validate()

    def has(self, key: str) -> bool:
        return key in self.weights.files

    def get(self, key: str) -> np.ndarray:
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware VAE decoder tail key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'decoder__conv_norm_out__weight', 'decoder__conv_norm_out__bias',
            'decoder__conv_out__weight', 'decoder__conv_out__bias',
        ]
        missing = [k for k in required if not self.has(k)]
        if missing:
            raise KeyError(f'Missing decoder tail keys: {missing}')

    def run_norm_out(self, x: np.ndarray) -> np.ndarray:
        return group_norm(
            x,
            self.get('decoder__conv_norm_out__weight'),
            self.get('decoder__conv_norm_out__bias'),
            self.groups,
            self.eps,
        )

    def run_act(self, x: np.ndarray) -> np.ndarray:
        return silu(x)

    def run_conv_out(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            x,
            self.get('decoder__conv_out__weight'),
            self.get('decoder__conv_out__bias'),
            stride=1,
            padding=1,
        )

    def run_decoder_tail(self, x: np.ndarray, return_intermediates: bool = False):
        norm = self.run_norm_out(x)
        act = self.run_act(norm)
        conv = self.run_conv_out(act)
        if return_intermediates:
            return {
                'decoder_tail_norm_out': norm.astype(np.float32),
                'decoder_tail_act': act.astype(np.float32),
                'decoder_tail_conv_out': conv.astype(np.float32),
                'decoder_tail': conv.astype(np.float32),
                'output': conv.astype(np.float32),
            }
        return conv.astype(np.float32)


class TimeVAEBlockTester:
    """CPU-compatible helpers for block-level TimeAware VAE numerical alignment.

    This is intentionally not a full TimeAware VAE implementation. It ports only
    leaf/stage operations needed by current oracle tests and raises for full forward.
    """

    def __init__(self, npz_path: str | Path = DEFAULT_TIME_VAE_NPZ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f'TimeAware VAE NPZ not found: {self.npz_path}')
        self.weights = np.load(self.npz_path)
        self.downblock0 = TimeVAEDownBlock0Tester(npz_path)
        self.downblock1 = TimeVAEDownBlock1Tester(npz_path)
        self.downblock2 = TimeVAEDownBlock2Tester(npz_path)
        self.downblock3 = TimeVAEDownBlock3Tester(npz_path)
        self.midblock = TimeVAEMidBlockTester(npz_path)
        self.encoder_tail = TimeVAEEncoderTailTester(npz_path)
        self.decoder_entry = TimeVAEDecoderEntryTester(npz_path)
        self.decoder_midblock = TimeVAEDecoderMidBlockTester(npz_path)
        self.decoder_upblock0 = TimeVAEUpDecoderBlockTester(npz_path, block_index=0)
        self.decoder_upblock1 = TimeVAEUpDecoderBlockTester(npz_path, block_index=1)
        self.decoder_upblock2 = TimeVAEUpDecoderBlockTester(npz_path, block_index=2)
        self.decoder_upblock3 = TimeVAEUpDecoderBlockTester(npz_path, block_index=3)
        self.decoder_tail = TimeVAEDecoderTailTester(npz_path)
        self.downblocks = [self.downblock0, self.downblock1, self.downblock2, self.downblock3]

    def get(self, key: str) -> np.ndarray:
        if key not in self.weights.files:
            raise KeyError(f'Missing TimeAware VAE key: {key}')
        return self.weights[key].astype(np.float32)

    def run_conv_in(self, x: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            x,
            self.get('encoder__conv_in__weight'),
            self.get('encoder__conv_in__bias'),
            stride=1,
            padding=1,
        )

    def run_first_resnet_block(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.downblock0.run_resnet0(x, temb)

    def run_downblock0(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.downblock0.run_downblock0(x, temb)

    def run_downblock1(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.downblock1.run_downblock1(x, temb)

    def run_downblock2(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.downblock2.run_downblock2(x, temb)

    def run_downblock3(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.downblock3.run_downblock3(x, temb)

    def run_downblock(self, block_index: int, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        if block_index < 0 or block_index >= len(self.downblocks):
            raise NotImplementedError(f'encoder.down_blocks.{block_index} is not ported in TimeVAEBlockTester yet')
        return self.downblocks[block_index].run_downblock(x, temb)

    def run_encoder_until_block(self, block_index: int, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        if block_index < 0:
            return self.run_conv_in(x)
        x = self.run_conv_in(x)
        for idx in range(block_index + 1):
            x = self.run_downblock(idx, x, temb)
        return x.astype(np.float32)

    def run_encoder_stage01(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_encoder_until_block(1, x, temb)

    def run_encoder_stage012(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_encoder_until_block(2, x, temb)

    def run_encoder_stage0123(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.run_encoder_until_block(3, x, temb)

    def run_midblock(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        return self.midblock.run_midblock(x, temb)

    def run_encoder_stage0123_mid(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        x = self.run_encoder_stage0123(x, temb)
        return self.run_midblock(x, temb)

    def run_conv_norm_out(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_tail.run_conv_norm_out(x)

    def run_conv_act(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_tail.run_conv_act(x)

    def run_conv_out(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_tail.run_conv_out(x)

    def run_encoder_tail(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_tail.run_encoder_tail(x)

    def run_quant_conv(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_tail.run_quant_conv(x)

    def run_encoder_stage0123_mid_tail(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        x = self.run_encoder_stage0123_mid(x, temb)
        return self.run_encoder_tail(x)

    def run_encoder_stage0123_mid_tail_quant(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        x = self.run_encoder_stage0123_mid_tail(x, temb)
        return self.run_quant_conv(x)


    def run_moments_split(self, moments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.decoder_entry.run_moments_split(moments)

    def run_posterior_mode(self, moments: np.ndarray) -> np.ndarray:
        return self.decoder_entry.run_posterior_mode(moments)

    def run_post_quant_conv(self, z: np.ndarray) -> np.ndarray:
        return self.decoder_entry.run_post_quant_conv(z)

    def run_decoder_conv_in(self, z: np.ndarray) -> np.ndarray:
        return self.decoder_entry.run_decoder_conv_in(z)

    def run_decoder_entry(self, z: np.ndarray) -> np.ndarray:
        return self.decoder_entry.run_decoder_entry(z)

    def run_quant_to_decoder_entry(self, moments: np.ndarray) -> np.ndarray:
        return self.decoder_entry.run_quant_to_decoder_entry(moments)

    def run_encoder_stage0123_mid_tail_quant_to_decoder_entry(self, x: np.ndarray, temb: np.ndarray) -> np.ndarray:
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        return self.run_quant_to_decoder_entry(moments)



    def run_decoder_midblock(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        return self.decoder_midblock.run_midblock(x, temb=temb, return_intermediates=return_intermediates)

    def run_decoder_entry_midblock(self, z: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'post_quant_conv': post.astype(np.float32), 'decoder_conv_in': conv.astype(np.float32)}
            out.update(mid)
            return out
        return mid

    def run_quant_to_decoder_midblock(self, moments: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'posterior_mode': mode.astype(np.float32), 'post_quant_conv': post.astype(np.float32), 'decoder_conv_in': conv.astype(np.float32)}
            out.update(mid)
            return out
        return mid

    def run_encoder_stage0123_mid_tail_quant_to_decoder_midblock(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_midblock(moments, temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_decoder_upblock0_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock0.run_resnet0(x, temb)

    def run_decoder_upblock0_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock0.run_resnet1(x, temb)

    def run_decoder_upblock0_resnet2(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock0.run_resnet2(x, temb)

    def run_decoder_upblock0_upsampler0(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_upblock0.run_upsampler0(x)

    def run_decoder_upblock0(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        return self.decoder_upblock0.run_upblock(x, temb=temb, return_intermediates=return_intermediates)

    def run_decoder_upblock1_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock1.run_resnet0(x, temb)

    def run_decoder_upblock1_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock1.run_resnet1(x, temb)

    def run_decoder_upblock1_resnet2(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock1.run_resnet2(x, temb)

    def run_decoder_upblock1_upsampler0(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_upblock1.run_upsampler0(x)

    def run_decoder_upblock1(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        return self.decoder_upblock1.run_upblock(x, temb=temb, return_intermediates=return_intermediates)

    def run_decoder_upblock2_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock2.run_resnet0(x, temb)

    def run_decoder_upblock2_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock2.run_resnet1(x, temb)

    def run_decoder_upblock2_resnet2(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock2.run_resnet2(x, temb)

    def run_decoder_upblock2_upsampler0(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_upblock2.run_upsampler0(x)

    def run_decoder_upblock2(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        return self.decoder_upblock2.run_upblock(x, temb=temb, return_intermediates=return_intermediates)

    def run_decoder_upblock3_resnet0(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock3.run_resnet0(x, temb)

    def run_decoder_upblock3_resnet1(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock3.run_resnet1(x, temb)

    def run_decoder_upblock3_resnet2(self, x: np.ndarray, temb: np.ndarray | None = None) -> np.ndarray:
        return self.decoder_upblock3.run_resnet2(x, temb)

    def run_decoder_upblock3_upsampler0(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_upblock3.run_upsampler0(x)

    def run_decoder_upblock3(self, x: np.ndarray, temb: np.ndarray | None = None, return_intermediates: bool = False):
        return self.decoder_upblock3.run_upblock(x, temb=temb, return_intermediates=return_intermediates)

    def run_decoder_midblock_upblock0(self, x: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mid = self.run_decoder_midblock(x, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        up = self.run_decoder_upblock0(mid_out, temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'decoder_midblock': mid_out.astype(np.float32)}
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(up)
            return out
        return up

    def run_decoder_entry_midblock_upblock0(self, z: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        bridge = self.run_decoder_midblock_upblock0(conv, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'post_quant_conv': post.astype(np.float32), 'decoder_conv_in': conv.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_quant_to_decoder_upblock0(self, moments: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        bridge = self.run_decoder_midblock_upblock0(conv, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'posterior_mode': mode.astype(np.float32), 'post_quant_conv': post.astype(np.float32), 'decoder_conv_in': conv.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_encoder_stage0123_mid_tail_quant_to_decoder_upblock0(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_upblock0(moments, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_decoder_upblock0_upblock1(self, x: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        up0 = self.run_decoder_upblock0(x, temb=decoder_temb, return_intermediates=return_intermediates)
        up0_out = up0['output'] if return_intermediates else up0
        up1 = self.run_decoder_upblock1(up0_out, temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'decoder_upblock0': up0_out.astype(np.float32)}
            out.update({f'decoder_upblock0_{k}': v.astype(np.float32) for k, v in up0.items() if k != 'output'})
            out['decoder_upblock1'] = up1['output'].astype(np.float32)
            out.update({f'decoder_upblock1_{k}': v.astype(np.float32) for k, v in up1.items() if k != 'output'})
            return out
        return up1

    def run_decoder_upblocks012(self, x: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        up0 = self.run_decoder_upblock0(x, temb=decoder_temb, return_intermediates=return_intermediates)
        up0_out = up0['output'] if return_intermediates else up0
        up1 = self.run_decoder_upblock1(up0_out, temb=decoder_temb, return_intermediates=return_intermediates)
        up1_out = up1['output'] if return_intermediates else up1
        up2 = self.run_decoder_upblock2(up1_out, temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'decoder_upblock0': up0_out.astype(np.float32)}
            out.update({f'decoder_upblock0_{k}': v.astype(np.float32) for k, v in up0.items() if k != 'output'})
            out['decoder_upblock1'] = up1_out.astype(np.float32)
            out.update({f'decoder_upblock1_{k}': v.astype(np.float32) for k, v in up1.items() if k != 'output'})
            out['decoder_upblock2'] = up2['output'].astype(np.float32)
            out.update({f'decoder_upblock2_{k}': v.astype(np.float32) for k, v in up2.items() if k != 'output'})
            return out
        return up2

    def run_decoder_upblocks0123(self, x: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        up0 = self.run_decoder_upblock0(x, temb=decoder_temb, return_intermediates=return_intermediates)
        up0_out = up0['output'] if return_intermediates else up0
        up1 = self.run_decoder_upblock1(up0_out, temb=decoder_temb, return_intermediates=return_intermediates)
        up1_out = up1['output'] if return_intermediates else up1
        up2 = self.run_decoder_upblock2(up1_out, temb=decoder_temb, return_intermediates=return_intermediates)
        up2_out = up2['output'] if return_intermediates else up2
        up3 = self.run_decoder_upblock3(up2_out, temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'decoder_upblock0': up0_out.astype(np.float32)}
            out.update({f'decoder_upblock0_{k}': v.astype(np.float32) for k, v in up0.items() if k != 'output'})
            out['decoder_upblock1'] = up1_out.astype(np.float32)
            out.update({f'decoder_upblock1_{k}': v.astype(np.float32) for k, v in up1.items() if k != 'output'})
            out['decoder_upblock2'] = up2_out.astype(np.float32)
            out.update({f'decoder_upblock2_{k}': v.astype(np.float32) for k, v in up2.items() if k != 'output'})
            out['decoder_upblock3'] = up3['output'].astype(np.float32)
            out.update({f'decoder_upblock3_{k}': v.astype(np.float32) for k, v in up3.items() if k != 'output'})
            return out
        return up3

    def run_decoder_entry_midblock_upblocks01(self, z: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblock0_upblock1(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_decoder_entry_midblock_upblocks012(self, z: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks012(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_decoder_entry_midblock_upblocks0123(self, z: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks0123(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_quant_to_decoder_upblock1(self, moments: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblock0_upblock1(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'posterior_mode': mode.astype(np.float32),
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_quant_to_decoder_upblock2(self, moments: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks012(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'posterior_mode': mode.astype(np.float32),
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_quant_to_decoder_upblock3(self, moments: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks0123(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'posterior_mode': mode.astype(np.float32),
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_encoder_stage0123_mid_tail_quant_to_decoder_upblock1(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_upblock1(moments, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_encoder_stage0123_mid_tail_quant_to_decoder_upblock2(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_upblock2(moments, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge

    def run_encoder_stage0123_mid_tail_quant_to_decoder_upblock3(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_upblock3(moments, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge


    def run_decoder_tail_norm_out(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_tail.run_norm_out(x)

    def run_decoder_tail_act(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_tail.run_act(x)

    def run_decoder_tail_conv_out(self, x: np.ndarray) -> np.ndarray:
        return self.decoder_tail.run_conv_out(x)

    def run_decoder_tail(self, x: np.ndarray, return_intermediates: bool = False):
        return self.decoder_tail.run_decoder_tail(x, return_intermediates=return_intermediates)

    def run_decoder_upblocks0123_to_tail(self, x: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        up = self.run_decoder_upblocks0123(x, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        up_out = up['decoder_upblock3'] if return_intermediates else up
        tail = self.run_decoder_tail(up_out, return_intermediates=return_intermediates)
        if return_intermediates:
            out = dict(up)
            out.update({k: v.astype(np.float32) for k, v in tail.items() if k != 'output'})
            return out
        return tail

    def run_decoder_entry_midblock_upblocks0123_tail(self, z: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        post = self.run_post_quant_conv(z)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks0123_to_tail(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_quant_to_decoder_tail(self, moments: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        mode = self.run_posterior_mode(moments)
        post = self.run_post_quant_conv(mode)
        conv = self.run_decoder_conv_in(post)
        mid = self.run_decoder_midblock(conv, temb=decoder_temb, return_intermediates=return_intermediates)
        mid_out = mid['output'] if return_intermediates else mid
        bridge = self.run_decoder_upblocks0123_to_tail(mid_out, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'posterior_mode': mode.astype(np.float32),
                'post_quant_conv': post.astype(np.float32),
                'decoder_conv_in': conv.astype(np.float32),
                'decoder_midblock': mid_out.astype(np.float32),
            }
            out.update({f'decoder_midblock_{k}': v.astype(np.float32) for k, v in mid.items() if k != 'output'})
            out.update(bridge)
            return out
        return bridge

    def run_encoder_stage0123_mid_tail_quant_to_decoder_tail(self, x: np.ndarray, temb: np.ndarray, decoder_temb: np.ndarray | None = None, return_intermediates: bool = False):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, temb)
        bridge = self.run_quant_to_decoder_tail(moments, decoder_temb=decoder_temb, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {'moments': moments.astype(np.float32)}
            out.update(bridge)
            return out
        return bridge


class TimeVAEFullBoundaryTester(TimeVAEBlockTester):
    """Alignment-only TimeAware VAE boundary wrapper.

    The official TADSR pipeline uses encode(...).latent_dist.sample(),
    multiplies by scaling_factor, later divides by scaling_factor, decodes,
    and clamps to [-1, 1]. This helper reproduces that boundary with an
    exported deterministic epsilon and an exported encoder time embedding.

    It deliberately does not implement the scheduler, VAEHook tiling, image
    saving, generic runtime LoRA, or full TADSR inference.
    """

    def run_posterior_std(self, moments: np.ndarray) -> np.ndarray:
        return TimeVAEDiagonalGaussianHelper.std(moments)

    def run_posterior_sample_with_epsilon(self, moments: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        return TimeVAEDiagonalGaussianHelper.sample_with_epsilon(moments, epsilon)

    @staticmethod
    def scale_latent(z: np.ndarray, scaling_factor: float) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * np.float32(scaling_factor)).astype(np.float32)

    @staticmethod
    def unscale_latent(z: np.ndarray, scaling_factor: float) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) / np.float32(scaling_factor)).astype(np.float32)

    @staticmethod
    def clamp_output(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32), lo, hi).astype(np.float32)

    def run_decode_boundary(self, decode_input: np.ndarray, return_intermediates: bool = False):
        decoded = self.run_decoder_entry_midblock_upblocks0123_tail(decode_input, return_intermediates=return_intermediates)
        if return_intermediates:
            out = dict(decoded)
            out['decoded_output'] = decoded['decoder_tail'].astype(np.float32)
            out['final_clamped_output'] = self.clamp_output(decoded['decoder_tail'])
            return out
        return self.clamp_output(decoded)

    def run_encode_sample_scale_decode_clamp_boundary(
        self,
        x: np.ndarray,
        encoder_temb: np.ndarray,
        epsilon: np.ndarray,
        scaling_factor: float = 0.18215,
        return_intermediates: bool = False,
    ):
        moments = self.run_encoder_stage0123_mid_tail_quant(x, encoder_temb)
        mean, raw_logvar, logvar = self.run_moments_split(moments)
        std = self.run_posterior_std(moments)
        sample = self.run_posterior_sample_with_epsilon(moments, epsilon)
        scaled = self.scale_latent(sample, scaling_factor)
        decode_input = self.unscale_latent(scaled, scaling_factor)
        decoded = self.run_decode_boundary(decode_input, return_intermediates=return_intermediates)
        if return_intermediates:
            out = {
                'moments': moments.astype(np.float32),
                'posterior_mean': mean.astype(np.float32),
                'posterior_raw_logvar': raw_logvar.astype(np.float32),
                'posterior_logvar': logvar.astype(np.float32),
                'posterior_std': std.astype(np.float32),
                'posterior_sample': sample.astype(np.float32),
                'scaled_latent': scaled.astype(np.float32),
                'decode_input': decode_input.astype(np.float32),
            }
            out.update(decoded)
            return out
        return decoded


class TimeVAEActualHookBehaviorTester(TimeVAEFullBoundaryTester):
    """Alignment-only wrapper for the official *actual* VAEHook behavior.

    Official TADSR installs two VAE hooks with different dispatch contracts:

    - encoder: VAEHook(..., time_vae=True), so large inputs may enter the tiled
      task-queue encoder path;
    - decoder: VAEHook(..., time_vae=False), so __call__ dispatches to
      decoder.original_forward(x), not to a tiled decoder.

    This helper mirrors that public contract for alignment tests. The decoder
    side reuses the already-aligned original-forward decoder stack. The encoder
    side mirrors the official ``VAEHook.vae_tile_forward`` task queue for the
    deterministic alignment input, including official tile split/crop/write
    semantics and cross-tile ``GroupNormParam`` aggregation.
    """

    encoder_tiled_task_queue_implemented = True

    @staticmethod
    def _task(kind: str, payload=None, label: str = '') -> list:
        return [kind, payload, label or kind]

    @staticmethod
    def _norm_spec(weight: np.ndarray, bias: np.ndarray, label: str) -> dict:
        return {
            'weight': np.asarray(weight, dtype=np.float32),
            'bias': np.asarray(bias, dtype=np.float32),
            'label': label,
        }

    @staticmethod
    def _find_next_task(task_queue: list[list], kind: str) -> int:
        for idx, task in enumerate(task_queue):
            if task[0] == kind:
                return idx
        raise ValueError(f'No pending {kind!r} task found in official encoder tile queue')

    @staticmethod
    def _clone_task_queue(task_queue: list[list]) -> list[list]:
        cloned = []
        for kind, payload, label in task_queue:
            if kind in {'add_res', 'scale_shift'}:
                payload = None
            cloned.append([kind, payload, label])
        return cloned

    def _append_resblock_tasks(self, queue: list[list], block: TimeAwareResnetBlock, label: str) -> None:
        def residual_func(tile, block=block):
            if block.has('conv_shortcut__weight'):
                return conv2d_nchw(tile, block.get('conv_shortcut__weight'), block.get('conv_shortcut__bias'), stride=1, padding=0)
            if block.has('nin_shortcut__weight'):
                return conv2d_nchw(tile, block.get('nin_shortcut__weight'), block.get('nin_shortcut__bias'), stride=1, padding=0)
            return np.asarray(tile, dtype=np.float32)

        queue.append(self._task('store_res', residual_func, f'{label}.store_res'))
        queue.append(self._task('pre_norm', self._norm_spec(block.get('norm1__weight'), block.get('norm1__bias'), f'{label}.norm1'), f'{label}.norm1'))
        queue.append(self._task('silu', silu, f'{label}.silu1'))
        queue.append(self._task(
            'conv',
            lambda tile, block=block: conv2d_nchw(tile, block.get('conv1__weight'), block.get('conv1__bias'), stride=1, padding=1),
            f'{label}.conv1',
        ))
        queue.append(self._task('pre_norm', self._norm_spec(block.get('norm2__weight'), block.get('norm2__bias'), f'{label}.norm2'), f'{label}.norm2'))
        queue.append(self._task(
            'time_emb_proj',
            lambda temb, block=block: linear(temb, block.get('time_emb_proj__weight'), block.get('time_emb_proj__bias')),
            f'{label}.time_emb_proj',
        ))
        queue.append(self._task('scale_shift', None, f'{label}.scale_shift'))
        queue.append(self._task('silu', silu, f'{label}.silu2'))
        queue.append(self._task(
            'conv',
            lambda tile, block=block: conv2d_nchw(tile, block.get('conv2__weight'), block.get('conv2__bias'), stride=1, padding=1),
            f'{label}.conv2',
        ))
        queue.append(self._task('add_res', None, f'{label}.add_res'))

    def _append_attention_tasks(self, queue: list[list], attention: TimeVAEMidBlockAttention, label: str) -> None:
        queue.append(self._task('store_res', lambda tile: np.asarray(tile, dtype=np.float32), f'{label}.store_res'))
        queue.append(self._task(
            'pre_norm',
            self._norm_spec(attention.get('group_norm__weight'), attention.get('group_norm__bias'), f'{label}.group_norm'),
            f'{label}.group_norm',
        ))
        queue.append(self._task('attn', attention.run_attention_from_normalized_nchw, f'{label}.attention'))
        queue.append(self._task('add_res', None, f'{label}.add_res'))

    def _build_official_encoder_task_queue(self) -> list[list]:
        queue: list[list] = [
            self._task(
                'conv',
                lambda tile: conv2d_nchw(tile, self.get('encoder__conv_in__weight'), self.get('encoder__conv_in__bias'), stride=1, padding=1),
                'encoder.conv_in',
            )
        ]
        for block_index, block in enumerate(self.downblocks):
            for resnet_index, resnet in enumerate(block.resnets):
                self._append_resblock_tasks(queue, resnet, f'encoder.down_blocks.{block_index}.resnets.{resnet_index}')
            if block_index != len(self.downblocks) - 1 and block.downsampler0 is not None:
                queue.append(self._task('downsample', block.downsampler0, f'encoder.down_blocks.{block_index}.downsamplers.0'))

        self._append_resblock_tasks(queue, self.midblock.resnet0, 'encoder.mid_block.resnets.0')
        if self.midblock.attention0 is not None:
            self._append_attention_tasks(queue, self.midblock.attention0, 'encoder.mid_block.attentions.0')
        if self.midblock.resnet1 is not None:
            self._append_resblock_tasks(queue, self.midblock.resnet1, 'encoder.mid_block.resnets.1')

        queue.append(self._task(
            'pre_norm',
            self._norm_spec(
                self.get('encoder__conv_norm_out__weight'),
                self.get('encoder__conv_norm_out__bias'),
                'encoder.conv_norm_out',
            ),
            'encoder.conv_norm_out',
        ))
        queue.append(self._task('silu', silu, 'encoder.conv_act'))
        queue.append(self._task(
            'conv',
            lambda tile: conv2d_nchw(tile, self.get('encoder__conv_out__weight'), self.get('encoder__conv_out__bias'), stride=1, padding=1),
            'encoder.conv_out',
        ))
        return queue

    def official_encoder_tiled_task_queue_summary(self) -> dict:
        queue = self._build_official_encoder_task_queue()
        counts: dict[str, int] = {}
        rows = []
        for index, task in enumerate(queue):
            kind, _, label = task
            counts[kind] = counts.get(kind, 0) + 1
            rows.append({'index': index, 'task': kind, 'label': label})
        return {'task_count': len(queue), 'task_counts': counts, 'tasks': rows}

    def _execute_official_encoder_task(self, tile: np.ndarray, task: list, task_queue: list[list], activated_temb: np.ndarray) -> np.ndarray:
        kind, payload, _ = task
        if kind == 'apply_norm':
            return payload(tile).astype(np.float32)
        if kind == 'store_res':
            res = payload(tile).astype(np.float32)
            task_queue[self._find_next_task(task_queue, 'add_res')][1] = res
            return np.asarray(tile, dtype=np.float32)
        if kind == 'add_res':
            if payload is None:
                raise ValueError('Official encoder tile queue reached add_res without stored residual')
            return (np.asarray(tile, dtype=np.float32) + np.asarray(payload, dtype=np.float32)).astype(np.float32)
        if kind == 'time_emb_proj':
            projected = payload(activated_temb).astype(np.float32)[:, :, None, None]
            task_queue[self._find_next_task(task_queue, 'scale_shift')][1] = projected
            return np.asarray(tile, dtype=np.float32)
        if kind == 'scale_shift':
            if payload is None:
                raise ValueError('Official encoder tile queue reached scale_shift without time_emb_proj output')
            scale, shift = np.split(np.asarray(payload, dtype=np.float32), 2, axis=1)
            return (np.asarray(tile, dtype=np.float32) * (scale + np.float32(1.0)) + shift).astype(np.float32)
        if kind in {'conv', 'silu', 'downsample', 'attn'}:
            return payload(tile).astype(np.float32)
        raise NotImplementedError(f'Unsupported official encoder tile task: {kind}')

    def run_actual_encoder_tiled_raw_for_alignment(
        self,
        x: np.ndarray,
        timestep: np.ndarray,
        tile_size: int = 16,
        pad: int = 32,
        return_intermediates: bool = False,
    ):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f'official actual encoder hook expects NCHW input, got {x.shape}')
        _, _, height, width = x.shape
        in_bboxes, out_bboxes, tile_metadata = official_split_tiles(height, width, tile_size=tile_size, pad=pad, is_decoder=False)
        tiles = [x[:, :, bbox[2]:bbox[3], bbox[0]:bbox[1]].astype(np.float32) for bbox in in_bboxes]
        base_queue = self._build_official_encoder_task_queue()
        task_queues = [self._clone_task_queue(base_queue) for _ in tiles]
        encoder_temb = self.run_encoder_time_embedding(timestep)
        activated_temb = silu(encoder_temb)
        result = None
        completed = 0
        norm_rounds = []
        executed_task_count = 0

        while completed < len(tiles):
            group_norm_param = OfficialGroupNormParam()
            saw_pre_norm = False
            made_progress = False
            for tile_index, task_queue in enumerate(task_queues):
                if task_queue is None:
                    continue
                tile = tiles[tile_index]
                while task_queue:
                    task = task_queue.pop(0)
                    if task[0] == 'pre_norm':
                        group_norm_param.add_tile(tile, task[1])
                        saw_pre_norm = True
                        made_progress = True
                        break
                    tile = self._execute_official_encoder_task(tile, task, task_queue, activated_temb)
                    executed_task_count += 1
                    made_progress = True
                tiles[tile_index] = tile.astype(np.float32)
                if len(task_queue) == 0:
                    if result is None:
                        result = np.zeros((x.shape[0], tile.shape[1], height // 8, width // 8), dtype=np.float32)
                    cropped = official_crop_valid_region(tile, in_bboxes[tile_index], out_bboxes[tile_index], is_decoder=False)
                    ob = out_bboxes[tile_index]
                    result[:, :, ob[2]:ob[3], ob[0]:ob[1]] = cropped
                    task_queues[tile_index] = None
                    completed += 1

            if completed >= len(tiles):
                break
            group_norm_func, stats = group_norm_param.summary()
            if saw_pre_norm and group_norm_func is not None:
                norm_rounds.append(stats)
                for queue in task_queues:
                    if queue is not None:
                        queue.insert(0, self._task('apply_norm', group_norm_func, f"apply_norm:{stats.get('label', 'pre_norm')}"))
            elif not made_progress:
                raise RuntimeError('Official encoder tile queue made no progress; likely malformed task queue')

        if result is None:
            raise RuntimeError('Official encoder tiled path produced no result')

        tile_records = []
        for idx, (ib, ob) in enumerate(zip(in_bboxes, out_bboxes)):
            padded_bbox = [v // 8 for v in ib]
            margin = [ob[i] - padded_bbox[i] for i in range(4)]
            tile_records.append({
                'tile_index': idx,
                'input_bbox': ib,
                'output_bbox': ob,
                'padded_output_bbox': padded_bbox,
                'crop_margin': margin,
            })

        info = {
            'raw_encoder_output': result.astype(np.float32),
            'encoder_temb': encoder_temb.astype(np.float32),
            'encoder_temb_after_official_silu': activated_temb.astype(np.float32),
            'tile_metadata': tile_metadata,
            'tile_records': tile_records,
            'task_queue_summary': self.official_encoder_tiled_task_queue_summary(),
            'group_norm_rounds': norm_rounds,
            'executed_task_count': executed_task_count,
            'write_back_rule': 'assignment_no_blending',
        }
        return info if return_intermediates else result.astype(np.float32)

    def run_encoder_time_embedding(self, timestep: np.ndarray) -> np.ndarray:
        raw = timestep_embedding_np(
            timestep,
            embedding_dim=128,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            max_period=10000,
        )
        h = linear(
            raw,
            self.weights['encoder__time_embedding__linear_1__weight'],
            self.weights['encoder__time_embedding__linear_1__bias'],
        )
        h = silu(h)
        h = linear(
            h,
            self.weights['encoder__time_embedding__linear_2__weight'],
            self.weights['encoder__time_embedding__linear_2__bias'],
        )
        return h.astype(np.float32)

    @staticmethod
    def official_decoder_hook_policy() -> dict:
        return {
            'decoder_hook_installed': True,
            'decoder_time_vae': False,
            'decoder_original_forward_used': True,
            'decoder_tiled_path_triggered': False,
            'decoder_tiled_path_status': 'NOT_APPLICABLE_BLOCKED_DECODER_HOOK_CONTRACT',
        }

    @staticmethod
    def official_encoder_hook_policy() -> dict:
        return {
            'encoder_hook_installed': True,
            'encoder_time_vae': True,
            'encoder_tiled_capable': True,
            'jittor_exact_tiled_task_queue_implemented': True,
            'tile_size': 16,
            'pad': 32,
            'note': 'The Jittor alignment helper mirrors official encoder VAEHook.vae_tile_forward task queue, split/crop/write semantics, and cross-tile GroupNorm aggregation.',
        }

    def run_actual_encoder_hook_boundary_for_alignment(
        self,
        x: np.ndarray,
        timestep: np.ndarray,
        return_intermediates: bool = False,
    ):
        tiled = self.run_actual_encoder_tiled_raw_for_alignment(x, timestep, return_intermediates=True)
        moments = self.run_quant_conv(tiled['raw_encoder_output'])
        if return_intermediates:
            return {
                'moments': moments.astype(np.float32),
                'raw_encoder_output': tiled['raw_encoder_output'].astype(np.float32),
                'encoder_temb': tiled['encoder_temb'].astype(np.float32),
                'encoder_temb_after_official_silu': tiled['encoder_temb_after_official_silu'].astype(np.float32),
                'encoder_policy': self.official_encoder_hook_policy(),
                'tile_metadata': tiled['tile_metadata'],
                'tile_records': tiled['tile_records'],
                'task_queue_summary': tiled['task_queue_summary'],
                'group_norm_rounds': tiled['group_norm_rounds'],
            }
        return moments.astype(np.float32)

    def run_actual_decoder_hook_boundary_for_alignment(
        self,
        decode_input: np.ndarray,
        return_intermediates: bool = False,
    ):
        decoded = self.run_decode_boundary(decode_input, return_intermediates=True)
        decoded['decoder_policy'] = self.official_decoder_hook_policy()
        if return_intermediates:
            return decoded
        return decoded['final_clamped_output']

    def run_actual_vaehook_behavior_boundary_for_alignment(
        self,
        x: np.ndarray,
        timestep: np.ndarray,
        epsilon: np.ndarray,
        scaling_factor: float = 0.18215,
        return_intermediates: bool = False,
    ):
        enc = self.run_actual_encoder_hook_boundary_for_alignment(x, timestep, return_intermediates=True)
        moments = enc['moments']
        mean, raw_logvar, logvar = self.run_moments_split(moments)
        std = self.run_posterior_std(moments)
        sample = self.run_posterior_sample_with_epsilon(moments, epsilon)
        scaled = self.scale_latent(sample, scaling_factor)
        decode_input = self.unscale_latent(scaled, scaling_factor)
        dec = self.run_actual_decoder_hook_boundary_for_alignment(decode_input, return_intermediates=True)
        if return_intermediates:
            out = {
                'moments': moments.astype(np.float32),
                'raw_encoder_output': enc['raw_encoder_output'].astype(np.float32),
                'encoder_temb': enc['encoder_temb'].astype(np.float32),
                'encoder_temb_after_official_silu': enc['encoder_temb_after_official_silu'].astype(np.float32),
                'posterior_mean': mean.astype(np.float32),
                'posterior_raw_logvar': raw_logvar.astype(np.float32),
                'posterior_logvar': logvar.astype(np.float32),
                'posterior_std': std.astype(np.float32),
                'posterior_sample': sample.astype(np.float32),
                'scaled_latent': scaled.astype(np.float32),
                'decode_input': decode_input.astype(np.float32),
                'encoder_policy': enc['encoder_policy'],
                'tile_metadata': enc['tile_metadata'],
                'tile_records': enc['tile_records'],
                'task_queue_summary': enc['task_queue_summary'],
                'group_norm_rounds': enc['group_norm_rounds'],
            }
            out.update(dec)
            return out
        return dec['final_clamped_output']


class TimeAwareVAEPortSkeleton:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Full TimeAware VAE forward is not implemented yet; use TimeVAEBlockTester for block-level migration.')
