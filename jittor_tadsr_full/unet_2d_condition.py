from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from .vae_time_aware import conv2d_nchw, group_norm, linear, silu, softmax, upsample_nearest2d

DEFAULT_UNET_ENTRY_NPZ = Path('experiments/full_repro/unet_alignment/converted_unet_entry_effective_weights.npz')
DEFAULT_UNET_NPZ = Path('/mnt/data/sj/checkpoints/TADSR/jittor_converted/diffusers/unet_weights.npz')
UNET_GROUPS = ['conv_in', 'down_blocks', 'mid_block', 'up_blocks', 'conv_out', 'attention', 'time_embedding', 'other']


def npz_prefix(prefix: str) -> str:
    return prefix.replace('.', '__')


def unet_key_groups(npz_path: str | Path):
    path = Path(npz_path)
    if not path.exists():
        return {'status': 'MISSING', 'path': str(path), 'groups': {}}
    data = np.load(path)
    groups = {k: [] for k in UNET_GROUPS}
    for key in data.files:
        low = key.lower()
        group = 'other'
        if 'attn' in low or 'attention' in low:
            group = 'attention'
        else:
            for candidate in ['conv_in', 'down_blocks', 'mid_block', 'up_blocks', 'conv_out', 'time_embedding']:
                if candidate in low:
                    group = candidate
                    break
        groups[group].append({'key': key, 'shape': list(data[key].shape), 'dtype': str(data[key].dtype)})
    return {'status': 'PARTIAL', 'path': str(path), 'groups': {k: v[:100] for k, v in groups.items()}, 'counts': {k: len(v) for k, v in groups.items()}, 'note': 'UNet key grouping plus entry/timestep helper modules; full UNet forward is not implemented.'}


def get_timestep_embedding_jittor_or_numpy(
    timesteps,
    num_channels: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> np.ndarray:
    ts = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    half_dim = num_channels // 2
    if half_dim == 0:
        raise ValueError(f'num_channels must be positive, got {num_channels}')
    exponent = -math.log(max_period) * np.arange(half_dim, dtype=np.float32)
    exponent = exponent / (half_dim - float(downscale_freq_shift))
    emb = np.exp(exponent).astype(np.float32)[None, :] * ts[:, None]
    emb = float(scale) * emb
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1).astype(np.float32)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1).astype(np.float32)
    if num_channels % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    return emb.astype(np.float32)


class UNetTimestepsTester:
    def __init__(self, num_channels: int = 320, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0.0, scale: float = 1.0, max_period: int = 10000):
        self.num_channels = int(num_channels)
        self.flip_sin_to_cos = bool(flip_sin_to_cos)
        self.downscale_freq_shift = float(downscale_freq_shift)
        self.scale = float(scale)
        self.max_period = int(max_period)

    def run_time_proj(self, timesteps) -> np.ndarray:
        return get_timestep_embedding_jittor_or_numpy(
            timesteps,
            num_channels=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            max_period=self.max_period,
        )


class UNetTimestepEmbeddingTester:
    def __init__(self, weights: str | Path | np.lib.npyio.NpzFile = DEFAULT_UNET_ENTRY_NPZ):
        self.weights = weights if hasattr(weights, 'files') else np.load(Path(weights))
        self._validate()

    def key(self, suffix: str) -> str:
        return f'time_embedding__{suffix}'

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            raise KeyError(f'Missing UNet time embedding key: {key}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(k) for k in ['linear_1__weight', 'linear_1__bias', 'linear_2__weight', 'linear_2__bias'] if self.key(k) not in self.weights.files]
        if missing:
            raise KeyError(f'Missing UNet time embedding weights: {missing}')

    def run_linear1(self, time_proj: np.ndarray) -> np.ndarray:
        return linear(time_proj, self.get('linear_1__weight'), self.get('linear_1__bias'))

    def run_act(self, linear1_out: np.ndarray) -> np.ndarray:
        return silu(linear1_out)

    def run_linear2(self, act_out: np.ndarray) -> np.ndarray:
        return linear(act_out, self.get('linear_2__weight'), self.get('linear_2__bias'))

    def run_time_embedding(self, time_proj: np.ndarray) -> np.ndarray:
        return self.run_linear2(self.run_act(self.run_linear1(time_proj)))


class TADSRUNetEntryTester:
    def __init__(
        self,
        weights: str | Path | np.lib.npyio.NpzFile = DEFAULT_UNET_ENTRY_NPZ,
        center_input_sample: bool = False,
        time_num_channels: int = 320,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
    ):
        self.weights = weights if hasattr(weights, 'files') else np.load(Path(weights))
        self.center_input_sample = bool(center_input_sample)
        self.timesteps = UNetTimestepsTester(time_num_channels, flip_sin_to_cos, downscale_freq_shift)
        self.time_embedding = UNetTimestepEmbeddingTester(self.weights)
        self._validate()

    def _validate(self) -> None:
        missing = [k for k in ['conv_in__weight', 'conv_in__bias'] if k not in self.weights.files]
        if missing:
            raise KeyError(f'Missing UNet entry weights: {missing}')

    def get(self, key: str) -> np.ndarray:
        if key not in self.weights.files:
            raise KeyError(f'Missing UNet entry key: {key}')
        return self.weights[key].astype(np.float32)

    def run_center_input(self, sample: np.ndarray) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32)
        return (2.0 * x - 1.0).astype(np.float32) if self.center_input_sample else x

    def run_conv_in(self, sample: np.ndarray, already_centered: bool = False) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32) if already_centered else self.run_center_input(sample)
        return conv2d_nchw(x, self.get('conv_in__weight'), self.get('conv_in__bias'), stride=1, padding=1)

    def run_time_proj(self, timesteps) -> np.ndarray:
        return self.timesteps.run_time_proj(timesteps)

    def run_time_embedding(self, time_proj: np.ndarray) -> np.ndarray:
        return self.time_embedding.run_time_embedding(time_proj)

    def run_entry(self, sample: np.ndarray, timesteps) -> dict[str, np.ndarray]:
        centered = self.run_center_input(sample)
        conv = self.run_conv_in(centered, already_centered=True)
        time_proj = self.run_time_proj(timesteps)
        linear1 = self.time_embedding.run_linear1(time_proj)
        act = self.time_embedding.run_act(linear1)
        temb = self.time_embedding.run_linear2(act)
        return {
            'centered_sample': centered,
            'conv_in': conv,
            'time_proj': time_proj,
            'time_embedding_linear1': linear1,
            'time_embedding_act': act,
            'time_embedding': temb,
        }


class UNet2DConditionPortSkeleton:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Full UNet forward is not implemented yet; port ResNet and attention blocks first.')



class UNetResnetBlock2DTester:
    def __init__(self, weights_npz_path: str | Path, prefix: str, metadata: dict | None = None):
        self.weights_path = Path(weights_npz_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f'UNet ResnetBlock2D effective weights missing: {self.weights_path}')
        self.weights = np.load(self.weights_path)
        self.prefix = prefix
        metadata = metadata or {}
        cfg = metadata.get('resnet_config')
        if cfg is None:
            normalized_prefix = str(prefix).replace('__', '_')
            if normalized_prefix.endswith('resnets_1'):
                cfg = metadata.get('resnet1_config')
            elif normalized_prefix.endswith('resnets_0'):
                cfg = metadata.get('resnet0_config')
        if cfg is None:
            cfg = metadata
        self.groups = int(cfg.get('groups', 32))
        self.groups_out = int(cfg.get('groups_out', self.groups))
        self.eps = float(cfg.get('eps', 1e-5))
        self.time_embedding_norm = str(cfg.get('time_embedding_norm', 'default'))
        self.output_scale_factor = float(cfg.get('output_scale_factor', 1.0))
        self.dropout = float(cfg.get('dropout', 0.0))
        self.skip_time_act = bool(cfg.get('skip_time_act', False))
        self.has_shortcut = bool(cfg.get('has_shortcut', False))
        if self.dropout != 0.0:
            raise NotImplementedError(f'UNet ResnetBlock2DTester only supports eval/dropout=0.0, got {self.dropout}')
        if self.time_embedding_norm not in {'default', 'scale_shift'}:
            raise NotImplementedError(f'Unsupported UNet Resnet time_embedding_norm: {self.time_embedding_norm}')
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:40]
            raise KeyError(f'Missing UNet Resnet key {key}; nearby={nearby}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        required = [
            'norm1_weight', 'norm1_bias', 'conv1_weight', 'conv1_bias',
            'time_emb_proj_weight', 'time_emb_proj_bias',
            'norm2_weight', 'norm2_bias', 'conv2_weight', 'conv2_bias',
        ]
        if self.has_shortcut:
            required += ['conv_shortcut_weight']
        missing = [self.key(s) for s in required if not self.has(s)]
        if missing:
            nearby = [k for k in self.weights.files if self.prefix in k][:80]
            raise KeyError(f'Missing keys for {self.prefix}: {missing}; nearby={nearby}')

    def run_norm1(self, x: np.ndarray) -> np.ndarray:
        return group_norm(x, self.get('norm1_weight'), self.get('norm1_bias'), self.groups, self.eps)

    def run_conv1_path(self, x: np.ndarray) -> dict[str, np.ndarray]:
        norm1 = self.run_norm1(x)
        act1 = silu(norm1)
        conv1 = conv2d_nchw(act1, self.get('conv1_weight'), self.get('conv1_bias'), stride=1, padding=1)
        return {'norm1': norm1, 'act1': act1, 'conv1': conv1}

    def run_time_emb_proj(self, emb: np.ndarray) -> np.ndarray:
        temb_in = np.asarray(emb, dtype=np.float32) if self.skip_time_act else silu(emb)
        return linear(temb_in, self.get('time_emb_proj_weight'), self.get('time_emb_proj_bias'))

    def run_after_temb_add(self, conv1_out: np.ndarray, temb_proj: np.ndarray) -> np.ndarray:
        temb4 = np.asarray(temb_proj, dtype=np.float32)[:, :, None, None]
        if self.time_embedding_norm == 'default':
            return (np.asarray(conv1_out, dtype=np.float32) + temb4).astype(np.float32)
        if self.time_embedding_norm == 'scale_shift':
            return np.asarray(conv1_out, dtype=np.float32)
        raise NotImplementedError(f'Unsupported time_embedding_norm: {self.time_embedding_norm}')

    def run_norm2_act_conv2(self, h: np.ndarray, temb_proj: np.ndarray | None = None) -> dict[str, np.ndarray]:
        norm2 = group_norm(h, self.get('norm2_weight'), self.get('norm2_bias'), self.groups_out, self.eps)
        if self.time_embedding_norm == 'scale_shift':
            if temb_proj is None:
                raise ValueError('scale_shift requires temb_proj')
            scale, shift = np.split(temb_proj[:, :, None, None], 2, axis=1)
            norm2 = norm2 * (1.0 + scale) + shift
        act2 = silu(norm2)
        conv2 = conv2d_nchw(act2, self.get('conv2_weight'), self.get('conv2_bias'), stride=1, padding=1)
        return {'norm2': norm2.astype(np.float32), 'act2': act2.astype(np.float32), 'conv2': conv2.astype(np.float32)}

    def run_shortcut(self, x: np.ndarray) -> np.ndarray:
        if self.has('conv_shortcut_weight'):
            bias = self.get('conv_shortcut_bias') if self.has('conv_shortcut_bias') else None
            return conv2d_nchw(x, self.get('conv_shortcut_weight'), bias, stride=1, padding=0)
        return np.asarray(x, dtype=np.float32)

    def run(self, x: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        conv1_path = self.run_conv1_path(x)
        temb_proj = self.run_time_emb_proj(emb)
        after = self.run_after_temb_add(conv1_path['conv1'], temb_proj)
        conv2_path = self.run_norm2_act_conv2(after, temb_proj=temb_proj)
        shortcut = self.run_shortcut(x)
        out = ((shortcut + conv2_path['conv2']) / self.output_scale_factor).astype(np.float32)
        if not return_intermediates:
            return out
        return {
            'norm1': conv1_path['norm1'],
            'act1': conv1_path['act1'],
            'conv1': conv1_path['conv1'],
            'time_emb_proj': temb_proj.astype(np.float32),
            'after_temb_add': after.astype(np.float32),
            'norm2': conv2_path['norm2'],
            'act2': conv2_path['act2'],
            'conv2': conv2_path['conv2'],
            'shortcut': shortcut.astype(np.float32),
            'output': out,
        }


class TADSRUNetDownBlock0Resnet0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        cfg = metadata.get('config', {})
        resnet_cfg = metadata.get('resnet0_config', {})
        self.entry = TADSRUNetEntryTester(
            entry_weights_npz_path,
            center_input_sample=bool(cfg.get('center_input_sample', False)),
            time_num_channels=int(cfg.get('block_out_channels', [320])[0]),
            flip_sin_to_cos=bool(cfg.get('flip_sin_to_cos', True)),
            downscale_freq_shift=float(cfg.get('freq_shift', 0)),
        )
        self.resnet0 = UNetResnetBlock2DTester(resnet0_weights_npz_path, resnet_cfg.get('prefix', 'down_blocks_0_resnets_0'), metadata)

    def run_downblock0_resnet0(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet0.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_to_downblock0_resnet0(self, sample: np.ndarray, timestep, return_intermediates: bool = False):
        entry_out = self.entry.run_entry(sample, timestep)
        res = self.resnet0.run(entry_out['conv_in'], entry_out['time_embedding'], return_intermediates=True)
        out = {
            'centered_sample': entry_out['centered_sample'],
            'conv_in': entry_out['conv_in'],
            'time_proj': entry_out['time_proj'],
            'time_embedding': entry_out['time_embedding'],
            'resnet0_output': res['output'],
        }
        if return_intermediates:
            out['resnet0_intermediates'] = res
        return out



def layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    return (y * np.asarray(weight, dtype=np.float32) + np.asarray(bias, dtype=np.float32)).astype(np.float32)


def gelu_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.special import erf as _erf
        erf_vals = _erf(x / np.sqrt(2.0))
    except Exception:
        import math
        erf_vals = np.vectorize(math.erf, otypes=[np.float32])(x / np.sqrt(2.0))
    return (0.5 * x * (1.0 + erf_vals)).astype(np.float32)


class UNetAttention0Transformer2DTester:
    def __init__(self, weights_npz_path: str | Path, metadata: dict | None = None):
        self.weights_path = Path(weights_npz_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f'UNet attention0 effective weights missing: {self.weights_path}')
        self.weights = np.load(self.weights_path)
        metadata = metadata or {}
        cfg = metadata.get('attention_config')
        if cfg is None:
            if 'attention1_config' in metadata:
                cfg = metadata.get('attention1_config')
            else:
                cfg = metadata.get('attention0_config', metadata)
        tcfg = metadata.get('transformer0_config', {})
        self.prefix = cfg.get('prefix', 'down_blocks_0_attentions_0')
        self.groups = int(cfg.get('norm_num_groups', 32))
        self.norm_eps = float(cfg.get('norm_eps', 1e-6))
        self.use_linear_projection = bool(cfg.get('use_linear_projection', True))
        self.in_channels = int(cfg.get('in_channels', 320))
        self.inner_dim = int(cfg.get('inner_dim', 320))
        self.heads = int(tcfg.get('heads', cfg.get('num_attention_heads', 5)))
        self.head_dim = int(tcfg.get('head_dim', self.inner_dim // self.heads))
        self.layer_norm_eps = float(tcfg.get('norm_eps', 1e-5))
        self.dropout = float(tcfg.get('dropout', 0.0))
        if not self.use_linear_projection:
            raise NotImplementedError('This tester currently supports official use_linear_projection=True only.')
        if self.dropout != 0.0:
            raise NotImplementedError(f'Attention0 tester supports eval/dropout=0.0 only, got {self.dropout}')
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:80]
            raise KeyError(f'Missing UNet attention0 key {key}; nearby={nearby}')
        return self.weights[key].astype(np.float32)

    def get_bias(self, suffix: str):
        return self.get(suffix) if self.has(suffix) else None

    def _validate(self) -> None:
        required = [
            'norm_weight', 'norm_bias', 'proj_in_weight', 'proj_in_bias', 'proj_out_weight', 'proj_out_bias',
            'transformer0_norm1_weight', 'transformer0_norm1_bias',
            'transformer0_norm2_weight', 'transformer0_norm2_bias',
            'transformer0_norm3_weight', 'transformer0_norm3_bias',
            'transformer0_attn1_to_q_weight', 'transformer0_attn1_to_k_weight', 'transformer0_attn1_to_v_weight',
            'transformer0_attn1_to_out_0_weight', 'transformer0_attn1_to_out_0_bias',
            'transformer0_attn2_to_q_weight', 'transformer0_attn2_to_k_weight', 'transformer0_attn2_to_v_weight',
            'transformer0_attn2_to_out_0_weight', 'transformer0_attn2_to_out_0_bias',
            'transformer0_ff_geglu_proj_weight', 'transformer0_ff_geglu_proj_bias',
            'transformer0_ff_out_weight', 'transformer0_ff_out_bias',
        ]
        missing = [self.key(s) for s in required if not self.has(s)]
        if missing:
            nearby = [k for k in self.weights.files if self.prefix in k][:120]
            raise KeyError(f'Missing attention0 keys: {missing}; nearby={nearby}')

    def run_norm(self, x: np.ndarray) -> np.ndarray:
        return group_norm(x, self.get('norm_weight'), self.get('norm_bias'), self.groups, self.norm_eps)

    def nchw_to_sequence(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        b, c, h, w = x.shape
        return x.transpose(0, 2, 3, 1).reshape(b, h * w, c).astype(np.float32)

    def sequence_to_nchw(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        b, _, c = x.shape
        return x.reshape(b, height, width, c).transpose(0, 3, 1, 2).astype(np.float32)

    def run_input_projection(self, x: np.ndarray) -> dict[str, np.ndarray]:
        norm = self.run_norm(x)
        seq = self.nchw_to_sequence(norm)
        proj = linear(seq, self.get('proj_in_weight'), self.get('proj_in_bias'))
        return {'norm': norm, 'sequence_input': seq, 'proj_in': proj}

    def run_attention(self, name: str, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray | None = None) -> dict[str, np.ndarray]:
        prefix = f'transformer0_{name}'
        q = linear(hidden_states, self.get(f'{prefix}_to_q_weight'), self.get_bias(f'{prefix}_to_q_bias'))
        source = hidden_states if encoder_hidden_states is None else np.asarray(encoder_hidden_states, dtype=np.float32)
        k = linear(source, self.get(f'{prefix}_to_k_weight'), self.get_bias(f'{prefix}_to_k_bias'))
        v = linear(source, self.get(f'{prefix}_to_v_weight'), self.get_bias(f'{prefix}_to_v_bias'))
        b = q.shape[0]
        qh = q.reshape(b, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(b, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(b, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = np.matmul(qh.astype(np.float32), kh.astype(np.float32).transpose(0, 1, 3, 2)) / np.sqrt(float(self.head_dim))
        probs = softmax(scores, axis=-1)
        ctx = np.matmul(probs.astype(np.float32), vh.astype(np.float32))
        ctx = ctx.transpose(0, 2, 1, 3).reshape(b, -1, self.heads * self.head_dim).astype(np.float32)
        out = linear(ctx, self.get(f'{prefix}_to_out_0_weight'), self.get(f'{prefix}_to_out_0_bias'))
        return {'q': q.astype(np.float32), 'k': k.astype(np.float32), 'v': v.astype(np.float32), 'output': out.astype(np.float32)}

    def run_attn1(self, x: np.ndarray) -> dict[str, np.ndarray]:
        n1 = layer_norm_np(x, self.get('transformer0_norm1_weight'), self.get('transformer0_norm1_bias'), self.layer_norm_eps)
        attn = self.run_attention('attn1', n1)
        after = (attn['output'] + np.asarray(x, dtype=np.float32)).astype(np.float32)
        return {'norm1': n1, **{f'attn1_{k}': v for k, v in attn.items()}, 'after_attn1': after}

    def run_attn2(self, x: np.ndarray, encoder_hidden_states: np.ndarray) -> dict[str, np.ndarray]:
        n2 = layer_norm_np(x, self.get('transformer0_norm2_weight'), self.get('transformer0_norm2_bias'), self.layer_norm_eps)
        attn = self.run_attention('attn2', n2, encoder_hidden_states)
        after = (attn['output'] + np.asarray(x, dtype=np.float32)).astype(np.float32)
        return {'norm2': n2, **{f'attn2_{k}': v for k, v in attn.items()}, 'after_attn2': after}

    def run_ff(self, x: np.ndarray) -> dict[str, np.ndarray]:
        n3 = layer_norm_np(x, self.get('transformer0_norm3_weight'), self.get('transformer0_norm3_bias'), self.layer_norm_eps)
        geglu_proj = linear(n3, self.get('transformer0_ff_geglu_proj_weight'), self.get('transformer0_ff_geglu_proj_bias'))
        hidden, gate = np.split(geglu_proj, 2, axis=-1)
        geglu = (hidden * gelu_np(gate)).astype(np.float32)
        ff_out = linear(geglu, self.get('transformer0_ff_out_weight'), self.get('transformer0_ff_out_bias'))
        out = (ff_out + np.asarray(x, dtype=np.float32)).astype(np.float32)
        return {'norm3': n3, 'ff_geglu_proj': geglu_proj.astype(np.float32), 'ff_geglu_output': geglu, 'ff_output': ff_out.astype(np.float32), 'transformer0_output': out}

    def run_transformer0(self, seq: np.ndarray, encoder_hidden_states: np.ndarray) -> dict[str, np.ndarray]:
        a1 = self.run_attn1(seq)
        a2 = self.run_attn2(a1['after_attn1'], encoder_hidden_states)
        ff = self.run_ff(a2['after_attn2'])
        return {**a1, **a2, **ff}

    def run_proj_out(self, transformer_out: np.ndarray, height: int, width: int) -> dict[str, np.ndarray]:
        seq = linear(transformer_out, self.get('proj_out_weight'), self.get('proj_out_bias'))
        nchw = self.sequence_to_nchw(seq, height, width)
        return {'proj_out_sequence': seq.astype(np.float32), 'proj_out_nchw': nchw.astype(np.float32)}

    def run_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        x = np.asarray(hidden_states, dtype=np.float32)
        b, c, h, w = x.shape
        inp = self.run_input_projection(x)
        tr = self.run_transformer0(inp['proj_in'], encoder_hidden_states)
        po = self.run_proj_out(tr['transformer0_output'], h, w)
        out = (po['proj_out_nchw'] + x).astype(np.float32)
        if not return_intermediates:
            return out
        return {**inp, **tr, **po, 'output': out}


class TADSRUNetEntryResnet0Attention0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.entry_resnet0 = TADSRUNetDownBlock0Resnet0Tester(entry_weights_npz_path, resnet0_weights_npz_path, metadata=metadata.get('resnet0_metadata', metadata))
        self.attention0 = UNetAttention0Transformer2DTester(attention0_weights_npz_path, metadata=metadata)

    def run_entry_resnet0_attention0(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        er = self.entry_resnet0.run_entry_to_downblock0_resnet0(sample, timestep, return_intermediates=True)
        att = self.attention0.run_attention0(er['resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': er['conv_in'],
            'time_proj': er['time_proj'],
            'time_embedding': er['time_embedding'],
            'resnet0_output': er['resnet0_output'],
            'attention0_output': att['output'],
        }
        if return_intermediates:
            out['attention0_intermediates'] = att
            out['resnet0_intermediates'] = er.get('resnet0_intermediates', {})
        return out


class TADSRUNetEntryResnet0Attention0Resnet1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        attention_meta = metadata.get('attention0_metadata', metadata)
        if 'resnet0_metadata' not in attention_meta:
            attention_meta = {**attention_meta, 'resnet0_metadata': metadata.get('resnet0_metadata', {})}
        self.entry_resnet0_attention0 = TADSRUNetEntryResnet0Attention0Tester(
            entry_weights_npz_path,
            resnet0_weights_npz_path,
            attention0_weights_npz_path,
            metadata=attention_meta,
        )
        resnet1_meta = metadata.get('resnet1_metadata', metadata)
        prefix = resnet1_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_0_resnets_1')
        self.resnet1 = UNetResnetBlock2DTester(resnet1_weights_npz_path, prefix, resnet1_meta)

    def run_downblock0_resnet1(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet1.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_resnet0_attention0_resnet1(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.entry_resnet0_attention0.run_entry_resnet0_attention0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        res1 = self.resnet1.run(prev['attention0_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'resnet0_output': prev['resnet0_output'],
            'attention0_output': prev['attention0_output'],
            'resnet1_output': res1['output'],
        }
        if return_intermediates:
            out['attention0_intermediates'] = prev.get('attention0_intermediates', {})
            out['resnet0_intermediates'] = prev.get('resnet0_intermediates', {})
            out['resnet1_intermediates'] = res1
        return out


class TADSRUNetEntryResnet0Attention0Resnet1Attention1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetEntryResnet0Attention0Resnet1Tester(
            entry_weights_npz_path,
            resnet0_weights_npz_path,
            attention0_weights_npz_path,
            resnet1_weights_npz_path,
            metadata={
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
                'resnet1_metadata': metadata.get('resnet1_metadata', {}),
            },
        )
        self.attention1 = UNetAttention0Transformer2DTester(attention1_weights_npz_path, metadata=metadata.get('attention1_metadata', metadata))

    def run_downblock0_attention1(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_resnet0_attention0_resnet1_attention1(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.prev.run_entry_resnet0_attention0_resnet1(sample, timestep, encoder_hidden_states, return_intermediates=True)
        att1 = self.attention1.run_attention0(prev['resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'resnet0_output': prev['resnet0_output'],
            'attention0_output': prev['attention0_output'],
            'resnet1_output': prev['resnet1_output'],
            'attention1_output': att1['output'],
        }
        if return_intermediates:
            out['attention1_intermediates'] = att1
            out['resnet1_intermediates'] = prev.get('resnet1_intermediates', {})
            out['attention0_intermediates'] = prev.get('attention0_intermediates', {})
            out['resnet0_intermediates'] = prev.get('resnet0_intermediates', {})
        return out
class TADSRUNetDownBlock0DownsamplerTester:
    def __init__(
        self,
        weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        self.weights_path = Path(weights_npz_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f'UNet down_blocks.0 downsampler effective weights missing: {self.weights_path}')
        self.weights = np.load(self.weights_path)
        metadata = metadata or {}
        cfg = metadata.get('downsampler_config', metadata)
        self.prefix = cfg.get('prefix', 'down_blocks_0_downsamplers_0')
        self.padding = int(cfg.get('padding', 1))
        self.stride = int(cfg.get('stride', [2, 2])[0] if isinstance(cfg.get('stride', [2, 2]), list) else cfg.get('stride', 2))
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:40]
            raise KeyError(f'Missing UNet downsampler key {key}; nearby={nearby}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(s) for s in ['conv_weight', 'conv_bias'] if not self.has(s)]
        if missing:
            raise KeyError(f'Missing UNet downsampler weights: {missing}')

    def run_downsampler(self, hidden_states: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            hidden_states,
            self.get('conv_weight'),
            self.get('conv_bias'),
            stride=self.stride,
            padding=self.padding,
        )


class TADSRUNetDownBlock0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.pre_downsampler = TADSRUNetEntryResnet0Attention0Resnet1Attention1Tester(
            entry_weights_npz_path,
            resnet0_weights_npz_path,
            attention0_weights_npz_path,
            resnet1_weights_npz_path,
            attention1_weights_npz_path,
            metadata={
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
                'resnet1_metadata': metadata.get('resnet1_metadata', {}),
                'attention1_metadata': metadata.get('attention1_metadata', {}),
            },
        )
        self.downsampler = TADSRUNetDownBlock0DownsamplerTester(downsampler_weights_npz_path, metadata=metadata.get('downsampler_metadata', metadata))

    def run_downblock0(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.pre_downsampler.run_entry_resnet0_attention0_resnet1_attention1(sample, timestep, encoder_hidden_states, return_intermediates=True)
        down = self.downsampler.run_downsampler(prev['attention1_output'])
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'resnet0_output': prev['resnet0_output'],
            'attention0_output': prev['attention0_output'],
            'resnet1_output': prev['resnet1_output'],
            'attention1_output': prev['attention1_output'],
            'downsampler_output': down,
            'downblock0_output': down,
        }
        if return_intermediates:
            out['attention1_intermediates'] = prev.get('attention1_intermediates', {})
            out['resnet1_intermediates'] = prev.get('resnet1_intermediates', {})
            out['attention0_intermediates'] = prev.get('attention0_intermediates', {})
            out['resnet0_intermediates'] = prev.get('resnet0_intermediates', {})
        return out


class TADSRUNetDownBlock1Resnet0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.downblock0 = TADSRUNetDownBlock0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            metadata=metadata.get('downblock0_metadata', metadata),
        )
        resnet_meta = metadata.get('resnet0_metadata', metadata)
        prefix = resnet_meta.get('resnet0_config', {}).get('prefix', 'down_blocks_1_resnets_0')
        self.resnet0 = UNetResnetBlock2DTester(down1_resnet0_weights_npz_path, prefix, resnet_meta)

    def run_downblock1_resnet0(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet0.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_resnet0(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        down0 = self.downblock0.run_downblock0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        res0 = self.resnet0.run(down0['downblock0_output'], down0['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': down0['conv_in'],
            'time_proj': down0['time_proj'],
            'time_embedding': down0['time_embedding'],
            'downblock0_output': down0['downblock0_output'],
            'downblock1_resnet0_output': res0['output'],
        }
        if return_intermediates:
            out['downblock0_intermediates'] = down0
            out['downblock1_resnet0_intermediates'] = res0
        return out


class TADSRUNetDownBlock1Attention0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock1Resnet0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            metadata={
                'downblock0_metadata': metadata.get('downblock0_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
            },
        )
        self.attention0 = UNetAttention0Transformer2DTester(down1_attention0_weights_npz_path, metadata=metadata.get('attention0_metadata', metadata))

    def run_downblock1_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_resnet0_attention0(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.prev.run_entry_downblock0_downblock1_resnet0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        att0 = self.attention0.run_attention0(prev['downblock1_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_resnet0_output': prev['downblock1_resnet0_output'],
            'downblock1_attention0_output': att0['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock1_attention0_intermediates'] = att0
        return out

class TADSRUNetDownBlock1Resnet1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock1Attention0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            metadata={
                'downblock0_metadata': metadata.get('downblock0_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
            },
        )
        resnet_meta = metadata.get('resnet1_metadata', metadata)
        prefix = resnet_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_1_resnets_1')
        self.resnet1 = UNetResnetBlock2DTester(down1_resnet1_weights_npz_path, prefix, resnet_meta)

    def run_downblock1_resnet1(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet1.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_resnet0_attention0_resnet1(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.prev.run_entry_downblock0_downblock1_resnet0_attention0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        res1 = self.resnet1.run(prev['downblock1_attention0_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_resnet0_output': prev['downblock1_resnet0_output'],
            'downblock1_attention0_output': prev['downblock1_attention0_output'],
            'downblock1_resnet1_output': res1['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock1_resnet1_intermediates'] = res1
        return out


class TADSRUNetDownBlock1Attention1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock1Resnet1Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            metadata={
                'downblock0_metadata': metadata.get('downblock0_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
                'resnet1_metadata': metadata.get('resnet1_metadata', {}),
            },
        )
        self.attention1 = UNetAttention0Transformer2DTester(down1_attention1_weights_npz_path, metadata=metadata.get('attention1_metadata', metadata))

    def run_downblock1_attention1(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_resnet0_attention0_resnet1_attention1(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.prev.run_entry_downblock0_downblock1_resnet0_attention0_resnet1(sample, timestep, encoder_hidden_states, return_intermediates=True)
        att1 = self.attention1.run_attention0(prev['downblock1_resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_resnet0_output': prev['downblock1_resnet0_output'],
            'downblock1_attention0_output': prev['downblock1_attention0_output'],
            'downblock1_resnet1_output': prev['downblock1_resnet1_output'],
            'downblock1_attention1_output': att1['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock1_attention1_intermediates'] = att1
        return out

class TADSRUNetDownBlock1DownsamplerTester:
    def __init__(
        self,
        weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        self.weights_path = Path(weights_npz_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f'UNet down_blocks.1 downsampler effective weights missing: {self.weights_path}')
        self.weights = np.load(self.weights_path)
        metadata = metadata or {}
        cfg = metadata.get('downsampler_config', metadata)
        self.prefix = cfg.get('prefix', 'down_blocks_1_downsamplers_0')
        self.padding = int(cfg.get('padding', 1))
        stride = cfg.get('stride', [2, 2])
        self.stride = int(stride[0] if isinstance(stride, list) else stride)
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:40]
            raise KeyError(f'Missing UNet down_blocks.1 downsampler key {key}; nearby={nearby}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(s) for s in ['conv_weight', 'conv_bias'] if not self.has(s)]
        if missing:
            raise KeyError(f'Missing UNet down_blocks.1 downsampler weights: {missing}')

    def run_downsampler(self, hidden_states: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            hidden_states,
            self.get('conv_weight'),
            self.get('conv_bias'),
            stride=self.stride,
            padding=self.padding,
        )


class TADSRUNetDownBlock1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.downblock0 = TADSRUNetDownBlock0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            metadata=metadata.get('downblock0_metadata', metadata),
        )
        resnet0_meta = metadata.get('resnet0_metadata', metadata)
        attention0_meta = metadata.get('attention0_metadata', metadata)
        resnet1_meta = metadata.get('resnet1_metadata', metadata)
        attention1_meta = metadata.get('attention1_metadata', metadata)
        self.resnet0 = UNetResnetBlock2DTester(
            down1_resnet0_weights_npz_path,
            resnet0_meta.get('resnet0_config', {}).get('prefix', 'down_blocks_1_resnets_0'),
            resnet0_meta,
        )
        self.attention0 = UNetAttention0Transformer2DTester(down1_attention0_weights_npz_path, metadata=attention0_meta)
        self.resnet1 = UNetResnetBlock2DTester(
            down1_resnet1_weights_npz_path,
            resnet1_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_1_resnets_1'),
            resnet1_meta,
        )
        self.attention1 = UNetAttention0Transformer2DTester(down1_attention1_weights_npz_path, metadata=attention1_meta)
        self.downsampler = TADSRUNetDownBlock1DownsamplerTester(
            down1_downsampler_weights_npz_path,
            metadata=metadata.get('downsampler_metadata', metadata),
        )

    def run_downblock1_local(
        self,
        hidden_states: np.ndarray,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_output_states: bool = False,
        return_intermediates: bool = False,
    ):
        res0 = self.resnet0.run(hidden_states, emb, return_intermediates=True)
        att0 = self.attention0.run_attention0(res0['output'], encoder_hidden_states, return_intermediates=True)
        res1 = self.resnet1.run(att0['output'], emb, return_intermediates=True)
        att1 = self.attention1.run_attention0(res1['output'], encoder_hidden_states, return_intermediates=True)
        down = self.downsampler.run_downsampler(att1['output'])
        output_states = [att0['output'], att1['output'], down]
        out = {
            'downblock1_resnet0_output': res0['output'],
            'downblock1_attention0_output': att0['output'],
            'downblock1_resnet1_output': res1['output'],
            'downblock1_attention1_output': att1['output'],
            'downblock1_downsampler_output': down,
            'downblock1_output': down,
        }
        if return_output_states:
            out['output_states'] = output_states
        if return_intermediates:
            out['downblock1_resnet0_intermediates'] = res0
            out['downblock1_attention0_intermediates'] = att0
            out['downblock1_resnet1_intermediates'] = res1
            out['downblock1_attention1_intermediates'] = att1
        return out

    def run_entry_downblock0_downblock1_resnet0_attention0_resnet1_attention1_downsampler(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
        return_output_states: bool = False,
    ):
        down0 = self.downblock0.run_downblock0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        local = self.run_downblock1_local(
            down0['downblock0_output'],
            down0['time_embedding'],
            encoder_hidden_states,
            return_output_states=return_output_states,
            return_intermediates=return_intermediates,
        )
        out = {
            'conv_in': down0['conv_in'],
            'time_proj': down0['time_proj'],
            'time_embedding': down0['time_embedding'],
            'downblock0_output': down0['downblock0_output'],
            **local,
        }
        if return_intermediates:
            out['downblock0_intermediates'] = down0
        return out

class TADSRUNetDownBlock2Resnet0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.downblock1 = TADSRUNetDownBlock1Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            metadata=metadata.get('downblock1_metadata', metadata),
        )
        resnet0_meta = metadata.get('resnet0_metadata', metadata)
        prefix = resnet0_meta.get('resnet0_config', {}).get('prefix', 'down_blocks_2_resnets_0')
        self.resnet0 = UNetResnetBlock2DTester(down2_resnet0_weights_npz_path, prefix, resnet0_meta)

    def run_downblock2_resnet0(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet0.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_downblock2_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.downblock1.run_entry_downblock0_downblock1_resnet0_attention0_resnet1_attention1_downsampler(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
            return_output_states=True,
        )
        res0 = self.resnet0.run(prev['downblock1_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock1_output_states': prev.get('output_states', []),
            'downblock2_resnet0_output': res0['output'],
        }
        if return_intermediates:
            out['downblock1_intermediates'] = prev
            out['downblock2_resnet0_intermediates'] = res0
        return out

class TADSRUNetDownBlock2Attention0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock2Resnet0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            metadata={
                'downblock1_metadata': metadata.get('downblock1_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
            },
        )
        self.attention0 = UNetAttention0Transformer2DTester(down2_attention0_weights_npz_path, metadata=metadata.get('attention0_metadata', metadata))

    def run_downblock2_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_downblock2_resnet0_attention0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.prev.run_entry_downblock0_downblock1_downblock2_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att0 = self.attention0.run_attention0(prev['downblock2_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock1_output_states': prev.get('downblock1_output_states', []),
            'downblock2_resnet0_output': prev['downblock2_resnet0_output'],
            'downblock2_attention0_output': att0['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock2_attention0_intermediates'] = att0
        return out

class TADSRUNetDownBlock2Resnet1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock2Attention0Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            metadata={
                'downblock1_metadata': metadata.get('downblock1_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
            },
        )
        resnet1_meta = metadata.get('resnet1_metadata', metadata)
        prefix = resnet1_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_2_resnets_1')
        self.resnet1 = UNetResnetBlock2DTester(down2_resnet1_weights_npz_path, prefix, resnet1_meta)

    def run_downblock2_resnet1(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet1.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.prev.run_entry_downblock0_downblock1_downblock2_resnet0_attention0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        res1 = self.resnet1.run(prev['downblock2_attention0_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock1_output_states': prev.get('downblock1_output_states', []),
            'downblock2_resnet0_output': prev['downblock2_resnet0_output'],
            'downblock2_attention0_output': prev['downblock2_attention0_output'],
            'downblock2_resnet1_output': res1['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock2_resnet1_intermediates'] = res1
        return out

class TADSRUNetDownBlock2Attention1Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock2Resnet1Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            metadata={
                'downblock1_metadata': metadata.get('downblock1_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
                'resnet1_metadata': metadata.get('resnet1_metadata', {}),
            },
        )
        self.attention1 = UNetAttention0Transformer2DTester(down2_attention1_weights_npz_path, metadata=metadata.get('attention1_metadata', metadata))

    def run_downblock2_attention1(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.prev.run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att1 = self.attention1.run_attention0(prev['downblock2_resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock1_output_states': prev.get('downblock1_output_states', []),
            'downblock2_resnet0_output': prev['downblock2_resnet0_output'],
            'downblock2_attention0_output': prev['downblock2_attention0_output'],
            'downblock2_resnet1_output': prev['downblock2_resnet1_output'],
            'downblock2_attention1_output': att1['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock2_attention1_intermediates'] = att1
        return out

class TADSRUNetDownBlock2DownsamplerTester:
    def __init__(
        self,
        weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        self.weights_path = Path(weights_npz_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f'UNet down_blocks.2 downsampler effective weights missing: {self.weights_path}')
        self.weights = np.load(self.weights_path)
        metadata = metadata or {}
        cfg = metadata.get('downsampler_config', metadata)
        self.prefix = cfg.get('prefix', 'down_blocks_2_downsamplers_0')
        self.padding = int(cfg.get('padding', 1))
        stride = cfg.get('stride', [2, 2])
        self.stride = int(stride[0] if isinstance(stride, list) else stride)
        self._validate()

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:40]
            raise KeyError(f'Missing UNet down_blocks.2 downsampler key {key}; nearby={nearby}')
        return self.weights[key].astype(np.float32)

    def _validate(self) -> None:
        missing = [self.key(s) for s in ['conv_weight', 'conv_bias'] if not self.has(s)]
        if missing:
            raise KeyError(f'Missing UNet down_blocks.2 downsampler weights: {missing}')

    def run_downsampler(self, hidden_states: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            hidden_states,
            self.get('conv_weight'),
            self.get('conv_bias'),
            stride=self.stride,
            padding=self.padding,
        )


class TADSRUNetDownBlock2Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock2Attention1Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            metadata={
                'downblock1_metadata': metadata.get('downblock1_metadata', {}),
                'resnet0_metadata': metadata.get('resnet0_metadata', {}),
                'attention0_metadata': metadata.get('attention0_metadata', {}),
                'resnet1_metadata': metadata.get('resnet1_metadata', {}),
                'attention1_metadata': metadata.get('attention1_metadata', {}),
            },
        )
        self.downsampler = TADSRUNetDownBlock2DownsamplerTester(
            down2_downsampler_weights_npz_path,
            metadata=metadata.get('downsampler_metadata', metadata),
        )
        resnet0_meta = metadata.get('resnet0_metadata', metadata)
        attention0_meta = metadata.get('attention0_metadata', metadata)
        resnet1_meta = metadata.get('resnet1_metadata', metadata)
        attention1_meta = metadata.get('attention1_metadata', metadata)
        self.resnet0 = UNetResnetBlock2DTester(
            down2_resnet0_weights_npz_path,
            resnet0_meta.get('resnet0_config', {}).get('prefix', 'down_blocks_2_resnets_0'),
            resnet0_meta,
        )
        self.attention0 = UNetAttention0Transformer2DTester(down2_attention0_weights_npz_path, metadata=attention0_meta)
        self.resnet1 = UNetResnetBlock2DTester(
            down2_resnet1_weights_npz_path,
            resnet1_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_2_resnets_1'),
            resnet1_meta,
        )
        self.attention1 = UNetAttention0Transformer2DTester(down2_attention1_weights_npz_path, metadata=attention1_meta)

    def run_downblock2_downsampler(self, hidden_states: np.ndarray, return_intermediates: bool = False):
        out = self.downsampler.run_downsampler(hidden_states)
        if return_intermediates:
            return {'output': out}
        return out

    def run_downblock2_local(
        self,
        hidden_states: np.ndarray,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_output_states: bool = False,
        return_intermediates: bool = False,
    ):
        res0 = self.resnet0.run(hidden_states, emb, return_intermediates=True)
        att0 = self.attention0.run_attention0(res0['output'], encoder_hidden_states, return_intermediates=True)
        res1 = self.resnet1.run(att0['output'], emb, return_intermediates=True)
        att1 = self.attention1.run_attention0(res1['output'], encoder_hidden_states, return_intermediates=True)
        down = self.downsampler.run_downsampler(att1['output'])
        output_states = [att0['output'], att1['output'], down]
        out = {
            'downblock2_resnet0_output': res0['output'],
            'downblock2_attention0_output': att0['output'],
            'downblock2_resnet1_output': res1['output'],
            'downblock2_attention1_output': att1['output'],
            'downblock2_downsampler_output': down,
            'downblock2_output': down,
        }
        if return_output_states:
            out['output_states'] = output_states
        if return_intermediates:
            out['downblock2_resnet0_intermediates'] = res0
            out['downblock2_attention0_intermediates'] = att0
            out['downblock2_resnet1_intermediates'] = res1
            out['downblock2_attention1_intermediates'] = att1
        return out

    def run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_downsampler(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
        return_output_states: bool = False,
    ):
        prev = self.prev.run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        down = self.downsampler.run_downsampler(prev['downblock2_attention1_output'])
        output_states = [prev['downblock2_attention0_output'], prev['downblock2_attention1_output'], down]
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock1_output_states': prev.get('downblock1_output_states', []),
            'downblock2_resnet0_output': prev['downblock2_resnet0_output'],
            'downblock2_attention0_output': prev['downblock2_attention0_output'],
            'downblock2_resnet1_output': prev['downblock2_resnet1_output'],
            'downblock2_attention1_output': prev['downblock2_attention1_output'],
            'downblock2_downsampler_output': down,
            'downblock2_output': down,
        }
        if return_output_states:
            out['output_states'] = output_states
        if return_intermediates:
            out['previous_intermediates'] = prev
        return out



class TADSRUNetDownBlock3Resnet0Tester:
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.prev = TADSRUNetDownBlock2Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            metadata=metadata.get('downblock2_metadata', metadata),
        )
        resnet0_meta = metadata.get('resnet0_metadata', metadata)
        prefix = resnet0_meta.get('resnet0_config', {}).get('prefix', 'down_blocks_3_resnets_0')
        self.resnet0 = UNetResnetBlock2DTester(down3_resnet0_weights_npz_path, prefix, resnet0_meta)

    def run_downblock3_resnet0(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet0.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblock0_downblock1_downblock2_downblock3_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.prev.run_entry_downblock0_downblock1_downblock2_resnet0_attention0_resnet1_attention1_downsampler(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
            return_output_states=True,
        )
        res0 = self.resnet0.run(prev['downblock2_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock2_output_states': prev.get('output_states', []),
            'downblock3_resnet0_output': res0['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock3_resnet0_intermediates'] = res0
        return out


class TADSRUNetDownBlock3Resnet1Tester(TADSRUNetDownBlock3Resnet0Tester):
    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        down3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        super().__init__(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            down3_resnet0_weights_npz_path,
            metadata=metadata,
        )
        resnet1_meta = metadata.get('resnet1_metadata', metadata)
        prefix = resnet1_meta.get('resnet1_config', {}).get('prefix', 'down_blocks_3_resnets_1')
        self.resnet1 = UNetResnetBlock2DTester(down3_resnet1_weights_npz_path, prefix, resnet1_meta)

    def run_downblock3_resnet1(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet1.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_downblock3_local(
        self,
        hidden_states: np.ndarray,
        emb: np.ndarray,
        return_output_states: bool = False,
        return_intermediates: bool = False,
    ):
        res0 = self.resnet0.run(hidden_states, emb, return_intermediates=True)
        res1 = self.resnet1.run(res0['output'], emb, return_intermediates=True)
        output_states = [res0['output'], res1['output']]
        out = {
            'downblock3_resnet0_output': res0['output'],
            'downblock3_resnet1_output': res1['output'],
            'downblock3_output': res1['output'],
        }
        if return_output_states:
            out['output_states'] = output_states
        if return_intermediates:
            out['downblock3_resnet0_intermediates'] = res0
            out['downblock3_resnet1_intermediates'] = res1
        return out

    def run_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
        return_output_states: bool = False,
    ):
        prev = self.run_entry_downblock0_downblock1_downblock2_downblock3_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        res1 = self.resnet1.run(prev['downblock3_resnet0_output'], prev['time_embedding'], return_intermediates=True)
        output_states = [prev['downblock3_resnet0_output'], res1['output']]
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock2_output_states': prev.get('downblock2_output_states', []),
            'downblock3_resnet0_output': prev['downblock3_resnet0_output'],
            'downblock3_resnet1_output': res1['output'],
            'downblock3_output': res1['output'],
        }
        if return_output_states:
            out['output_states'] = output_states
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['downblock3_resnet1_intermediates'] = res1
        return out


class TADSRUNetDownBlock3Tester(TADSRUNetDownBlock3Resnet1Tester):
    """Complete local down_blocks.3 tester: resnet0 -> resnet1.

    This remains a block-local tester only. It intentionally does not call
    mid_block, up_blocks or the full UNet forward.
    """


class TADSRUNetMidBlockResnet0Tester:
    """Partial UNet mid_block tester for resnets.0 only.

    This class deliberately stops before mid_block.attentions.0/resnets.1,
    up_blocks, and full UNet forward.
    """

    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        down3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz',
        midblock_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        self.downblock3 = TADSRUNetDownBlock3Tester(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            down3_resnet0_weights_npz_path,
            down3_resnet1_weights_npz_path,
            metadata=metadata.get('downblock3_metadata', metadata),
        )
        resnet0_meta = metadata.get('midblock_resnet0_metadata', metadata.get('resnet0_metadata', metadata))
        prefix = resnet0_meta.get('resnet0_config', {}).get('prefix', 'mid_block_resnets_0')
        self.resnet0 = UNetResnetBlock2DTester(midblock_resnet0_weights_npz_path, prefix, resnet0_meta)

    def run_midblock_resnet0(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet0.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.downblock3.run_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
            return_output_states=True,
        )
        mid0 = self.resnet0.run(prev['downblock3_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock3_output': prev['downblock3_output'],
            'downblock3_output_states': prev.get('output_states', []),
            'midblock_resnet0_output': mid0['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['midblock_resnet0_intermediates'] = mid0
        return out


class TADSRUNetMidBlockAttention0Tester(TADSRUNetMidBlockResnet0Tester):
    """Partial UNet mid_block tester through attentions.0 only.

    This class deliberately stops before mid_block.resnets.1, up_blocks,
    and full UNet forward.
    """

    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        down3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz',
        midblock_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet0_effective_weights.npz',
        midblock_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_attention0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        super().__init__(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            down3_resnet0_weights_npz_path,
            down3_resnet1_weights_npz_path,
            midblock_resnet0_weights_npz_path,
            metadata=metadata,
        )
        att0_meta = metadata.get('midblock_attention0_metadata', metadata.get('attention0_metadata', metadata))
        self.attention0 = UNetAttention0Transformer2DTester(midblock_attention0_weights_npz_path, metadata=att0_meta)

    def run_midblock_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_resnet0_attention0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att0 = self.attention0.run_attention0(prev['midblock_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock3_output': prev['downblock3_output'],
            'midblock_resnet0_output': prev['midblock_resnet0_output'],
            'midblock_attention0_output': att0['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['midblock_attention0_intermediates'] = att0
        return out


class TADSRUNetMidBlockResnet1Tester(TADSRUNetMidBlockAttention0Tester):
    """Local UNet mid_block tester through resnets.1.

    This completes only the local mid_block hidden-state chain:
    resnets.0 -> attentions.0 -> resnets.1. It deliberately stops before
    up_blocks, conv_out, full UNet forward, and full TADSR inference.
    """

    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        down3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz',
        midblock_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet0_effective_weights.npz',
        midblock_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_attention0_effective_weights.npz',
        midblock_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet1_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        super().__init__(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            down3_resnet0_weights_npz_path,
            down3_resnet1_weights_npz_path,
            midblock_resnet0_weights_npz_path,
            midblock_attention0_weights_npz_path,
            metadata=metadata,
        )
        resnet1_meta = metadata.get('midblock_resnet1_metadata', metadata.get('resnet1_metadata', metadata))
        prefix = resnet1_meta.get('resnet1_config', {}).get('prefix', 'mid_block_resnets_1')
        self.resnet1 = UNetResnetBlock2DTester(midblock_resnet1_weights_npz_path, prefix, resnet1_meta)

    def run_midblock_resnet1(self, hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        return self.resnet1.run(hidden_states, emb, return_intermediates=return_intermediates)

    def run_midblock_local(
        self,
        hidden_states: np.ndarray,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        mid0 = self.resnet0.run(hidden_states, emb, return_intermediates=True)
        att0 = self.attention0.run_attention0(mid0['output'], encoder_hidden_states, return_intermediates=True)
        res1 = self.resnet1.run(att0['output'], emb, return_intermediates=True)
        out = {
            'midblock_resnet0_output': mid0['output'],
            'midblock_attention0_output': att0['output'],
            'midblock_resnet1_output': res1['output'],
            'midblock_hidden_output': res1['output'],
        }
        if return_intermediates:
            out['midblock_resnet0_intermediates'] = mid0
            out['midblock_attention0_intermediates'] = att0
            out['midblock_resnet1_intermediates'] = res1
        return out

    def run_entry_downblocks_midblock_resnet0_attention0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_resnet0_attention0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        res1 = self.resnet1.run(prev['midblock_attention0_output'], prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock3_output': prev['downblock3_output'],
            'midblock_resnet0_output': prev['midblock_resnet0_output'],
            'midblock_attention0_output': prev['midblock_attention0_output'],
            'midblock_resnet1_output': res1['output'],
            'midblock_hidden_output': res1['output'],
        }
        if return_intermediates:
            out['previous_intermediates'] = prev
            out['midblock_resnet1_intermediates'] = res1
        return out


class TADSRUNetUpBlock0Resnet0Tester(TADSRUNetMidBlockResnet1Tester):
    """First up-path leaf tester: up_blocks.0.resnets.0 only.

    This wrapper verifies official skip residual consumption and channel
    concatenation before reusing UNetResnetBlock2DTester. It deliberately
    stops before up_blocks.0.resnets.1, upsamplers.0, full UNet forward and
    full TADSR inference.
    """

    def __init__(
        self,
        entry_weights_npz_path: str | Path = DEFAULT_UNET_ENTRY_NPZ,
        down0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet0_effective_weights.npz',
        down0_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention0_effective_weights.npz',
        down0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_resnet1_effective_weights.npz',
        down0_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_attention1_effective_weights.npz',
        down0_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock0_downsampler_effective_weights.npz',
        down1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet0_effective_weights.npz',
        down1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention0_effective_weights.npz',
        down1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_resnet1_effective_weights.npz',
        down1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_attention1_effective_weights.npz',
        down1_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock1_downsampler_effective_weights.npz',
        down2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet0_effective_weights.npz',
        down2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention0_effective_weights.npz',
        down2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_resnet1_effective_weights.npz',
        down2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_attention1_effective_weights.npz',
        down2_downsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock2_downsampler_effective_weights.npz',
        down3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet0_effective_weights.npz',
        down3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_downblock3_resnet1_effective_weights.npz',
        midblock_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet0_effective_weights.npz',
        midblock_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_attention0_effective_weights.npz',
        midblock_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_midblock_resnet1_effective_weights.npz',
        upblock0_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet0_effective_weights.npz',
        metadata: dict | None = None,
    ):
        metadata = metadata or {}
        super().__init__(
            entry_weights_npz_path,
            down0_resnet0_weights_npz_path,
            down0_attention0_weights_npz_path,
            down0_resnet1_weights_npz_path,
            down0_attention1_weights_npz_path,
            down0_downsampler_weights_npz_path,
            down1_resnet0_weights_npz_path,
            down1_attention0_weights_npz_path,
            down1_resnet1_weights_npz_path,
            down1_attention1_weights_npz_path,
            down1_downsampler_weights_npz_path,
            down2_resnet0_weights_npz_path,
            down2_attention0_weights_npz_path,
            down2_resnet1_weights_npz_path,
            down2_attention1_weights_npz_path,
            down2_downsampler_weights_npz_path,
            down3_resnet0_weights_npz_path,
            down3_resnet1_weights_npz_path,
            midblock_resnet0_weights_npz_path,
            midblock_attention0_weights_npz_path,
            midblock_resnet1_weights_npz_path,
            metadata=metadata,
        )
        up_meta = metadata.get('upblock0_resnet0_metadata', metadata.get('resnet0_metadata', metadata))
        prefix = up_meta.get('resnet0_config', {}).get('prefix', 'up_blocks_0_resnets_0')
        self.upblock0_resnet0 = UNetResnetBlock2DTester(upblock0_resnet0_weights_npz_path, prefix, up_meta)
        self.upblock0_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock0_resnet0_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock0 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock0_resnet0(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock0_resnet0_input(hidden_states, res_hidden_states)
        res = self.upblock0_resnet0.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            **res,
        }

    def run_entry_downblocks_midblock_upblock0_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        down = self.downblock3.run_entry_downblock0_downblock1_downblock2_downblock3_resnet0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
            return_output_states=True,
        )
        mid = self.run_midblock_local(
            down['downblock3_output'],
            down['time_embedding'],
            encoder_hidden_states,
            return_intermediates=True,
        )
        downblock3_states = down.get('output_states', [])
        if not downblock3_states:
            raise ValueError('down_blocks.3 output_states are required to consume up_blocks.0.resnets.0 residual')
        res_hidden = np.asarray(downblock3_states[-1], dtype=np.float32)
        up = self.run_upblock0_resnet0(mid['midblock_hidden_output'], res_hidden, down['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': down['conv_in'],
            'time_proj': down['time_proj'],
            'time_embedding': down['time_embedding'],
            'downblock0_output': down['downblock0_output'],
            'downblock1_output': down['downblock1_output'],
            'downblock2_output': down['downblock2_output'],
            'downblock3_output': down['downblock3_output'],
            'midblock_hidden_output': mid['midblock_hidden_output'],
            'upblock0_resnet0_res_hidden': res_hidden,
            'upblock0_resnet0_concat_input': up['concat_input'],
            'upblock0_resnet0_output': up['output'],
        }
        if return_intermediates:
            out['downblocks_intermediates'] = down
            out['midblock_intermediates'] = mid
            out['upblock0_resnet0_intermediates'] = up
        return out


class TADSRUNetUpBlock0Resnet1Tester(TADSRUNetUpBlock0Resnet0Tester):
    """Second up-path leaf tester: up_blocks.0.resnets.1 only."""

    def __init__(
        self,
        *args,
        upblock0_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock0_resnet1_metadata', metadata.get('resnet1_metadata', metadata))
        prefix = up_meta.get('resnet1_config', {}).get('prefix', 'up_blocks_0_resnets_1')
        self.upblock0_resnet1 = UNetResnetBlock2DTester(upblock0_resnet1_weights_npz_path, prefix, up_meta)

    def concat_upblock0_resnet1_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock0 resnet1 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock0_resnet1(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        concat = self.concat_upblock0_resnet1_input(hidden_states, res_hidden_states)
        res = self.upblock0_resnet1.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {'hidden_input': np.asarray(hidden_states, dtype=np.float32), 'res_hidden': np.asarray(res_hidden_states, dtype=np.float32), 'concat_input': concat, **res}

    def run_entry_downblocks_midblock_upblock0_resnet0_resnet1(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.run_entry_downblocks_midblock_upblock0_resnet0(sample, timestep, encoder_hidden_states, return_intermediates=True)
        downblock3_states = prev.get('downblocks_intermediates', {}).get('output_states', [])
        if len(downblock3_states) < 2:
            raise ValueError('down_blocks.3 output_states with at least two residuals are required for up_blocks.0.resnets.1')
        res_hidden = np.asarray(downblock3_states[-2], dtype=np.float32)
        up1 = self.run_upblock0_resnet1(prev['upblock0_resnet0_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock3_output': prev['downblock3_output'],
            'midblock_hidden_output': prev['midblock_hidden_output'],
            'upblock0_resnet0_res_hidden': prev['upblock0_resnet0_res_hidden'],
            'upblock0_resnet0_concat_input': prev['upblock0_resnet0_concat_input'],
            'upblock0_resnet0_output': prev['upblock0_resnet0_output'],
            'upblock0_resnet1_res_hidden': res_hidden,
            'upblock0_resnet1_concat_input': up1['concat_input'],
            'upblock0_resnet1_output': up1['output'],
        }
        if return_intermediates:
            out['downblocks_intermediates'] = prev.get('downblocks_intermediates', {})
            out['midblock_intermediates'] = prev.get('midblock_intermediates', {})
            out['upblock0_resnet0_intermediates'] = prev.get('upblock0_resnet0_intermediates', {})
            out['upblock0_resnet1_intermediates'] = up1
        return out


class TADSRUNetUpBlock0Resnet2Tester(TADSRUNetUpBlock0Resnet1Tester):
    """Third up-path leaf tester: up_blocks.0.resnets.2 only."""

    def __init__(
        self,
        *args,
        upblock0_resnet2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock0_resnet2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock0_resnet2_metadata', metadata.get('resnet2_metadata', metadata))
        prefix = up_meta.get('resnet2_config', {}).get('prefix', 'up_blocks_0_resnets_2')
        self.upblock0_resnet2 = UNetResnetBlock2DTester(upblock0_resnet2_weights_npz_path, prefix, up_meta)

    def concat_upblock0_resnet2_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock0 resnet2 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock0_resnet2(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray, emb: np.ndarray, return_intermediates: bool = False):
        concat = self.concat_upblock0_resnet2_input(hidden_states, res_hidden_states)
        res = self.upblock0_resnet2.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {'hidden_input': np.asarray(hidden_states, dtype=np.float32), 'res_hidden': np.asarray(res_hidden_states, dtype=np.float32), 'concat_input': concat, **res}

    def run_entry_downblocks_midblock_upblock0_resnet0_resnet1_resnet2(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        prev = self.run_entry_downblocks_midblock_upblock0_resnet0_resnet1(sample, timestep, encoder_hidden_states, return_intermediates=True)
        # Diffusers pops residuals in reverse order. After resnets.0/.1 consume
        # down_blocks.3 states, resnets.2 consumes down_blocks.2.output_state_2.
        res_hidden = np.asarray(prev['downblock2_output'], dtype=np.float32)
        up2 = self.run_upblock0_resnet2(prev['upblock0_resnet1_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = {
            'conv_in': prev['conv_in'],
            'time_proj': prev['time_proj'],
            'time_embedding': prev['time_embedding'],
            'downblock0_output': prev['downblock0_output'],
            'downblock1_output': prev['downblock1_output'],
            'downblock2_output': prev['downblock2_output'],
            'downblock3_output': prev['downblock3_output'],
            'midblock_hidden_output': prev['midblock_hidden_output'],
            'upblock0_resnet0_res_hidden': prev['upblock0_resnet0_res_hidden'],
            'upblock0_resnet0_concat_input': prev['upblock0_resnet0_concat_input'],
            'upblock0_resnet0_output': prev['upblock0_resnet0_output'],
            'upblock0_resnet1_res_hidden': prev['upblock0_resnet1_res_hidden'],
            'upblock0_resnet1_concat_input': prev['upblock0_resnet1_concat_input'],
            'upblock0_resnet1_output': prev['upblock0_resnet1_output'],
            'upblock0_resnet2_res_hidden': res_hidden,
            'upblock0_resnet2_concat_input': up2['concat_input'],
            'upblock0_resnet2_output': up2['output'],
        }
        if return_intermediates:
            out['downblocks_intermediates'] = prev.get('downblocks_intermediates', {})
            out['midblock_intermediates'] = prev.get('midblock_intermediates', {})
            out['upblock0_resnet0_intermediates'] = prev.get('upblock0_resnet0_intermediates', {})
            out['upblock0_resnet1_intermediates'] = prev.get('upblock0_resnet1_intermediates', {})
            out['upblock0_resnet2_intermediates'] = up2
        return out


class UNetUpsample2DTester:
    """Static effective-weight tester for diffusers Upsample2D.

    This tester intentionally implements only the deterministic path used by
    TADSR up_blocks.0.upsamplers.0: nearest-neighbor 2x interpolation followed
    by the exported effective 3x3 convolution. Runtime LoRA composition remains
    outside this local checker.
    """

    def __init__(self, weights_npz_path: str | Path, prefix: str = 'up_blocks_0_upsamplers_0', metadata: dict | None = None):
        self.weights_npz_path = Path(weights_npz_path)
        if not self.weights_npz_path.exists():
            raise FileNotFoundError(f'UNet upsampler weights NPZ not found: {self.weights_npz_path}')
        self.weights = np.load(self.weights_npz_path)
        self.prefix = str(prefix)
        self.metadata = metadata or {}
        cfg = self.metadata.get('upsampler_config', self.metadata)
        self.scale_factor = int(cfg.get('scale_factor', 2))
        self.padding = int(cfg.get('conv_padding', cfg.get('padding', 1)))

    def key(self, suffix: str) -> str:
        return f'{self.prefix}_{suffix}'

    def has(self, suffix: str) -> bool:
        return self.key(suffix) in self.weights.files

    def get(self, suffix: str) -> np.ndarray:
        key = self.key(suffix)
        if key not in self.weights.files:
            nearby = [k for k in self.weights.files if self.prefix in k][:80]
            raise KeyError(f'Missing UNet upsampler key: {key}; nearby keys={nearby}')
        return self.weights[key].astype(np.float32)

    def run_interpolation(self, hidden_states: np.ndarray) -> np.ndarray:
        return upsample_nearest2d(np.asarray(hidden_states, dtype=np.float32), self.scale_factor)

    def run_conv(self, interpolated_hidden_states: np.ndarray) -> np.ndarray:
        return conv2d_nchw(
            np.asarray(interpolated_hidden_states, dtype=np.float32),
            self.get('conv_weight'),
            self.get('conv_bias') if self.has('conv_bias') else None,
            stride=1,
            padding=self.padding,
        ).astype(np.float32)

    def run(self, hidden_states: np.ndarray, return_intermediates: bool = False):
        interp = self.run_interpolation(hidden_states)
        conv = self.run_conv(interp)
        if not return_intermediates:
            return conv
        return {
            'input': np.asarray(hidden_states, dtype=np.float32),
            'interpolation': interp,
            'conv': conv,
            'output': conv,
        }


class TADSRUNetUpBlock0UpsamplerTester(TADSRUNetUpBlock0Resnet2Tester):
    """Completes local up_blocks.0 through upsamplers.0 only."""

    def __init__(
        self,
        *args,
        upblock0_upsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock0_upsampler_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock0_upsampler_metadata', metadata.get('upsampler_metadata', metadata))
        prefix = up_meta.get('upsampler_config', {}).get('prefix', 'up_blocks_0_upsamplers_0')
        self.upblock0_upsampler = UNetUpsample2DTester(upblock0_upsampler_weights_npz_path, prefix, up_meta)

    def run_upblock0_upsampler(self, hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.upblock0_upsampler.run(hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_resnets012_upsampler(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_resnet0_resnet1_resnet2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        upsampler = self.run_upblock0_upsampler(prev['upblock0_resnet2_output'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock0_upsampler_input': prev['upblock0_resnet2_output'],
            'upblock0_upsampler_interpolation_output': upsampler['interpolation'],
            'upblock0_upsampler_conv_output': upsampler['conv'],
            'upblock0_upsampler_output': upsampler['output'],
            'upblock0_output_hidden': upsampler['output'],
            'upblock0_output_states': [],
        })
        if return_intermediates:
            out['upblock0_upsampler_intermediates'] = upsampler
        return out

    def run_upblock0_local(self, sample: np.ndarray, timestep, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.run_entry_downblocks_midblock_upblock0_resnets012_upsampler(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=return_intermediates,
        )


class TADSRUNetUpBlock1Resnet0Tester(TADSRUNetUpBlock0UpsamplerTester):
    """First up_blocks.1 leaf tester.

    This wrapper deliberately stops immediately after up_blocks.1.resnets.0.
    It verifies the official residual tuple contract after the already aligned
    local up_blocks.0 output, but does not execute up_blocks.1.attentions.0,
    the rest of up_blocks.1, later up_blocks, or full UNet forward.
    """

    def __init__(
        self,
        *args,
        upblock1_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_resnet0_metadata', metadata.get('resnet0_metadata', metadata))
        prefix = up_meta.get('resnet0_config', {}).get('prefix', 'up_blocks_1_resnets_0')
        self.upblock1_resnet0 = UNetResnetBlock2DTester(upblock1_resnet0_weights_npz_path, prefix, up_meta)
        self.upblock1_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock1_resnet0_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock1 resnet0 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock1_resnet0(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock1_resnet0_input(hidden_states, res_hidden_states)
        res = self.upblock1_resnet0.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            **res,
        }

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_resnets012_upsampler(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        downblock2_states = prev.get('downblocks_intermediates', {}).get('downblock2_output_states', [])
        if len(downblock2_states) < 2:
            raise ValueError('down_blocks.2 output_states with at least two residuals are required for up_blocks.1.resnets.0')
        # Official CrossAttnUpBlock2D slices the remaining accumulated tuple for
        # up_blocks.1 and then pops from the end for resnets.0. After up_blocks.0
        # consumes down_blocks.3 states and down_blocks.2.output_state_2, this is
        # down_blocks.2.output_state_1.
        res_hidden = np.asarray(downblock2_states[-2], dtype=np.float32)
        up = self.run_upblock1_resnet0(prev['upblock0_output_hidden'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_resnet0_hidden_input': prev['upblock0_output_hidden'],
            'upblock1_resnet0_res_hidden': res_hidden,
            'upblock1_resnet0_concat_input': up['concat_input'],
            'upblock1_resnet0_output': up['output'],
        })
        if return_intermediates:
            out['upblock1_resnet0_intermediates'] = up
        return out


class TADSRUNetUpBlock1Attention0Tester(TADSRUNetUpBlock1Resnet0Tester):
    """Partial UNet up_blocks.1 tester through attentions.0 only.

    This wrapper deliberately stops before up_blocks.1.resnets.1,
    attentions.1, resnets.2, attentions.2, upsamplers.0, later up blocks,
    full UNet forward, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock1_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_attention0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_attention0_metadata', metadata.get('attention0_metadata', metadata))
        self.upblock1_attention0 = UNetAttention0Transformer2DTester(upblock1_attention0_weights_npz_path, metadata=up_meta)

    def run_upblock1_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.upblock1_attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att0 = self.run_upblock1_attention0(prev['upblock1_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_attention0_input': prev['upblock1_resnet0_output'],
            'upblock1_attention0_output': att0['output'],
        })
        if return_intermediates:
            out['upblock1_attention0_intermediates'] = att0
        return out


class TADSRUNetUpBlock1Resnet1Tester(TADSRUNetUpBlock1Attention0Tester):
    """Partial UNet up_blocks.1 tester through resnets.1 only.

    This wrapper deliberately stops before up_blocks.1.attentions.1,
    resnets.2, attentions.2, upsamplers.0, later up blocks, full UNet
    forward, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock1_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_resnet1_metadata', metadata.get('resnet1_metadata', metadata))
        prefix = up_meta.get('resnet1_config', {}).get('prefix', 'up_blocks_1_resnets_1')
        self.upblock1_resnet1 = UNetResnetBlock2DTester(upblock1_resnet1_weights_npz_path, prefix, up_meta)
        self.upblock1_resnet1_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock1_resnet1_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock1 resnet1 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock1_resnet1(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock1_resnet1_input(hidden_states, res_hidden_states)
        res = self.upblock1_resnet1.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            **res,
        }

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        downblock2_states = prev.get('downblocks_intermediates', {}).get('downblock2_output_states', [])
        if len(downblock2_states) < 3:
            raise ValueError('down_blocks.2 output_states with at least three residuals are required for up_blocks.1.resnets.1')
        # Official CrossAttnUpBlock2D pops residuals from the local residual
        # tuple. up_blocks.0 consumes down_blocks.3 states and
        # down_blocks.2.output_state_2; resnets.0 consumes
        # down_blocks.2.output_state_1; attentions.0 consumes no residual.
        # Therefore resnets.1 consumes down_blocks.2.output_state_0.
        res_hidden = np.asarray(downblock2_states[-3], dtype=np.float32)
        up = self.run_upblock1_resnet1(prev['upblock1_attention0_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_resnet1_hidden_input': prev['upblock1_attention0_output'],
            'upblock1_resnet1_res_hidden': res_hidden,
            'upblock1_resnet1_concat_input': up['concat_input'],
            'upblock1_resnet1_output': up['output'],
        })
        if return_intermediates:
            out['upblock1_resnet1_intermediates'] = up
        return out


class TADSRUNetUpBlock1Attention1Tester(TADSRUNetUpBlock1Resnet1Tester):
    """Partial UNet up_blocks.1 tester through attentions.1 only.

    This wrapper deliberately stops before up_blocks.1.resnets.2,
    attentions.2, upsamplers.0, later up blocks, full UNet forward, and
    full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock1_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_attention1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_attention1_metadata', metadata.get('attention1_metadata', metadata))
        self.upblock1_attention1 = UNetAttention0Transformer2DTester(upblock1_attention1_weights_npz_path, metadata=up_meta)

    def run_upblock1_attention1(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.upblock1_attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att1 = self.run_upblock1_attention1(prev['upblock1_resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_attention1_input': prev['upblock1_resnet1_output'],
            'upblock1_attention1_output': att1['output'],
        })
        if return_intermediates:
            out['upblock1_attention1_intermediates'] = att1
        return out


class TADSRUNetUpBlock1Resnet2Tester(TADSRUNetUpBlock1Attention1Tester):
    """Partial UNet up_blocks.1 tester through resnets.2 only.

    This wrapper deliberately stops before up_blocks.1.attentions.2,
    upsamplers.0, later up blocks, full UNet forward, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock1_resnet2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_resnet2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_resnet2_metadata', metadata.get('resnet2_metadata', metadata))
        prefix = up_meta.get('resnet2_config', {}).get('prefix', 'up_blocks_1_resnets_2')
        self.upblock1_resnet2 = UNetResnetBlock2DTester(upblock1_resnet2_weights_npz_path, prefix, up_meta)
        self.upblock1_resnet2_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock1_resnet2_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4:
            raise ValueError(f'Expected NCHW hidden/residual, got {hidden.shape} and {residual.shape}')
        if hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock1 resnet2 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock1_resnet2(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock1_resnet2_input(hidden_states, res_hidden_states)
        res = self.upblock1_resnet2.run(concat, emb, return_intermediates=True)
        if not return_intermediates:
            return res['output']
        return {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            **res,
        }

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        down_meta = prev.get('downblocks_intermediates', {})
        downblock1_states = down_meta.get('downblock1_output_states', [])
        cursor = down_meta
        while len(downblock1_states) < 3 and isinstance(cursor, dict):
            cursor = cursor.get('previous_intermediates')
            if isinstance(cursor, dict):
                downblock1_states = cursor.get('downblock1_output_states', [])
        if len(downblock1_states) < 3:
            raise ValueError('down_blocks.1 output_states with at least three residuals are required for up_blocks.1.resnets.2')
        # Official CrossAttnUpBlock2D pops residuals from the local residual
        # tuple. up_blocks.0 consumes down_blocks.3 states and
        # down_blocks.2.output_state_2; resnets.0 consumes
        # down_blocks.2.output_state_1; attentions.0 consumes no residual;
        # resnets.1 consumes down_blocks.2.output_state_0; attentions.1
        # consumes no residual. Therefore resnets.2 consumes
        # down_blocks.1.output_state_2.
        res_hidden = np.asarray(downblock1_states[-1], dtype=np.float32)
        up = self.run_upblock1_resnet2(prev['upblock1_attention1_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_resnet2_hidden_input': prev['upblock1_attention1_output'],
            'upblock1_resnet2_res_hidden': res_hidden,
            'upblock1_resnet2_concat_input': up['concat_input'],
            'upblock1_resnet2_output': up['output'],
        })
        if return_intermediates:
            out['upblock1_resnet2_intermediates'] = up
        return out


class TADSRUNetUpBlock1Attention2Tester(TADSRUNetUpBlock1Resnet2Tester):
    """Partial UNet up_blocks.1 tester through attentions.2 only.

    This wrapper deliberately stops before up_blocks.1.upsamplers.0,
    later up blocks, full UNet forward, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock1_attention2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_attention2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_attention2_metadata', metadata.get('attention2_metadata', metadata))
        self.upblock1_attention2 = UNetAttention0Transformer2DTester(upblock1_attention2_weights_npz_path, metadata=up_meta)

    def run_upblock1_attention2(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        return self.upblock1_attention2.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2_attention2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att2 = self.run_upblock1_attention2(prev['upblock1_resnet2_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_attention2_input': prev['upblock1_resnet2_output'],
            'upblock1_attention2_output': att2['output'],
        })
        if return_intermediates:
            out['upblock1_attention2_intermediates'] = att2
        return out


class TADSRUNetUpBlock1UpsamplerTester(TADSRUNetUpBlock1Attention2Tester):
    """Partial UNet up_blocks.1 tester through upsamplers.0 only.

    This wrapper appends the final up_blocks.1 upsampler after the already
    aligned resnet/attention chain. It deliberately stops before up_blocks.2,
    full up_blocks.1 aggregate comparison, full UNet forward, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock1_upsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock1_upsampler_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock1_upsampler_metadata', metadata.get('upsampler_metadata', metadata))
        prefix = up_meta.get('upsampler_config', {}).get('prefix', 'up_blocks_1_upsamplers_0')
        self.upblock1_upsampler = UNetUpsample2DTester(upblock1_upsampler_weights_npz_path, prefix, up_meta)

    def run_upblock1_upsampler(
        self,
        hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        if upsample_size is not None:
            raise NotImplementedError('up_blocks.1.upsamplers.0 non-None upsample_size is not supported by this local tester yet.')
        return self.upblock1_upsampler.run(hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_upblock1_through_upsampler(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_resnet0_attention0_resnet1_attention1_resnet2_attention2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        up = self.run_upblock1_upsampler(prev['upblock1_attention2_output'], upsample_size=upsample_size, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock1_upsampler_input': prev['upblock1_attention2_output'],
            'upblock1_upsampler_interpolation_output': up['interpolation'],
            'upblock1_upsampler_conv_output': up['conv'],
            'upblock1_upsampler_output': up['output'],
        })
        if return_intermediates:
            out['upblock1_upsampler_intermediates'] = up
        return out


class TADSRUNetUpBlock1LocalTester(TADSRUNetUpBlock1UpsamplerTester):
    """Complete local up_blocks.1 aggregate tester.

    This class verifies the local aggregate chain
    resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 -> resnets.2
    -> attentions.2 -> upsamplers.0. It deliberately stops before
    up_blocks.2, all-up-block aggregate checks, full UNet forward, and full
    TADSR inference.
    """

    @staticmethod
    def _find_downblock_states(intermediates: dict, key: str):
        cursor = intermediates
        while isinstance(cursor, dict):
            states = cursor.get(key, [])
            if states:
                return states
            cursor = cursor.get('previous_intermediates')
        return []

    def run_upblock1_local(
        self,
        hidden_states: np.ndarray,
        residual_samples,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        residual_tuple = tuple(np.asarray(x, dtype=np.float32) for x in residual_samples)
        if len(residual_tuple) != 3:
            raise ValueError(f'up_blocks.1 local aggregate requires exactly 3 residual samples, got {len(residual_tuple)}')

        res0_hidden = residual_tuple[-1]
        remaining_after_resnet0 = residual_tuple[:-1]
        res0 = self.run_upblock1_resnet0(hidden_states, res0_hidden, emb, return_intermediates=True)
        att0 = self.run_upblock1_attention0(res0['output'], encoder_hidden_states, return_intermediates=True)

        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        res1 = self.run_upblock1_resnet1(att0['output'], res1_hidden, emb, return_intermediates=True)
        att1 = self.run_upblock1_attention1(res1['output'], encoder_hidden_states, return_intermediates=True)

        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        res2 = self.run_upblock1_resnet2(att1['output'], res2_hidden, emb, return_intermediates=True)
        att2 = self.run_upblock1_attention2(res2['output'], encoder_hidden_states, return_intermediates=True)
        upsampler = self.run_upblock1_upsampler(att2['output'], upsample_size=None, return_intermediates=True)

        out = {
            'input_hidden_states': np.asarray(hidden_states, dtype=np.float32),
            'input_residual_samples': residual_tuple,
            'upblock1_resnet0_res_hidden': res0_hidden,
            'upblock1_resnet0_output': res0['output'],
            'upblock1_attention0_output': att0['output'],
            'upblock1_resnet1_res_hidden': res1_hidden,
            'upblock1_resnet1_output': res1['output'],
            'upblock1_attention1_output': att1['output'],
            'upblock1_resnet2_res_hidden': res2_hidden,
            'upblock1_resnet2_output': res2['output'],
            'upblock1_attention2_output': att2['output'],
            'upblock1_upsampler_interpolation_output': upsampler['interpolation'],
            'upblock1_upsampler_conv_output': upsampler['conv'],
            'upblock1_upsampler_output': upsampler['output'],
            'upblock1_output_hidden': upsampler['output'],
            'upblock1_output_states': [],
            'remaining_internal_residual_samples_after_upblock1': remaining_after_resnet2,
        }
        if return_intermediates:
            out.update({
                'upblock1_resnet0_intermediates': res0,
                'upblock1_attention0_intermediates': att0,
                'upblock1_resnet1_intermediates': res1,
                'upblock1_attention1_intermediates': att1,
                'upblock1_resnet2_intermediates': res2,
                'upblock1_attention2_intermediates': att2,
                'upblock1_upsampler_intermediates': upsampler,
            })
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_local(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_resnets012_upsampler(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        down_meta = prev.get('downblocks_intermediates', {})
        downblock2_states = self._find_downblock_states(down_meta, 'downblock2_output_states')
        downblock1_states = self._find_downblock_states(down_meta, 'downblock1_output_states')
        if len(downblock2_states) < 3 or len(downblock1_states) < 3:
            raise ValueError('down_blocks.1/2 output states are required to reconstruct the official up_blocks.1 residual tuple')
        residual_tuple = (
            np.asarray(downblock1_states[-1], dtype=np.float32),
            np.asarray(downblock2_states[-3], dtype=np.float32),
            np.asarray(downblock2_states[-2], dtype=np.float32),
        )
        local = self.run_upblock1_local(
            prev['upblock0_output_hidden'],
            residual_tuple,
            prev['time_embedding'],
            encoder_hidden_states,
            return_intermediates=True,
        )
        out = dict(prev)
        out.update(local)
        out.update({
            'upblock1_local_residual_tuple': residual_tuple,
            'upblock1_output_hidden': local['upblock1_output_hidden'],
            'upblock1_output_states': [],
        })
        if return_intermediates:
            out['upblock1_local_intermediates'] = local
        return out


class TADSRUNetUpBlock2Resnet0Tester(TADSRUNetUpBlock1LocalTester):
    """Minimal tester for only up_blocks.2.resnets.0.

    This class deliberately stops immediately after up_blocks.2.resnets.0.
    It does not execute up_blocks.2.attentions.0, the rest of up_blocks.2,
    up_blocks.3, full UNet forward, generic runtime LoRA, or full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock2_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock2_resnet0_metadata', metadata.get('resnet0_metadata', metadata))
        prefix = up_meta.get('resnet0_config', {}).get('prefix', 'up_blocks_2_resnets_0')
        self.upblock2_resnet0 = UNetResnetBlock2DTester(upblock2_resnet0_weights_npz_path, prefix, up_meta)
        self.upblock2_resnet0_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock2_resnet0_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock2 resnet0 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock2_resnet0(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock2_resnet0_input(hidden_states, res_hidden_states)
        res = self.upblock2_resnet0.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_local(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        down_meta = prev.get('downblocks_intermediates', {})
        downblock0_states = self._find_downblock_states(down_meta, 'downblock0_output_states')
        downblock1_states = self._find_downblock_states(down_meta, 'downblock1_output_states')
        if len(downblock0_states) < 3:
            # Older downblock0 local testers did not expose an
            # output_states list, but their intermediate dict contains the
            # exact official residual outputs in order: attention0,
            # attention1, then downsampler. Reconstructing this tuple keeps
            # the bridge contract faithful without executing any later
            # up_blocks.2 modules.
            def _find_nested_dict(obj, key: str):
                if isinstance(obj, dict):
                    if key in obj and isinstance(obj[key], dict):
                        return obj[key]
                    for value in obj.values():
                        found = _find_nested_dict(value, key)
                        if found is not None:
                            return found
                return None

            downblock0_inter = _find_nested_dict(down_meta, 'downblock0_intermediates')
            if downblock0_inter is not None:
                candidates = [
                    downblock0_inter.get('attention0_output'),
                    downblock0_inter.get('attention1_output'),
                    downblock0_inter.get('downsampler_output'),
                ]
                if all(x is not None for x in candidates):
                    downblock0_states = [np.asarray(x, dtype=np.float32) for x in candidates]
        if len(downblock0_states) < 3 or len(downblock1_states) < 2:
            raise ValueError('down_blocks.0/1 output states are required to reconstruct the official up_blocks.2 residual tuple')
        residual_tuple = (
            np.asarray(downblock0_states[-1], dtype=np.float32),
            np.asarray(downblock1_states[-3], dtype=np.float32),
            np.asarray(downblock1_states[-2], dtype=np.float32),
        )
        res_hidden = residual_tuple[-1]
        remaining_after_resnet0 = residual_tuple[:-1]
        up = self.run_upblock2_resnet0(prev['upblock1_output_hidden'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_local_residual_tuple': residual_tuple,
            'upblock2_resnet0_hidden_input': prev['upblock1_output_hidden'],
            'upblock2_resnet0_res_hidden': res_hidden,
            'upblock2_resnet0_concat_input': up['concat_input'],
            'upblock2_resnet0_output': up['output'],
            'remaining_internal_residual_samples_after_upblock2_resnet0': remaining_after_resnet0,
        })
        if return_intermediates:
            out['upblock2_resnet0_intermediates'] = up
        return out


class TADSRUNetUpBlock2Attention0Tester(TADSRUNetUpBlock2Resnet0Tester):
    """Minimal tester for only up_blocks.2.attentions.0.

    The entry bridge executes only the already aligned path:
    entry -> down_blocks -> mid_block -> full local up_blocks.0 ->
    full local up_blocks.1 -> up_blocks.2.resnets.0 ->
    up_blocks.2.attentions.0.

    It deliberately stops before up_blocks.2.resnets.1, the rest of
    up_blocks.2, up_blocks.3, full UNet forward, runtime LoRA, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock2_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_attention0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock2_attention0_metadata', metadata.get('attention0_metadata', metadata))
        self.upblock2_attention0 = UNetAttention0Transformer2DTester(upblock2_attention0_weights_npz_path, metadata=att_meta)

    def run_upblock2_attention0(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        att = self.upblock2_attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        att = self.run_upblock2_attention0(prev['upblock2_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_attention0_input': prev['upblock2_resnet0_output'],
            'upblock2_attention0_output': att['output'],
            'upblock2_output_hidden_after_attention0': att['output'],
        })
        if return_intermediates:
            out['upblock2_attention0_intermediates'] = att
        return out


class TADSRUNetUpBlock2Resnet1Tester(TADSRUNetUpBlock2Attention0Tester):
    """Minimal tester for only up_blocks.2.resnets.1.

    The bridge executes only the already aligned path:
    entry -> down_blocks -> mid_block -> full local up_blocks.0 ->
    full local up_blocks.1 -> up_blocks.2.resnets.0 ->
    up_blocks.2.attentions.0 -> up_blocks.2.resnets.1.

    It deliberately stops before up_blocks.2.attentions.1, the rest of
    up_blocks.2, up_blocks.3, full UNet forward, runtime LoRA, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock2_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock2_resnet1_metadata', metadata.get('resnet1_metadata', metadata))
        prefix = up_meta.get('resnet1_config', {}).get('prefix', 'up_blocks_2_resnets_1')
        self.upblock2_resnet1 = UNetResnetBlock2DTester(upblock2_resnet1_weights_npz_path, prefix, up_meta)
        self.upblock2_resnet1_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock2_resnet1_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock2 resnet1 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock2_resnet1(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock2_resnet1_input(hidden_states, res_hidden_states)
        res = self.upblock2_resnet1.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_attention0 = prev.get('remaining_internal_residual_samples_after_upblock2_resnet0')
        if remaining_after_attention0 is None or len(remaining_after_attention0) < 1:
            raise ValueError('up_blocks.2.resnets.1 requires the residual tuple left after up_blocks.2.resnets.0; attention0 must not consume it')
        res_hidden = np.asarray(remaining_after_attention0[-1], dtype=np.float32)
        remaining_after_resnet1 = tuple(remaining_after_attention0[:-1])
        up = self.run_upblock2_resnet1(prev['upblock2_attention0_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_resnet1_hidden_input': prev['upblock2_attention0_output'],
            'upblock2_resnet1_res_hidden': res_hidden,
            'upblock2_resnet1_concat_input': up['concat_input'],
            'upblock2_resnet1_output': up['output'],
            'upblock2_output_hidden_after_resnet1': up['output'],
            'remaining_internal_residual_samples_after_upblock2_resnet1': remaining_after_resnet1,
        })
        if return_intermediates:
            out['upblock2_resnet1_intermediates'] = up
        return out


class TADSRUNetUpBlock2Attention1Tester(TADSRUNetUpBlock2Resnet1Tester):
    """Minimal tester for only up_blocks.2.attentions.1.

    The bridge executes only the already aligned path:
    entry -> down_blocks -> mid_block -> full local up_blocks.0 ->
    full local up_blocks.1 -> up_blocks.2.resnets.0 ->
    up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 ->
    up_blocks.2.attentions.1.

    It deliberately stops before up_blocks.2.resnets.2, the rest of
    up_blocks.2, up_blocks.3, full UNet forward, runtime LoRA, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock2_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_attention1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock2_attention1_metadata', metadata.get('attention1_metadata', metadata))
        self.upblock2_attention1 = UNetAttention0Transformer2DTester(upblock2_attention1_weights_npz_path, metadata=att_meta)

    def run_upblock2_attention1(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        att = self.upblock2_attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_resnet1 = prev.get('remaining_internal_residual_samples_after_upblock2_resnet1')
        if remaining_after_resnet1 is None:
            raise ValueError('up_blocks.2.attentions.1 expects the residual tuple left after up_blocks.2.resnets.1; attention1 must not consume it')
        att = self.run_upblock2_attention1(prev['upblock2_resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_attention1_input': prev['upblock2_resnet1_output'],
            'upblock2_attention1_output': att['output'],
            'upblock2_output_hidden_after_attention1': att['output'],
            'remaining_internal_residual_samples_after_upblock2_attention1': remaining_after_resnet1,
        })
        if return_intermediates:
            out['upblock2_attention1_intermediates'] = att
        return out


class TADSRUNetUpBlock2Resnet2Tester(TADSRUNetUpBlock2Attention1Tester):
    """Minimal tester for only up_blocks.2.resnets.2.

    The bridge executes only the already aligned path:
    entry -> down_blocks -> mid_block -> full local up_blocks.0 ->
    full local up_blocks.1 -> up_blocks.2.resnets.0 ->
    up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 ->
    up_blocks.2.attentions.1 -> up_blocks.2.resnets.2.

    It deliberately stops before up_blocks.2.attentions.2, the rest of
    up_blocks.2, up_blocks.3, full UNet forward, runtime LoRA, and full TADSR
    inference.
    """

    def __init__(
        self,
        *args,
        upblock2_resnet2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_resnet2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock2_resnet2_metadata', metadata.get('resnet2_metadata', metadata))
        prefix = up_meta.get('resnet2_config', {}).get('prefix', 'up_blocks_2_resnets_2')
        self.upblock2_resnet2 = UNetResnetBlock2DTester(upblock2_resnet2_weights_npz_path, prefix, up_meta)
        self.upblock2_resnet2_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock2_resnet2_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock2 resnet2 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock2_resnet2(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock2_resnet2_input(hidden_states, res_hidden_states)
        res = self.upblock2_resnet2.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_attention1 = prev.get('remaining_internal_residual_samples_after_upblock2_attention1')
        if remaining_after_attention1 is None or len(remaining_after_attention1) < 1:
            raise ValueError('up_blocks.2.resnets.2 requires the residual tuple left after up_blocks.2.attentions.1; attention1 must not consume it')
        res_hidden = np.asarray(remaining_after_attention1[-1], dtype=np.float32)
        remaining_after_resnet2 = tuple(remaining_after_attention1[:-1])
        up = self.run_upblock2_resnet2(prev['upblock2_attention1_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_resnet2_hidden_input': prev['upblock2_attention1_output'],
            'upblock2_resnet2_res_hidden': res_hidden,
            'upblock2_resnet2_concat_input': up['concat_input'],
            'upblock2_resnet2_output': up['output'],
            'upblock2_output_hidden_after_resnet2': up['output'],
            'remaining_internal_residual_samples_after_upblock2_resnet2': remaining_after_resnet2,
        })
        if return_intermediates:
            out['upblock2_resnet2_intermediates'] = up
        return out


class TADSRUNetUpBlock2Attention2Tester(TADSRUNetUpBlock2Resnet2Tester):
    """Minimal tester for only up_blocks.2.attentions.2.

    The bridge executes only the already aligned path:
    entry -> down_blocks -> mid_block -> full local up_blocks.0 ->
    full local up_blocks.1 -> up_blocks.2.resnets.0 ->
    up_blocks.2.attentions.0 -> up_blocks.2.resnets.1 ->
    up_blocks.2.attentions.1 -> up_blocks.2.resnets.2 ->
    up_blocks.2.attentions.2.

    It deliberately stops before up_blocks.2.upsamplers.0, up_blocks.3,
    full UNet forward, runtime LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock2_attention2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_attention2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock2_attention2_metadata', metadata.get('attention2_metadata', metadata))
        self.upblock2_attention2 = UNetAttention0Transformer2DTester(upblock2_attention2_weights_npz_path, metadata=att_meta)

    def run_upblock2_attention2(self, hidden_states: np.ndarray, encoder_hidden_states: np.ndarray, return_intermediates: bool = False):
        att = self.upblock2_attention2.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2_attention2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_resnet2 = prev.get('remaining_internal_residual_samples_after_upblock2_resnet2')
        if remaining_after_resnet2 is None:
            raise ValueError('up_blocks.2.attentions.2 expects the residual tuple left after up_blocks.2.resnets.2; attention2 must not consume it')
        att = self.run_upblock2_attention2(prev['upblock2_resnet2_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_attention2_input': prev['upblock2_resnet2_output'],
            'upblock2_attention2_output': att['output'],
            'upblock2_output_hidden_after_attention2': att['output'],
            'remaining_internal_residual_samples_after_upblock2_attention2': remaining_after_resnet2,
        })
        if return_intermediates:
            out['upblock2_attention2_intermediates'] = att
        return out


class TADSRUNetUpBlock2UpsamplerTester(TADSRUNetUpBlock2Attention2Tester):
    """Partial UNet up_blocks.2 tester through upsamplers.0 only.

    This wrapper appends only up_blocks.2.upsamplers.0 after the already
    aligned local chain through up_blocks.2.attentions.2. It deliberately
    stops before full up_blocks.2 aggregate verification, up_blocks.3, full
    UNet forward, runtime LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock2_upsampler_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock2_upsampler_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock2_upsampler_metadata', metadata.get('upsampler_metadata', metadata))
        prefix = up_meta.get('upsampler_config', {}).get('prefix', 'up_blocks_2_upsamplers_0')
        self.upblock2_upsampler = UNetUpsample2DTester(upblock2_upsampler_weights_npz_path, prefix, up_meta)

    def run_upblock2_upsampler(
        self,
        hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        if upsample_size is not None:
            raise NotImplementedError('up_blocks.2.upsamplers.0 non-None upsample_size is not supported by this local tester yet.')
        return self.upblock2_upsampler.run(hidden_states, return_intermediates=return_intermediates)

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_through_upsampler(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_resnet0_attention0_resnet1_attention1_resnet2_attention2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_attention2 = prev.get('remaining_internal_residual_samples_after_upblock2_attention2')
        if remaining_after_attention2 is None:
            raise ValueError('up_blocks.2.upsamplers.0 expects the residual tuple left after up_blocks.2.attentions.2; upsampler must not consume it')
        up = self.run_upblock2_upsampler(prev['upblock2_attention2_output'], upsample_size=upsample_size, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock2_upsampler_input': prev['upblock2_attention2_output'],
            'upblock2_upsampler_interpolation_output': up['interpolation'],
            'upblock2_upsampler_conv_output': up['conv'],
            'upblock2_upsampler_output': up['output'],
            'upblock2_output_hidden_after_upsampler': up['output'],
            'remaining_internal_residual_samples_after_upblock2_upsampler': remaining_after_attention2,
        })
        if return_intermediates:
            out['upblock2_upsampler_intermediates'] = up
        return out


class TADSRUNetUpBlock2LocalTester(TADSRUNetUpBlock2UpsamplerTester):
    """Local aggregate tester for TADSR UNet up_blocks.2 only.

    The aggregate exactly follows the official local order:
    resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 ->
    resnets.2 -> attentions.2 -> upsamplers.0.

    It deliberately stops before up_blocks.3, full up_blocks traversal,
    full UNet forward, runtime LoRA integration, and full TADSR inference.
    """

    def run_upblock2_local(
        self,
        hidden_states: np.ndarray,
        residual_samples,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        residual_tuple = tuple(np.asarray(x, dtype=np.float32) for x in residual_samples)
        if len(residual_tuple) != 3:
            raise ValueError(f'up_blocks.2 local tester expects exactly 3 residual samples, got {len(residual_tuple)}')

        res0_hidden = residual_tuple[-1]
        remaining_after_resnet0 = residual_tuple[:-1]
        res0 = self.run_upblock2_resnet0(hidden_states, res0_hidden, emb, return_intermediates=True)
        att0 = self.run_upblock2_attention0(res0['output'], encoder_hidden_states, return_intermediates=True)

        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        res1 = self.run_upblock2_resnet1(att0['output'], res1_hidden, emb, return_intermediates=True)
        att1 = self.run_upblock2_attention1(res1['output'], encoder_hidden_states, return_intermediates=True)

        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        res2 = self.run_upblock2_resnet2(att1['output'], res2_hidden, emb, return_intermediates=True)
        att2 = self.run_upblock2_attention2(res2['output'], encoder_hidden_states, return_intermediates=True)

        up = self.run_upblock2_upsampler(att2['output'], upsample_size=upsample_size, return_intermediates=True)
        out = {
            'input_hidden_states': np.asarray(hidden_states, dtype=np.float32),
            'input_residual_samples': residual_tuple,
            'upblock2_resnet0_res_hidden': res0_hidden,
            'upblock2_resnet0_output': res0['output'],
            'upblock2_attention0_output': att0['output'],
            'upblock2_resnet1_res_hidden': res1_hidden,
            'upblock2_resnet1_output': res1['output'],
            'upblock2_attention1_output': att1['output'],
            'upblock2_resnet2_res_hidden': res2_hidden,
            'upblock2_resnet2_output': res2['output'],
            'upblock2_attention2_output': att2['output'],
            'upblock2_upsampler_interpolation_output': up['interpolation'],
            'upblock2_upsampler_conv_output': up['conv'],
            'upblock2_upsampler_output': up['output'],
            'upblock2_output_hidden': up['output'],
            'upblock2_output_states': [],
            'remaining_internal_residual_samples_after_upblock2': remaining_after_resnet2,
        }
        if return_intermediates:
            out.update({
                'upblock2_resnet0_intermediates': res0,
                'upblock2_attention0_intermediates': att0,
                'upblock2_resnet1_intermediates': res1,
                'upblock2_attention1_intermediates': att1,
                'upblock2_resnet2_intermediates': res2,
                'upblock2_attention2_intermediates': att2,
                'upblock2_upsampler_intermediates': up,
            })
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_local(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        upsample_size=None,
        return_intermediates: bool = False,
    ):
        out = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_through_upsampler(
            sample,
            timestep,
            encoder_hidden_states,
            upsample_size=upsample_size,
            return_intermediates=True,
        )
        out['upblock2_output_hidden'] = out['upblock2_upsampler_output']
        out['upblock2_output_states'] = []
        out['remaining_internal_residual_samples_after_upblock2'] = out[
            'remaining_internal_residual_samples_after_upblock2_upsampler'
        ]
        if not return_intermediates:
            for key in list(out.keys()):
                if key.endswith('_intermediates'):
                    out.pop(key, None)
        return out


class TADSRUNetUpBlock3Resnet0Tester(TADSRUNetUpBlock2LocalTester):
    """Minimal tester for only up_blocks.3.resnets.0.

    The bridge executes the already aligned chain:
    entry -> all down_blocks -> full local mid_block -> full local
    up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 ->
    up_blocks.3.resnets.0.

    It deliberately stops before up_blocks.3.attentions.0, the rest of
    up_blocks.3, all-up-block aggregate checks, full UNet forward, runtime
    LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_resnet0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock3_resnet0_metadata', metadata.get('resnet0_metadata', metadata))
        prefix = up_meta.get('resnet0_config', {}).get('prefix', 'up_blocks_3_resnets_0')
        self.upblock3_resnet0 = UNetResnetBlock2DTester(upblock3_resnet0_weights_npz_path, prefix, up_meta)
        self.upblock3_resnet0_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock3_resnet0_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock3 resnet0 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock3_resnet0(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock3_resnet0_input(hidden_states, res_hidden_states)
        res = self.upblock3_resnet0.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_local(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        down_meta = prev.get('downblocks_intermediates', {})
        downblock0_states = self._find_downblock_states(down_meta, 'downblock0_output_states')
        if len(downblock0_states) < 2:
            def _find_nested_dict(obj, key: str):
                if isinstance(obj, dict):
                    if key in obj and isinstance(obj[key], dict):
                        return obj[key]
                    for value in obj.values():
                        found = _find_nested_dict(value, key)
                        if found is not None:
                            return found
                return None

            downblock0_inter = _find_nested_dict(down_meta, 'downblock0_intermediates')
            if downblock0_inter is not None:
                candidates = [
                    downblock0_inter.get('attention0_output'),
                    downblock0_inter.get('attention1_output'),
                    downblock0_inter.get('downsampler_output'),
                ]
                if all(x is not None for x in candidates):
                    downblock0_states = [np.asarray(x, dtype=np.float32) for x in candidates]
        if len(downblock0_states) < 2:
            raise ValueError('conv_in plus down_blocks.0 output_state_0/1 are required to reconstruct the official up_blocks.3 residual tuple')
        remaining_after_upblock2_external = (
            np.asarray(prev['conv_in'], dtype=np.float32),
            np.asarray(downblock0_states[0], dtype=np.float32),
            np.asarray(downblock0_states[1], dtype=np.float32),
        )
        res_hidden = np.asarray(remaining_after_upblock2_external[-1], dtype=np.float32)
        remaining_after_resnet0 = tuple(remaining_after_upblock2_external[:-1])
        up = self.run_upblock3_resnet0(prev['upblock2_output_hidden'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'remaining_external_residual_samples_after_upblock2': remaining_after_upblock2_external,
            'upblock3_local_residual_tuple': remaining_after_upblock2_external,
            'upblock3_resnet0_hidden_input': prev['upblock2_output_hidden'],
            'upblock3_resnet0_res_hidden': res_hidden,
            'upblock3_resnet0_concat_input': up['concat_input'],
            'upblock3_resnet0_output': up['output'],
            'upblock3_output_hidden_after_resnet0': up['output'],
            'remaining_internal_residual_samples_after_upblock3_resnet0': remaining_after_resnet0,
        })
        if return_intermediates:
            out['upblock3_resnet0_intermediates'] = up
        return out


class TADSRUNetUpBlock3Attention0Tester(TADSRUNetUpBlock3Resnet0Tester):
    """Minimal tester for only up_blocks.3.attentions.0.

    The bridge executes the already aligned chain:
    entry -> all down_blocks -> full local mid_block -> full local
    up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 ->
    up_blocks.3.resnets.0 -> up_blocks.3.attentions.0.

    It deliberately stops before up_blocks.3.resnets.1, the rest of
    up_blocks.3, all-up-block aggregate checks, full UNet forward, runtime
    LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_attention0_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_attention0_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock3_attention0_metadata', metadata.get('attention0_metadata', metadata))
        self.upblock3_attention0 = UNetAttention0Transformer2DTester(upblock3_attention0_weights_npz_path, metadata=att_meta)

    def run_upblock3_attention0(
        self,
        hidden_states: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        att = self.upblock3_attention0.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_resnet0 = prev.get('remaining_internal_residual_samples_after_upblock3_resnet0')
        if remaining_after_resnet0 is None:
            raise ValueError('up_blocks.3.attentions.0 expects the residual tuple left after up_blocks.3.resnets.0; attention0 must not consume it')
        att = self.run_upblock3_attention0(prev['upblock3_resnet0_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock3_attention0_input': prev['upblock3_resnet0_output'],
            'upblock3_attention0_output': att['output'],
            'upblock3_output_hidden_after_attention0': att['output'],
            'remaining_internal_residual_samples_after_upblock3_attention0': remaining_after_resnet0,
        })
        if return_intermediates:
            out['upblock3_attention0_intermediates'] = att
        return out


class TADSRUNetUpBlock3Resnet1Tester(TADSRUNetUpBlock3Attention0Tester):
    """Minimal tester for only up_blocks.3.resnets.1.

    The bridge executes the already aligned chain:
    entry -> all down_blocks -> full local mid_block -> full local
    up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 ->
    up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 ->
    up_blocks.3.resnets.1.

    It deliberately stops before up_blocks.3.attentions.1, the rest of
    up_blocks.3, all-up-block aggregate checks, full UNet forward, runtime
    LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_resnet1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock3_resnet1_metadata', metadata.get('resnet1_metadata', metadata))
        prefix = up_meta.get('resnet1_config', {}).get('prefix', 'up_blocks_3_resnets_1')
        self.upblock3_resnet1 = UNetResnetBlock2DTester(upblock3_resnet1_weights_npz_path, prefix, up_meta)
        self.upblock3_resnet1_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock3_resnet1_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock3 resnet1 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock3_resnet1(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock3_resnet1_input(hidden_states, res_hidden_states)
        res = self.upblock3_resnet1.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_attention0 = prev.get('remaining_internal_residual_samples_after_upblock3_attention0')
        if remaining_after_attention0 is None:
            raise ValueError('up_blocks.3.resnets.1 expects the residual tuple left after up_blocks.3.attentions.0')
        if len(remaining_after_attention0) < 1:
            raise ValueError('up_blocks.3.resnets.1 requires one remaining official skip residual')
        res_hidden = np.asarray(remaining_after_attention0[-1], dtype=np.float32)
        remaining_after_resnet1 = tuple(remaining_after_attention0[:-1])
        up = self.run_upblock3_resnet1(prev['upblock3_attention0_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock3_resnet1_hidden_input': prev['upblock3_attention0_output'],
            'upblock3_resnet1_res_hidden': res_hidden,
            'upblock3_resnet1_concat_input': up['concat_input'],
            'upblock3_resnet1_output': up['output'],
            'upblock3_output_hidden_after_resnet1': up['output'],
            'remaining_internal_residual_samples_after_upblock3_resnet1': remaining_after_resnet1,
        })
        if return_intermediates:
            out['upblock3_resnet1_intermediates'] = up
        return out


class TADSRUNetUpBlock3Attention1Tester(TADSRUNetUpBlock3Resnet1Tester):
    """Minimal tester for only up_blocks.3.attentions.1.

    The bridge executes the already aligned chain:
    entry -> all down_blocks -> full local mid_block -> full local
    up_blocks.0 -> full local up_blocks.1 -> full local up_blocks.2 ->
    up_blocks.3.resnets.0 -> up_blocks.3.attentions.0 ->
    up_blocks.3.resnets.1 -> up_blocks.3.attentions.1.

    It deliberately stops before up_blocks.3.resnets.2, the rest of
    up_blocks.3, all-up-block aggregate checks, full UNet forward, runtime
    LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_attention1_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_attention1_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock3_attention1_metadata', metadata.get('attention1_metadata', metadata))
        self.upblock3_attention1 = UNetAttention0Transformer2DTester(upblock3_attention1_weights_npz_path, metadata=att_meta)

    def run_upblock3_attention1(
        self,
        hidden_states: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        att = self.upblock3_attention1.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_resnet1 = prev.get('remaining_internal_residual_samples_after_upblock3_resnet1')
        if remaining_after_resnet1 is None:
            raise ValueError('up_blocks.3.attentions.1 expects the residual tuple left after up_blocks.3.resnets.1; attention1 must not consume it')
        att = self.run_upblock3_attention1(prev['upblock3_resnet1_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock3_attention1_input': prev['upblock3_resnet1_output'],
            'upblock3_attention1_output': att['output'],
            'upblock3_output_hidden_after_attention1': att['output'],
            'remaining_internal_residual_samples_after_upblock3_attention1': remaining_after_resnet1,
        })
        if return_intermediates:
            out['upblock3_attention1_intermediates'] = att
        return out



class TADSRUNetUpBlock3Resnet2Tester(TADSRUNetUpBlock3Attention1Tester):
    """Minimal tester for only up_blocks.3.resnets.2.

    The bridge executes the already aligned chain through
    up_blocks.3.attentions.1 and then consumes the final official skip
    residual for up_blocks.3.resnets.2. It deliberately stops before
    up_blocks.3.attentions.2, full up_blocks.3, full UNet forward, runtime
    LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_resnet2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_resnet2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        up_meta = metadata.get('upblock3_resnet2_metadata', metadata.get('resnet2_metadata', metadata))
        prefix = up_meta.get('resnet2_config', {}).get('prefix', 'up_blocks_3_resnets_2')
        self.upblock3_resnet2 = UNetResnetBlock2DTester(upblock3_resnet2_weights_npz_path, prefix, up_meta)
        self.upblock3_resnet2_residual_metadata = up_meta.get('residual_consumption', {})

    def concat_upblock3_resnet2_input(self, hidden_states: np.ndarray, res_hidden_states: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden_states, dtype=np.float32)
        residual = np.asarray(res_hidden_states, dtype=np.float32)
        if hidden.ndim != 4 or residual.ndim != 4 or hidden.shape[0] != residual.shape[0] or hidden.shape[2:] != residual.shape[2:]:
            raise ValueError(f'Cannot concat official upblock3 resnet2 residual: hidden={hidden.shape}, residual={residual.shape}')
        return np.concatenate([hidden, residual], axis=1).astype(np.float32)

    def run_upblock3_resnet2(
        self,
        hidden_states: np.ndarray,
        res_hidden_states: np.ndarray,
        emb: np.ndarray,
        return_intermediates: bool = False,
    ):
        concat = self.concat_upblock3_resnet2_input(hidden_states, res_hidden_states)
        res = self.upblock3_resnet2.run(concat, emb, return_intermediates=True)
        out = {
            'hidden_input': np.asarray(hidden_states, dtype=np.float32),
            'res_hidden': np.asarray(res_hidden_states, dtype=np.float32),
            'concat_input': concat,
            'output': res['output'],
        }
        if return_intermediates:
            out.update(res)
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_attention1 = prev.get('remaining_internal_residual_samples_after_upblock3_attention1')
        if remaining_after_attention1 is None:
            raise ValueError('up_blocks.3.resnets.2 expects the residual tuple left after up_blocks.3.attentions.1')
        if len(remaining_after_attention1) < 1:
            raise ValueError('up_blocks.3.resnets.2 requires one remaining official skip residual')
        res_hidden = np.asarray(remaining_after_attention1[-1], dtype=np.float32)
        remaining_after_resnet2 = tuple(remaining_after_attention1[:-1])
        up = self.run_upblock3_resnet2(prev['upblock3_attention1_output'], res_hidden, prev['time_embedding'], return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock3_resnet2_hidden_input': prev['upblock3_attention1_output'],
            'upblock3_resnet2_res_hidden': res_hidden,
            'upblock3_resnet2_concat_input': up['concat_input'],
            'upblock3_resnet2_output': up['output'],
            'upblock3_output_hidden_after_resnet2': up['output'],
            'remaining_internal_residual_samples_after_upblock3_resnet2': remaining_after_resnet2,
        })
        if return_intermediates:
            out['upblock3_resnet2_intermediates'] = up
        return out



class TADSRUNetUpBlock3Attention2Tester(TADSRUNetUpBlock3Resnet2Tester):
    """Minimal tester for only up_blocks.3.attentions.2.

    The bridge executes the already aligned chain through
    up_blocks.3.resnets.2 and then applies up_blocks.3.attentions.2.
    It deliberately stops before the output tail, full up_blocks.3,
    full UNet forward, runtime LoRA, and full TADSR inference.
    """

    def __init__(
        self,
        *args,
        upblock3_attention2_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_upblock3_attention2_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        att_meta = metadata.get('upblock3_attention2_metadata', metadata.get('attention2_metadata', metadata))
        self.upblock3_attention2 = UNetAttention0Transformer2DTester(upblock3_attention2_weights_npz_path, metadata=att_meta)

    def run_upblock3_attention2(
        self,
        hidden_states: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        att = self.upblock3_attention2.run_attention0(hidden_states, encoder_hidden_states, return_intermediates=True)
        if not return_intermediates:
            return att['output']
        return att

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2_attention2(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        prev = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        remaining_after_resnet2 = prev.get('remaining_internal_residual_samples_after_upblock3_resnet2')
        if remaining_after_resnet2 is None:
            raise ValueError('up_blocks.3.attentions.2 expects the residual tuple left after up_blocks.3.resnets.2; attention2 must not consume it')
        att = self.run_upblock3_attention2(prev['upblock3_resnet2_output'], encoder_hidden_states, return_intermediates=True)
        out = dict(prev)
        out.update({
            'upblock3_attention2_input': prev['upblock3_resnet2_output'],
            'upblock3_attention2_output': att['output'],
            'upblock3_output_hidden_after_attention2': att['output'],
            'remaining_internal_residual_samples_after_upblock3_attention2': remaining_after_resnet2,
        })
        if return_intermediates:
            out['upblock3_attention2_intermediates'] = att
        return out


class TADSRUNetUpBlock3LocalTester(TADSRUNetUpBlock3Attention2Tester):
    """Local aggregate tester for TADSR UNet up_blocks.3 only.

    The aggregate follows the official final up block order:
    resnets.0 -> attentions.0 -> resnets.1 -> attentions.1 ->
    resnets.2 -> attentions.2.

    It deliberately stops at the output-tail boundary. It does not run
    conv_norm_out, conv_act, conv_out, full UNet forward, runtime LoRA, or full
    TADSR inference.
    """

    def run_upblock3_local(
        self,
        hidden_states: np.ndarray,
        residual_samples,
        emb: np.ndarray,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        residual_tuple = tuple(np.asarray(x, dtype=np.float32) for x in residual_samples)
        if len(residual_tuple) != 3:
            raise ValueError(f'up_blocks.3 local tester expects exactly 3 residual samples, got {len(residual_tuple)}')

        res0_hidden = residual_tuple[-1]
        remaining_after_resnet0 = residual_tuple[:-1]
        res0 = self.run_upblock3_resnet0(hidden_states, res0_hidden, emb, return_intermediates=True)
        att0 = self.run_upblock3_attention0(res0['output'], encoder_hidden_states, return_intermediates=True)

        res1_hidden = remaining_after_resnet0[-1]
        remaining_after_resnet1 = remaining_after_resnet0[:-1]
        res1 = self.run_upblock3_resnet1(att0['output'], res1_hidden, emb, return_intermediates=True)
        att1 = self.run_upblock3_attention1(res1['output'], encoder_hidden_states, return_intermediates=True)

        res2_hidden = remaining_after_resnet1[-1]
        remaining_after_resnet2 = remaining_after_resnet1[:-1]
        res2 = self.run_upblock3_resnet2(att1['output'], res2_hidden, emb, return_intermediates=True)
        att2 = self.run_upblock3_attention2(res2['output'], encoder_hidden_states, return_intermediates=True)

        out = {
            'input_hidden_states': np.asarray(hidden_states, dtype=np.float32),
            'input_residual_samples': residual_tuple,
            'upblock3_resnet0_res_hidden': res0_hidden,
            'upblock3_resnet0_output': res0['output'],
            'upblock3_attention0_output': att0['output'],
            'upblock3_resnet1_res_hidden': res1_hidden,
            'upblock3_resnet1_output': res1['output'],
            'upblock3_attention1_output': att1['output'],
            'upblock3_resnet2_res_hidden': res2_hidden,
            'upblock3_resnet2_output': res2['output'],
            'upblock3_attention2_output': att2['output'],
            'upblock3_output_hidden': att2['output'],
            'upblock3_output_states': [],
            'remaining_internal_residual_samples_after_upblock3': remaining_after_resnet2,
        }
        if return_intermediates:
            out.update({
                'upblock3_resnet0_intermediates': res0,
                'upblock3_attention0_intermediates': att0,
                'upblock3_resnet1_intermediates': res1,
                'upblock3_attention1_intermediates': att1,
                'upblock3_resnet2_intermediates': res2,
                'upblock3_attention2_intermediates': att2,
            })
        return out

    def run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_local(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        out = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_resnet0_attention0_resnet1_attention1_resnet2_attention2(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        out['upblock3_output_hidden'] = out['upblock3_attention2_output']
        out['upblock3_output_states'] = []
        out['remaining_internal_residual_samples_after_upblock3'] = out[
            'remaining_internal_residual_samples_after_upblock3_attention2'
        ]
        out['output_tail_boundary'] = 'output_tail'
        if not return_intermediates:
            for key in list(out.keys()):
                if key.endswith('_intermediates'):
                    out.pop(key, None)
        return out


class TADSRUNetOutputTailTester(TADSRUNetUpBlock3LocalTester):
    """Minimal tester for the TADSR UNet output tail.

    The tested tail is exactly:
    conv_norm_out -> conv_act -> conv_out.

    It can run either from an already computed hidden tensor or from the
    currently aligned manual chain:
    entry -> all down_blocks -> full local mid_block -> full local up_blocks.0
    -> full local up_blocks.1 -> full local up_blocks.2 -> full local
    up_blocks.3 -> output tail.

    This still does not implement official UNet.forward, scheduler denoising,
    runtime LoRA application, VAE integration, image postprocessing, or full
    TADSR inference.
    """

    def __init__(
        self,
        *args,
        output_tail_weights_npz_path: str | Path = 'experiments/full_repro/unet_alignment/converted_unet_output_tail_effective_weights.npz',
        metadata: dict | None = None,
        **kwargs,
    ):
        metadata = metadata or {}
        super().__init__(*args, metadata=metadata, **kwargs)
        self.output_tail_weights_path = Path(output_tail_weights_npz_path)
        if not self.output_tail_weights_path.exists():
            raise FileNotFoundError(f'UNet output tail effective weights missing: {self.output_tail_weights_path}')
        self.output_tail_weights = np.load(self.output_tail_weights_path)
        tail_meta = metadata.get('output_tail_metadata', metadata.get('output_tail_config', metadata))
        self.output_tail_groups = int(tail_meta.get('conv_norm_out_num_groups', tail_meta.get('num_groups', 32)))
        self.output_tail_eps = float(tail_meta.get('conv_norm_out_eps', tail_meta.get('eps', 1e-5)))
        padding = tail_meta.get('conv_out_padding', 1)
        if isinstance(padding, (list, tuple)):
            padding = padding[0]
        stride = tail_meta.get('conv_out_stride', 1)
        if isinstance(stride, (list, tuple)):
            stride = stride[0]
        self.output_tail_conv_padding = int(padding)
        self.output_tail_conv_stride = int(stride)
        if self.output_tail_conv_stride != 1:
            raise NotImplementedError(f'Output tail conv_out stride must be 1 for this tester, got {self.output_tail_conv_stride}')
        self._validate_output_tail_weights()

    def tail_get(self, key: str) -> np.ndarray:
        candidates = [key, f'output_tail_{key}']
        for candidate in candidates:
            if candidate in self.output_tail_weights.files:
                return self.output_tail_weights[candidate].astype(np.float32)
        nearby = list(self.output_tail_weights.files)[:80]
        raise KeyError(f'Missing UNet output tail key {key}; nearby={nearby}')

    def _validate_output_tail_weights(self) -> None:
        required = [
            'output_tail_conv_norm_out_weight',
            'output_tail_conv_norm_out_bias',
            'output_tail_conv_out_weight',
            'output_tail_conv_out_bias',
        ]
        missing = [key for key in required if key not in self.output_tail_weights.files]
        if missing:
            nearby = list(self.output_tail_weights.files)[:80]
            raise KeyError(f'Missing UNet output tail effective weights: {missing}; nearby={nearby}')

    def run_output_tail(self, hidden_states: np.ndarray, return_intermediates: bool = False):
        hidden = np.asarray(hidden_states, dtype=np.float32)
        norm = group_norm(
            hidden,
            self.tail_get('output_tail_conv_norm_out_weight'),
            self.tail_get('output_tail_conv_norm_out_bias'),
            self.output_tail_groups,
            self.output_tail_eps,
        )
        act = silu(norm)
        conv = conv2d_nchw(
            act,
            self.tail_get('output_tail_conv_out_weight'),
            self.tail_get('output_tail_conv_out_bias'),
            stride=1,
            padding=self.output_tail_conv_padding,
        )
        out = {
            'output_tail_input': hidden,
            'output_tail_norm_output': norm,
            'output_tail_act_output': act,
            'output_tail_conv_out_output': conv,
            'output_tail_output': conv,
        }
        if not return_intermediates:
            return conv
        return out

    def run_entry_downblocks_midblock_upblocks_output_tail(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        up = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_local(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        tail = self.run_output_tail(up['upblock3_output_hidden'], return_intermediates=True)
        out = dict(up)
        out.update({
            'output_tail_norm_output': tail['output_tail_norm_output'],
            'output_tail_act_output': tail['output_tail_act_output'],
            'output_tail_conv_out_output': tail['output_tail_conv_out_output'],
            'output_tail_output': tail['output_tail_output'],
        })
        if return_intermediates:
            out['output_tail_intermediates'] = tail
            return out
        for key in list(out.keys()):
            if key.endswith('_intermediates'):
                out.pop(key, None)
        return out


class TADSRUNetManualWrapperTester(TADSRUNetOutputTailTester):
    """Alignment-only manual UNet chain wrapper.

    This wrapper composes the already verified migration testers from UNet
    input tensors through the output tail:
    entry -> down_blocks -> mid_block -> up_blocks.0/1/2/3 -> output tail.

    It is deliberately not a production forward method. It does not call the
    official PyTorch UNet forward, does not implement scheduler denoising, VAE
    integration, runtime LoRA, image generation, or full TADSR inference.
    """

    def run_manual_unet_chain_for_alignment(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        out = self.run_entry_downblocks_midblock_upblocks_output_tail(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        out['manual_wrapper_output'] = out['output_tail_output']
        out['manual_wrapper_return_type'] = 'tensor' if not return_intermediates else 'dict_with_intermediates'
        out['manual_wrapper_scope'] = (
            'alignment_only_entry_downblocks_midblock_upblocks_output_tail; '
            'not_official_unet_forward'
        )
        if return_intermediates:
            return out
        return out['manual_wrapper_output']

    def run_entry_downblocks_midblock_upblocks_output_tail(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_intermediates: bool = False,
    ):
        up = self.run_entry_downblocks_midblock_upblock0_upblock1_upblock2_upblock3_local(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        tail = self.run_output_tail(up['upblock3_output_hidden'], return_intermediates=True)
        out = dict(up)
        out.update({
            'output_tail_norm_output': tail['output_tail_norm_output'],
            'output_tail_act_output': tail['output_tail_act_output'],
            'output_tail_conv_out_output': tail['output_tail_conv_out_output'],
            'output_tail_output': tail['output_tail_output'],
        })
        if return_intermediates:
            out['output_tail_intermediates'] = tail
            return out
        for key in list(out.keys()):
            if key.endswith('_intermediates'):
                out.pop(key, None)
        return out


class TADSRUNetFullForwardAlignmentTester(TADSRUNetManualWrapperTester):
    """Alignment-only wrapper matching the official UNet full-forward contract.

    This method is intentionally named for alignment/testing. It reuses the
    manual full-chain wrapper and can return either a tensor or a minimal
    dict-like {"sample": output} contract for tests. It is not a production
    forward path and does not enable scheduler, VAE, runtime LoRA, image
    generation, or full TADSR inference.
    """

    def run_full_forward_for_alignment(
        self,
        sample: np.ndarray,
        timestep,
        encoder_hidden_states: np.ndarray,
        return_dict: bool = False,
        return_intermediates: bool = False,
    ):
        out = self.run_manual_unet_chain_for_alignment(
            sample,
            timestep,
            encoder_hidden_states,
            return_intermediates=True,
        )
        final = out['manual_wrapper_output']
        if return_intermediates:
            out = dict(out)
            out['full_forward_alignment_output'] = final
            out['official_contract_return_dict'] = bool(return_dict)
            return out
        if return_dict:
            return {'sample': final}
        return final
