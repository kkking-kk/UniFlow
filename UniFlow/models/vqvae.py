import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encdec import Decoder, Encoder
from models.quantize_cnn import (
    LFQ,
    MultiScaleQuantizeEMAResetSim,
    QuantizeEMA,
    QuantizeEMAReset,
    QuantizeEMAResetSim,
    QuantizeReset,
    Quantizer,
)
from models.resnet import Resnet1D

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def _get_input_emb_width(dataname):
    widths = {
        "stock": 6,
        "energy": 28,
        "sine": 5,
        "etth": 7,
        "fmri": 50,
        "mujoco": 14,
    }
    if dataname not in widths:
        raise ValueError(f"Unsupported dataname: {dataname}")
    return widths[dataname]


def _split_dims_evenly(total_dim, num_parts):
    base = total_dim // num_parts
    remainder = total_dim % num_parts
    dims = []
    for idx in range(num_parts):
        dims.append(base + (1 if idx < remainder else 0))
    return dims


def _conv1d_out_length(length, kernel_size, stride, padding, dilation=1):
    return (
        (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
    ) + 1


def _downsample_length(length, down_t, stride_t):
    filter_t = stride_t * 2
    pad_t = stride_t // 2
    for _ in range(down_t):
        length = _conv1d_out_length(length, filter_t, stride_t, pad_t)
    return length


def _patchify_length(length, patch_size):
    return (length + patch_size - 1) // patch_size


def _search_exact_patch_downs(raw_length, target_length, stride_t, max_steps=8):
    if raw_length == target_length:
        return 0

    current_length = raw_length
    for step in range(1, max_steps + 1):
        current_length = _downsample_length(current_length, 1, stride_t)
        if current_length == target_length:
            return step
        if current_length < target_length:
            break
    return None


def _build_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "silu":
        return nn.SiLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


def _resolve_multiscale_config(
    args,
    code_dim,
    depth,
    dilation_growth_rate,
    down_t,
    stride_t,
):
    if not getattr(args, "multi_scale_vq", False):
        return None

    if args.quantizer != "ema_reset_sim":
        raise ValueError("Multi-scale VQ is only implemented for quantizer=ema_reset_sim")

    scale_dims = getattr(args, "multi_scale_dims", None)
    if not scale_dims:
        scale_dims = _split_dims_evenly(code_dim, 2)
    scale_dims = [int(dim) for dim in scale_dims]

    num_scales = len(scale_dims)
    if sum(scale_dims) != code_dim:
        raise ValueError(
            f"Sum of multi-scale dims {scale_dims} must equal code_dim {code_dim}"
        )

    mode = getattr(args, "multi_scale_mode", "receptive")
    if mode not in {"receptive", "patch"}:
        raise ValueError(f"Unsupported multi-scale mode: {mode}")

    scale_depths = getattr(args, "multi_scale_depths", None)
    if not scale_depths:
        if mode == "patch":
            scale_depths = [depth] * num_scales
        elif num_scales == 1:
            scale_depths = [depth]
        elif num_scales == 2:
            scale_depths = [max(1, depth - 1), depth + 1]
        else:
            scale_depths = [max(1, depth - 1 + idx) for idx in range(num_scales)]
    scale_depths = [int(value) for value in scale_depths]

    scale_dilations = getattr(args, "multi_scale_dilations", None)
    if not scale_dilations:
        if mode == "patch":
            scale_dilations = [dilation_growth_rate] * num_scales
        elif num_scales == 1:
            scale_dilations = [dilation_growth_rate]
        elif num_scales == 2:
            scale_dilations = [1, dilation_growth_rate]
        else:
            scale_dilations = [1] + [dilation_growth_rate] * (num_scales - 1)
    scale_dilations = [int(value) for value in scale_dilations]

    scale_weights = getattr(args, "multi_scale_weights", None)
    if scale_weights:
        scale_weights = [float(value) for value in scale_weights]

    if len(scale_depths) != num_scales:
        raise ValueError(
            f"Expected {num_scales} multi-scale depths, got {len(scale_depths)}"
        )
    if len(scale_dilations) != num_scales:
        raise ValueError(
            f"Expected {num_scales} multi-scale dilations, got {len(scale_dilations)}"
        )
    if scale_weights is not None and len(scale_weights) != num_scales:
        raise ValueError(
            f"Expected {num_scales} multi-scale weights, got {len(scale_weights)}"
        )

    cfg = {
        "mode": mode,
        "scale_dims": scale_dims,
        "scale_depths": scale_depths,
        "scale_dilations": scale_dilations,
        "scale_weights": scale_weights,
    }

    if mode == "patch":
        scale_patches = getattr(args, "multi_scale_patches", None)
        if not scale_patches:
            raise ValueError(
                "Patch-mode multi-scale VQ requires --multi-scale-patches, "
                "for example --multi-scale-patches 3 6"
            )
        scale_patches = [int(value) for value in scale_patches]
        if len(scale_patches) != num_scales:
            raise ValueError(
                f"Expected {num_scales} patch sizes, got {len(scale_patches)}"
            )
        if any(patch <= 0 for patch in scale_patches):
            raise ValueError(f"Patch sizes must be positive, got {scale_patches}")

        raw_lengths = [
            _patchify_length(int(args.window_size), patch_size)
            for patch_size in scale_patches
        ]
        patch_downs = getattr(args, "multi_scale_patch_downs", None)
        if patch_downs:
            patch_downs = [int(value) for value in patch_downs]
            if len(patch_downs) != num_scales:
                raise ValueError(
                    f"Expected {num_scales} patch downsample counts, got {len(patch_downs)}"
                )
        else:
            target_length = min(raw_lengths)
            patch_downs = []
            for raw_length in raw_lengths:
                down_steps = _search_exact_patch_downs(
                    raw_length,
                    target_length,
                    stride_t=stride_t,
                    max_steps=max(down_t + 6, 8),
                )
                if down_steps is None:
                    raise ValueError(
                        "Could not align true multi-patch token lengths by only adding "
                        "stride downsampling layers. Please set --multi-scale-patch-downs "
                        f"explicitly. Raw lengths={raw_lengths}, target={target_length}."
                    )
                patch_downs.append(down_steps)

        aligned_lengths = [
            _downsample_length(raw_length, down_steps, stride_t)
            for raw_length, down_steps in zip(raw_lengths, patch_downs)
        ]
        if len(set(aligned_lengths)) != 1:
            raise ValueError(
                "True patch-mode multiscale requires all branches to align to the same "
                f"token length, got raw_lengths={raw_lengths}, patch_downs={patch_downs}, "
                f"aligned_lengths={aligned_lengths}"
            )

        cfg.update(
            {
                "scale_patches": scale_patches,
                "scale_patch_downs": patch_downs,
                "scale_raw_lengths": raw_lengths,
                "scale_aligned_lengths": aligned_lengths,
                "shared_index_length": aligned_lengths[0],
                "decoder_input_length": _downsample_length(
                    int(args.window_size), down_t, stride_t
                ),
            }
        )

    return cfg


class PatchEncoder1D(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        patch_size,
        patch_down_t,
        stride_t,
        width,
        depth,
        dilation_growth_rate,
        activation="relu",
        norm=None,
        quantizer=None,
    ):
        super().__init__()
        self.patch_size = int(patch_size)

        filter_t = stride_t * 2
        pad_t = stride_t // 2
        blocks = [
            nn.Conv1d(input_emb_width, width, self.patch_size, self.patch_size, 0),
            _build_activation(activation),
            Resnet1D(
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            ),
        ]
        for _ in range(int(patch_down_t)):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                    Resnet1D(
                        width,
                        depth,
                        dilation_growth_rate,
                        activation=activation,
                        norm=norm,
                    ),
                )
            )
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        if quantizer == "lfq":
            blocks.append(nn.Tanh())
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        pad_right = (-x.shape[-1]) % self.patch_size
        if pad_right > 0:
            if x.shape[-1] > 1:
                x = F.pad(x, (0, pad_right), mode="replicate")
            else:
                x = F.pad(x, (0, pad_right))
        return self.model(x)


class LengthAdapter1D(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = int(target_length)

    def forward(self, x):
        if x.shape[-1] == self.target_length:
            return x
        return F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)


class TSVQVAE(nn.Module):
    def __init__(
        self,
        args,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.args = args
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.use_multiscale = False
        self.multiscale_mode = None
        self.output_seq_len = int(getattr(args, "window_size", 0))

        input_emb_width = _get_input_emb_width(args.dataname)
        output_emb_width = self.code_dim

        if args.quantizer == "lfq":
            output_emb_width = math.ceil(math.log2(self.num_code))
            self.code_dim = output_emb_width

        multiscale_cfg = _resolve_multiscale_config(
            args,
            code_dim=self.code_dim,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            down_t=down_t,
            stride_t=stride_t,
        )

        if multiscale_cfg is not None:
            self.use_multiscale = True
            self.multiscale_mode = multiscale_cfg["mode"]
            self.scale_dims = multiscale_cfg["scale_dims"]
            self.scale_depths = multiscale_cfg["scale_depths"]
            self.scale_dilations = multiscale_cfg["scale_dilations"]
            self.scale_weights = multiscale_cfg["scale_weights"]

            if self.multiscale_mode == "patch":
                self.scale_patches = multiscale_cfg["scale_patches"]
                self.scale_patch_downs = multiscale_cfg["scale_patch_downs"]
                self.scale_raw_lengths = multiscale_cfg["scale_raw_lengths"]
                self.scale_aligned_lengths = multiscale_cfg["scale_aligned_lengths"]
                self.shared_index_length = multiscale_cfg["shared_index_length"]
                self.decoder_input_length = multiscale_cfg["decoder_input_length"]

                self.encoders = nn.ModuleList(
                    [
                        PatchEncoder1D(
                            input_emb_width,
                            scale_dim,
                            patch_size,
                            patch_down_t,
                            stride_t,
                            width,
                            scale_depth,
                            scale_dilation,
                            activation=activation,
                            norm=norm,
                            quantizer=args.quantizer,
                        )
                        for scale_dim, scale_depth, scale_dilation, patch_size, patch_down_t in zip(
                            self.scale_dims,
                            self.scale_depths,
                            self.scale_dilations,
                            self.scale_patches,
                            self.scale_patch_downs,
                        )
                    ]
                )
                self.decoder_input_adapter = LengthAdapter1D(self.decoder_input_length)
            else:
                self.shared_index_length = _downsample_length(
                    int(args.window_size), down_t, stride_t
                )
                self.decoder_input_length = self.shared_index_length
                self.encoders = nn.ModuleList(
                    [
                        Encoder(
                            input_emb_width,
                            scale_dim,
                            down_t,
                            stride_t,
                            width,
                            scale_depth,
                            scale_dilation,
                            activation=activation,
                            norm=norm,
                            quantizer=args.quantizer,
                        )
                        for scale_dim, scale_depth, scale_dilation in zip(
                            self.scale_dims,
                            self.scale_depths,
                            self.scale_dilations,
                        )
                    ]
                )
                self.decoder_input_adapter = nn.Identity()

            self.decoder = Decoder(
                input_emb_width,
                self.code_dim,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.quantizer = MultiScaleQuantizeEMAResetSim(
                nb_code,
                self.code_dim,
                self.scale_dims,
                args,
                scale_weights=self.scale_weights,
            )

            self.register_buffer("ms_mode", torch.tensor(
                1 if self.multiscale_mode == "patch" else 0, dtype=torch.long
            ))
            self.register_buffer(
                "ms_scale_dims",
                torch.tensor(self.scale_dims, dtype=torch.long),
            )
            self.register_buffer(
                "ms_scale_depths",
                torch.tensor(self.scale_depths, dtype=torch.long),
            )
            self.register_buffer(
                "ms_scale_dilations",
                torch.tensor(self.scale_dilations, dtype=torch.long),
            )
            if self.scale_weights is not None:
                self.register_buffer(
                    "ms_scale_weights",
                    torch.tensor(self.scale_weights, dtype=torch.float32),
                )
            if self.multiscale_mode == "patch":
                self.register_buffer(
                    "ms_scale_patches",
                    torch.tensor(self.scale_patches, dtype=torch.long),
                )
                self.register_buffer(
                    "ms_scale_patch_downs",
                    torch.tensor(self.scale_patch_downs, dtype=torch.long),
                )
                self.register_buffer(
                    "ms_scale_raw_lengths",
                    torch.tensor(self.scale_raw_lengths, dtype=torch.long),
                )
                self.register_buffer(
                    "ms_scale_aligned_lengths",
                    torch.tensor(self.scale_aligned_lengths, dtype=torch.long),
                )
                self.register_buffer(
                    "ms_decoder_input_length",
                    torch.tensor([self.decoder_input_length], dtype=torch.long),
                )
        else:
            self.encoder = Encoder(
                input_emb_width,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
                quantizer=args.quantizer,
            )
            self.decoder = Decoder(
                input_emb_width,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
            self.decoder_input_adapter = nn.Identity()
            if args.quantizer == "ema_reset":
                self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
            elif args.quantizer == "orig":
                self.quantizer = Quantizer(nb_code, code_dim, 1.0)
            elif args.quantizer == "ema":
                self.quantizer = QuantizeEMA(nb_code, code_dim, args)
            elif args.quantizer == "reset":
                self.quantizer = QuantizeReset(nb_code, code_dim, args)
            elif args.quantizer == "lfq":
                self.quantizer = LFQ(self.num_code, self.code_dim, args)
            elif args.quantizer == "ema_reset_sim":
                self.quantizer = QuantizeEMAResetSim(nb_code, code_dim, args)
            else:
                raise ValueError(f"Unsupported quantizer: {args.quantizer}")

    def get_multiscale_meta(self):
        if not self.use_multiscale:
            return None

        meta = {
            "mode": self.multiscale_mode,
            "scale_dims": list(self.scale_dims),
            "scale_depths": list(self.scale_depths),
            "scale_dilations": list(self.scale_dilations),
            "scale_weights": None
            if self.scale_weights is None
            else [float(value) for value in self.scale_weights],
        }
        if self.multiscale_mode == "patch":
            meta.update(
                {
                    "scale_patches": list(self.scale_patches),
                    "scale_patch_downs": list(self.scale_patch_downs),
                }
            )
        return meta

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def _match_output_length(self, x, target_length):
        if target_length is None or target_length <= 0 or x.shape[-1] == target_length:
            return x
        if x.shape[-1] > target_length:
            return x[..., :target_length]
        return F.interpolate(x, size=target_length, mode="linear", align_corners=False)

    def _encode_multiscale(self, x_in):
        x_encoders = [encoder(x_in) for encoder in self.encoders]
        seq_lens = {tensor.shape[-1] for tensor in x_encoders}
        if len(seq_lens) != 1:
            raise ValueError(
                f"All multi-scale encoders must output the same token length, got {sorted(seq_lens)}"
            )
        return x_encoders

    def _decode_quantized(self, x_quantized, target_length=None):
        x_quantized = self.decoder_input_adapter(x_quantized)
        x_decoder = self.decoder(x_quantized)
        x_decoder = self._match_output_length(x_decoder, target_length)
        return self.postprocess(x_decoder)

    def _normalize_selected_scales(self, selected_scales):
        if not self.use_multiscale:
            if selected_scales is None:
                return [0]
            selected = [int(scale_id) for scale_id in selected_scales]
            if selected != [0]:
                raise ValueError(
                    "Single-scale VQ-VAE only supports selected_scales=None or [0]"
                )
            return selected

        if selected_scales is None:
            return list(range(len(self.scale_dims)))

        selected = sorted({int(scale_id) for scale_id in selected_scales})
        if not selected:
            raise ValueError("selected_scales must not be empty")

        max_scale = len(self.scale_dims) - 1
        for scale_id in selected:
            if scale_id < 0 or scale_id > max_scale:
                raise ValueError(
                    f"Scale id {scale_id} is out of range [0, {max_scale}]"
                )
        return selected

    def build_quantized_latent_from_indices(self, code_idx, selected_scales=None):
        if code_idx.dim() != 2:
            raise ValueError(
                f"Expected code_idx with shape [B, L], got {tuple(code_idx.shape)}"
            )

        batch_size, seq_len = code_idx.shape
        if not self.use_multiscale:
            x_d = self.quantizer.dequantize(code_idx)
            return x_d.view(batch_size, seq_len, self.code_dim)

        selected = set(self._normalize_selected_scales(selected_scales))
        latent_parts = []
        for scale_id, (scale_dim, scale_quantizer) in enumerate(
            zip(self.scale_dims, self.quantizer.scale_quantizers)
        ):
            if scale_id in selected:
                latent_part = scale_quantizer.dequantize(code_idx)
            else:
                latent_part = torch.zeros(
                    batch_size,
                    seq_len,
                    scale_dim,
                    device=code_idx.device,
                    dtype=scale_quantizer.codebook.dtype,
                )
            latent_parts.append(latent_part)

        return torch.cat(latent_parts, dim=-1)

    def decode_from_quantized_latent(self, x_quantized, target_length=None):
        if x_quantized.dim() != 3:
            raise ValueError(
                f"Expected quantized latent with shape [B, L, D], got {tuple(x_quantized.shape)}"
            )
        if x_quantized.shape[-1] != self.code_dim:
            raise ValueError(
                f"Expected latent dim {self.code_dim}, got {x_quantized.shape[-1]}"
            )

        x_quantized = x_quantized.permute(0, 2, 1).contiguous()
        decode_target = self.output_seq_len if target_length is None else target_length
        return self._decode_quantized(x_quantized, target_length=decode_target)

    def forward_decoder_selected_scales(self, code_idx, selected_scales=None):
        x_quantized = self.build_quantized_latent_from_indices(
            code_idx, selected_scales=selected_scales
        )
        return self.decode_from_quantized_latent(x_quantized)

    def encode2(self, x):
        x_in = self.preprocess(x)
        if self.use_multiscale:
            x_encoder = torch.cat(self._encode_multiscale(x_in), dim=1)
        else:
            x_encoder = self.encoder(x_in)
        return self.postprocess(x_encoder)

    def encode_scale_latents(self, x):
        x_in = self.preprocess(x)
        if self.use_multiscale:
            return [self.postprocess(branch) for branch in self._encode_multiscale(x_in)]
        return [self.postprocess(self.encoder(x_in))]

    def quantize_single_scale_latent(self, scale_latent, scale_id=0):
        if scale_latent.dim() != 3:
            raise ValueError(
                f"Expected scale_latent with shape [B, L, D], got {tuple(scale_latent.shape)}"
            )

        batch_size, seq_len, latent_dim = scale_latent.shape

        if self.use_multiscale:
            if scale_id < 0 or scale_id >= len(self.scale_dims):
                raise ValueError(
                    f"Scale id {scale_id} is out of range [0, {len(self.scale_dims) - 1}]"
                )
            expected_dim = self.scale_dims[scale_id]
            if latent_dim != expected_dim:
                raise ValueError(
                    f"Scale {scale_id} expects latent dim {expected_dim}, got {latent_dim}"
                )
            scale_quantizer = self.quantizer.scale_quantizers[scale_id]
            scale_latent = F.normalize(
                scale_latent.contiguous().view(-1, latent_dim),
                dim=-1,
            )
            similarity = scale_quantizer.compute_similarity(scale_latent)
            _, code_idx = torch.max(similarity, dim=-1)
            return code_idx.view(batch_size, seq_len)

        if scale_id != 0:
            raise ValueError("Single-scale VQ-VAE only supports scale_id=0")

        scale_latent = scale_latent.contiguous().view(-1, latent_dim)
        code_idx = self.quantizer.quantize(scale_latent)
        return code_idx.view(batch_size, seq_len)

    def encode_single_scale(self, x, scale_id=0):
        scale_latents = self.encode_scale_latents(x)
        return self.quantize_single_scale_latent(scale_latents[scale_id], scale_id=scale_id)

    def encode(self, x):
        batch_size = x.shape[0]
        x_in = self.preprocess(x)
        if self.use_multiscale:
            x_encoders = self._encode_multiscale(x_in)
            seq_len = x_encoders[0].shape[-1]
            code_idx = self.quantizer.quantize(x_encoders)
            code_idx = code_idx.view(batch_size, seq_len)
            return code_idx

        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(batch_size, -1)
        return code_idx

    def forward(self, x):
        x_in = self.preprocess(x)
        if self.use_multiscale:
            x_encoder = self._encode_multiscale(x_in)
            x_quantized, loss, perplexity = self.quantizer(x_encoder)
        else:
            x_encoder = self.encoder(x_in)
            x_quantized, loss, perplexity = self.quantizer(x_encoder)

        x_out = self._decode_quantized(x_quantized, target_length=x_in.shape[-1])
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(x.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()
        return self._decode_quantized(x_d, target_length=self.output_seq_len)


class VQVAE(nn.Module):
    def __init__(
        self,
        args,
        nb_code=512,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self._base_args = copy.deepcopy(args)
        self._builder_kwargs = {
            "nb_code": nb_code,
            "code_dim": code_dim,
            "down_t": down_t,
            "stride_t": stride_t,
            "width": width,
            "depth": depth,
            "dilation_growth_rate": dilation_growth_rate,
            "activation": activation,
            "norm": norm,
        }
        self.vqvae = TSVQVAE(args, **self._builder_kwargs)

    def _build_model(self, args):
        return TSVQVAE(args, **self._builder_kwargs)

    def _extract_multiscale_meta(self, state_dict):
        dims_key = "vqvae.ms_scale_dims"
        if dims_key not in state_dict:
            return None

        mode_key = "vqvae.ms_mode"
        mode = "patch" if int(state_dict.get(mode_key, torch.tensor(0)).item()) == 1 else "receptive"
        meta = {
            "mode": mode,
            "scale_dims": state_dict[dims_key].tolist(),
            "scale_depths": state_dict["vqvae.ms_scale_depths"].tolist(),
            "scale_dilations": state_dict["vqvae.ms_scale_dilations"].tolist(),
        }
        weights_key = "vqvae.ms_scale_weights"
        if weights_key in state_dict:
            meta["scale_weights"] = [
                float(value) for value in state_dict[weights_key].tolist()
            ]
        else:
            meta["scale_weights"] = None

        patch_key = "vqvae.ms_scale_patches"
        if patch_key in state_dict:
            meta["scale_patches"] = state_dict[patch_key].tolist()
            meta["scale_patch_downs"] = state_dict[
                "vqvae.ms_scale_patch_downs"
            ].tolist()
        return meta

    def _maybe_rebuild_for_state_dict(self, state_dict):
        meta = self._extract_multiscale_meta(state_dict)
        current_meta = self.vqvae.get_multiscale_meta()

        if meta == current_meta:
            return

        ref_param = next(self.vqvae.parameters(), None)
        ref_buffer = next(self.vqvae.buffers(), None)
        ref_device = None
        ref_dtype = None
        if ref_param is not None:
            ref_device = ref_param.device
            ref_dtype = ref_param.dtype
        elif ref_buffer is not None:
            ref_device = ref_buffer.device
            ref_dtype = ref_buffer.dtype

        rebuilt_args = copy.deepcopy(self._base_args)
        if meta is None:
            setattr(rebuilt_args, "multi_scale_vq", False)
        else:
            setattr(rebuilt_args, "multi_scale_vq", True)
            setattr(rebuilt_args, "multi_scale_mode", meta["mode"])
            setattr(rebuilt_args, "multi_scale_dims", meta["scale_dims"])
            setattr(rebuilt_args, "multi_scale_depths", meta["scale_depths"])
            setattr(rebuilt_args, "multi_scale_dilations", meta["scale_dilations"])
            setattr(rebuilt_args, "multi_scale_weights", meta["scale_weights"])
            if meta["mode"] == "patch":
                setattr(rebuilt_args, "multi_scale_patches", meta["scale_patches"])
                setattr(
                    rebuilt_args,
                    "multi_scale_patch_downs",
                    meta["scale_patch_downs"],
                )
        rebuilt_model = self._build_model(rebuilt_args)
        if ref_device is not None:
            if ref_dtype is not None:
                rebuilt_model = rebuilt_model.to(device=ref_device, dtype=ref_dtype)
            else:
                rebuilt_model = rebuilt_model.to(device=ref_device)
        self.vqvae = rebuilt_model

    def load_state_dict(self, state_dict, strict=True):
        self._maybe_rebuild_for_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def encode(self, x):
        return self.vqvae.encode(x)

    def encode2(self, x):
        return self.vqvae.encode2(x)

    def encode_scale_latents(self, x):
        return self.vqvae.encode_scale_latents(x)

    def quantize_single_scale_latent(self, scale_latent, scale_id=0):
        return self.vqvae.quantize_single_scale_latent(scale_latent, scale_id=scale_id)

    def encode_single_scale(self, x, scale_id=0):
        return self.vqvae.encode_single_scale(x, scale_id=scale_id)

    def forward(self, x):
        return self.vqvae(x)

    def forward_decoder(self, x):
        return self.vqvae.forward_decoder(x)

    def build_quantized_latent_from_indices(self, code_idx, selected_scales=None):
        return self.vqvae.build_quantized_latent_from_indices(
            code_idx, selected_scales=selected_scales
        )

    def decode_from_quantized_latent(self, x_quantized, target_length=None):
        return self.vqvae.decode_from_quantized_latent(
            x_quantized, target_length=target_length
        )

    def forward_decoder_selected_scales(self, code_idx, selected_scales=None):
        return self.vqvae.forward_decoder_selected_scales(
            code_idx, selected_scales=selected_scales
        )
