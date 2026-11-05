"""Define shared layers, task gates, and convolution blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

INPUT_DIM = 1024


RMS_EPS = 1e-6


class FrameRMSNorm(nn.Module):

    def __init__(self, dim: int = INPUT_DIM, *, eps: float = RMS_EPS) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(
        self, tensor: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = tensor.float() * torch.rsqrt(
            tensor.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        output = (normalized * self.weight.float()).to(dtype=tensor.dtype)
        if pad_mask is not None:
            output = output.masked_fill(pad_mask.unsqueeze(-1), 0)
        return output


class SinusoidalPE(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, D = x.shape[1], x.shape[2]

        pos = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(0, D, 2, device=x.device, dtype=torch.float32)

        angles = pos / (10000.0 ** (dim / D))

        pe = torch.zeros(T, D, device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return x + pe.to(dtype=x.dtype)


def _check_lengths_and_mask(
    features: torch.Tensor,
    lengths: torch.Tensor,
    pad_mask: torch.Tensor,
) -> None:

    if features.ndim < 3:
        raise ValueError("features 至少需要 B×T×D 三维")
    batch_size, time_steps = features.shape[:2]
    if lengths.shape != (batch_size,):
        raise ValueError("lengths 必须为 B 维真实帧数")
    if pad_mask.shape != (batch_size, time_steps):
        raise ValueError("pad_mask 必须为 B×T")


# Task-specific gating.
class TaskGate(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.input_linear = nn.Linear(256, 64)
        self.activation = nn.GELU()
        self.output_linear = nn.Linear(64, 256)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:

        _check_lengths_and_mask(features, lengths, pad_mask)
        if features.shape[-1] != 256:
            raise ValueError("TaskGate 需要 B×T×256 输入")
        x = features.masked_fill(pad_mask.unsqueeze(-1), 0)
        hidden = self.activation(self.input_linear(self.norm(x)))
        weights = 0.5 + torch.sigmoid(self.output_linear(hidden))
        return (x * weights).masked_fill(pad_mask.unsqueeze(-1), 0)


def _zero_padding(tensor: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:

    expanded_mask = pad_mask
    while expanded_mask.ndim < tensor.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    return tensor.masked_fill(expanded_mask, 0)


def _pad_mask_from_lengths(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:

    return torch.arange(time_steps, device=lengths.device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)


class LayerScaledDepthwiseSeparableBlock(nn.Module):

    def __init__(self, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            256,
            256,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=256,
        )
        self.pre_norm = nn.LayerNorm(256)
        self.pointwise = nn.Conv1d(256, 256, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        residual = _zero_padding(x, pad_mask)
        normalized = _zero_padding(self.pre_norm(residual), pad_mask)
        transformed = _zero_padding(
            self.depthwise(normalized.transpose(1, 2)).transpose(1, 2), pad_mask
        )
        transformed = _zero_padding(
            self.pointwise(transformed.transpose(1, 2)).transpose(1, 2), pad_mask
        )
        transformed = _zero_padding(self.activation(transformed), pad_mask)
        transformed = _zero_padding(self.dropout(transformed), pad_mask)
        return _zero_padding(residual + self.residual_scale * transformed, pad_mask)


# Shared feature encoder.
class SharedLightweightFrontEnd(nn.Module):

    def __init__(self, input_dim: int = 1024) -> None:
        super().__init__()
        self.rms_norm = FrameRMSNorm(input_dim)
        self.projection = nn.Linear(input_dim, 256)
        self.activation = nn.GELU()
        self.block1 = LayerScaledDepthwiseSeparableBlock(
            kernel_size=3, dilation=1, dropout=0.1
        )
        self.block2 = LayerScaledDepthwiseSeparableBlock(
            kernel_size=3, dilation=1, dropout=0.1
        )
        self.norm = nn.LayerNorm(256)

    def forward(
        self,
        features: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = _zero_padding(self.rms_norm(features, pad_mask), pad_mask)
        projected = _zero_padding(
            self.projection(_zero_padding(normalized, pad_mask)), pad_mask
        )
        x = _zero_padding(self.activation(_zero_padding(projected, pad_mask)), pad_mask)
        x = self.block1(_zero_padding(x, pad_mask), pad_mask)
        x = self.block2(_zero_padding(x, pad_mask), pad_mask)
        return _zero_padding(self.norm(x), pad_mask)


class SameRateDecoderBlock(nn.Module):

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=padding, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        residual = _zero_padding(x, pad_mask)
        x = _zero_padding(
            self.depthwise(residual.transpose(1, 2)).transpose(1, 2), pad_mask
        )
        x = _zero_padding(self.pointwise(x.transpose(1, 2)).transpose(1, 2), pad_mask)
        x = _zero_padding(self.activation(x), pad_mask)
        x = _zero_padding(self.dropout(x), pad_mask)
        return _zero_padding(residual + x, pad_mask)


class StrideDepthwiseSeparableBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 5, stride=2, padding=2, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, output_mask: torch.Tensor) -> torch.Tensor:
        x = _zero_padding(
            self.depthwise(x.transpose(1, 2)).transpose(1, 2), output_mask
        )
        x = _zero_padding(
            self.pointwise(x.transpose(1, 2)).transpose(1, 2), output_mask
        )
        return _zero_padding(self.activation(x), output_mask)


class MaskedDecoderHead(nn.Module):

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(256, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = _zero_padding(self.projection(_zero_padding(x, pad_mask)), pad_mask)
        x = _zero_padding(self.activation(x), pad_mask)
        return _zero_padding(self.output(x), pad_mask)
