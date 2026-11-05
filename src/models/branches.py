"""Define task encoders and prediction heads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import (
    MaskedDecoderHead,
    SameRateDecoderBlock,
    SinusoidalPE,
    StrideDepthwiseSeparableBlock,
    _pad_mask_from_lengths,
    _zero_padding,
)

if TYPE_CHECKING:
    from src.models.model import TaskRandomStreams
from torch.utils.checkpoint import checkpoint


def _clear_pad(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:

    return x.masked_fill(pad_mask.unsqueeze(-1), 0)


# Chord branch: Conformer.
class _HalfStepFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.input = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = _clear_pad(self.norm(_clear_pad(x, pad_mask)), pad_mask)
        x = _clear_pad(self.input(x), pad_mask)
        x = _clear_pad(self.activation(x), pad_mask)
        x = _clear_pad(self.dropout(x), pad_mask)
        return _clear_pad(self.output(x), pad_mask)


class _GlobalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.attention_query_chunk_size: int | None = None
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        h = _clear_pad(self.norm(_clear_pad(x, pad_mask)), pad_mask)
        qkv = self.qkv(h).reshape(
            batch_size, time_steps, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        key_mask = torch.zeros(
            batch_size, 1, 1, time_steps, device=x.device, dtype=x.dtype
        ).masked_fill(pad_mask[:, None, None, :], float("-inf"))

        def attend(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        chunk_size = self.attention_query_chunk_size
        if chunk_size is not None and time_steps > chunk_size:
            chunks = []
            for start in range(0, time_steps, chunk_size):
                query = q[..., start : start + chunk_size, :]
                if self.training and torch.is_grad_enabled():
                    output = checkpoint(
                        attend, query, k, v, key_mask, use_reentrant=False
                    )
                else:
                    output = attend(query, k, v, key_mask)
                chunks.append(output)
            o = torch.cat(chunks, dim=-2)
        else:
            o = attend(q, k, v, key_mask)
        o = o.permute(0, 2, 1, 3).reshape(batch_size, time_steps, -1)
        o = _clear_pad(self.proj(_clear_pad(o, pad_mask)), pad_mask)
        return _clear_pad(self.dropout(o), pad_mask)


class _ConformerConvolution(nn.Module):

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = _clear_pad(self.norm(_clear_pad(x, pad_mask)), pad_mask)
        h = _clear_pad(self.pointwise_in(h.transpose(1, 2)).transpose(1, 2), pad_mask)
        h = _clear_pad(F.glu(h, dim=-1), pad_mask)
        h = _clear_pad(self.depthwise(h.transpose(1, 2)).transpose(1, 2), pad_mask)
        h = _clear_pad(self.channel_norm(h), pad_mask)
        h = _clear_pad(self.activation(h), pad_mask)
        h = _clear_pad(self.pointwise_out(h.transpose(1, 2)).transpose(1, 2), pad_mask)
        return _clear_pad(self.dropout(h), pad_mask)


class _ChordConformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        conv_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ff_first = _HalfStepFeedForward(d_model, d_ff, dropout)
        self.attention = _GlobalSelfAttention(d_model, n_heads, dropout)
        self.convolution = _ConformerConvolution(d_model, conv_kernel_size, dropout)
        self.ff_last = _HalfStepFeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    @property
    def attention_query_chunk_size(self) -> int | None:
        return self.attention.attention_query_chunk_size

    @attention_query_chunk_size.setter
    def attention_query_chunk_size(self, chunk_size: int | None) -> None:
        self.attention.attention_query_chunk_size = chunk_size

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = _clear_pad(x, pad_mask)
        x = _clear_pad(x + 0.5 * self.ff_first(x, pad_mask), pad_mask)
        x = _clear_pad(x + self.attention(x, pad_mask), pad_mask)
        x = _clear_pad(x + self.convolution(x, pad_mask), pad_mask)
        x = _clear_pad(x + 0.5 * self.ff_last(x, pad_mask), pad_mask)
        return _clear_pad(self.norm(x), pad_mask)


class ChordConformerEncoder(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_layers <= 0 or n_heads <= 0 or d_ff <= 0:
            raise ValueError("Conformer 的宽度、层数、头数和前馈宽度必须为正")
        if d_model % n_heads:
            raise ValueError("Conformer 的模型宽度必须能被注意力头数整除")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("Conformer 局部卷积核必须为正奇数")
        if not 0 <= dropout < 1:
            raise ValueError("Conformer dropout 必须在 [0, 1) 内")
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            _ChordConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(n_layers)
        )

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("注意力 query 分块长度必须为正或 None")
        for block in self.blocks:
            block.attention_query_chunk_size = chunk_size

    def forward(
        self, coarse: torch.Tensor, coarse_pad_mask: torch.Tensor
    ) -> torch.Tensor:

        if coarse.ndim != 3 or coarse.shape[-1] != self.d_model:
            raise ValueError("Chord Conformer 输入必须为 B×T_coarse×d_model")
        if coarse.shape[1] == 0:
            raise ValueError("Chord Conformer 粗时间轴不能为空")
        if (
            coarse_pad_mask.shape != coarse.shape[:2]
            or coarse_pad_mask.dtype != torch.bool
        ):
            raise ValueError("Chord Conformer padding 掩码必须为 B×T_coarse bool")
        x = _clear_pad(coarse, coarse_pad_mask)
        for block in self.blocks:
            x = block(x, coarse_pad_mask)
        return _clear_pad(x, coarse_pad_mask)


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数，收到 {value!r}")


# Section branch: dilated neighborhood attention.
class NeighborhoodAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("dim", dim),
            ("n_heads", n_heads),
            ("kernel_size", kernel_size),
            ("dilation", dilation),
            ("ffn_dim", ffn_dim),
        ):
            _positive_integer(name, value)
        if dim % n_heads != 0:
            raise ValueError(f"dim 必须能被 n_heads 整除，收到 {dim} 和 {n_heads}")
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size 必须为奇数，收到 {kernel_size}")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError(f"dropout 必须在 [0, 1) 内，收到 {dropout!r}")

        self.dim = dim
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = (dim // n_heads) ** -0.5
        self.query_chunk_size: int | None = 256
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.output_projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        if chunk_size is not None:
            _positive_integer("chunk_size", chunk_size)
        self.query_chunk_size = chunk_size

    def _validate_input(self, x: torch.Tensor, pad_mask: torch.Tensor) -> list[int]:
        if x.ndim != 3 or x.shape[-1] != self.dim or x.shape[1] < 1:
            raise ValueError(f"x 必须为非空 B×T×{self.dim} 浮点张量")
        if not torch.is_floating_point(x):
            raise ValueError("x 必须为浮点张量")
        if (
            pad_mask.ndim != 2
            or pad_mask.shape != x.shape[:2]
            or pad_mask.dtype != torch.bool
        ):
            raise ValueError("pad_mask 必须为 B×T 的布尔张量，True 表示右侧 padding")
        if pad_mask.device != x.device:
            raise ValueError("pad_mask 与 x 必须位于同一设备")
        if torch.any(pad_mask[:, :-1] & ~pad_mask[:, 1:]):
            raise ValueError("pad_mask 只能标记连续的右侧 padding")
        lengths = (~pad_mask).sum(dim=1).tolist()
        if any(length < 1 for length in lengths):
            raise ValueError("每条序列至少要有一个有效位置")
        return lengths

    def _attention(self, normalized: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        batch_size, time_steps, dim = normalized.shape
        head_dim = dim // self.n_heads
        qkv = self.qkv(normalized).reshape(
            batch_size, time_steps, 3, self.n_heads, head_dim
        )
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        chunk_size = self.query_chunk_size or time_steps
        batch_rows: list[torch.Tensor] = []

        for batch_index, length in enumerate(lengths):
            width = min(self.kernel_size, length)
            spacing = min(self.dilation, max(1, length // width))
            offsets = torch.arange(width, device=normalized.device)
            row_chunks: list[torch.Tensor] = []
            for start in range(0, time_steps, chunk_size):
                end = min(start + chunk_size, time_steps)
                valid_end = min(end, length)
                if valid_end > start:
                    query_positions = torch.arange(
                        start, valid_end, device=normalized.device
                    )
                    left_min = torch.clamp(
                        width - 1 - (length - 1 - query_positions) // spacing,
                        min=0,
                    )
                    left_max = torch.clamp(query_positions // spacing, max=width - 1)
                    left = torch.minimum(
                        torch.maximum(torch.full_like(left_min, width // 2), left_min),
                        left_max,
                    )
                    neighbors = (
                        query_positions[:, None]
                        + (offsets[None, :] - left[:, None]) * spacing
                    )
                    queries = q[batch_index, :, start:valid_end, :]
                    keys = k[batch_index, :, neighbors, :]
                    values = v[batch_index, :, neighbors, :]
                    score_dtype = (
                        torch.float64
                        if queries.dtype == torch.float64
                        else torch.float32
                    )
                    scores = (
                        queries.to(score_dtype).unsqueeze(-2) * keys.to(score_dtype)
                    ).sum(-1) * self.scale
                    probabilities = torch.softmax(scores.float(), dim=-1).to(
                        values.dtype
                    )
                    attended = (probabilities.unsqueeze(-1) * values).sum(-2)
                    row_chunks.append(attended)
                padded_start = max(start, valid_end)
                if padded_start < end:
                    row_chunks.append(
                        q[batch_index, :, padded_start:end, :].new_zeros(
                            self.n_heads, end - padded_start, head_dim
                        )
                    )
            batch_rows.append(torch.cat(row_chunks, dim=1))

        return (
            torch.stack(batch_rows, dim=0)
            .transpose(1, 2)
            .reshape(batch_size, time_steps, dim)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        lengths = self._validate_input(x, pad_mask)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0)
        normalized = self.attention_norm(x).masked_fill(pad_mask.unsqueeze(-1), 0)
        attended = self._attention(normalized, lengths)
        attended = self.output_projection(attended).masked_fill(
            pad_mask.unsqueeze(-1), 0
        )
        x = (x + self.attention_dropout(attended)).masked_fill(
            pad_mask.unsqueeze(-1), 0
        )
        feed = self.feed_forward(self.feed_forward_norm(x)).masked_fill(
            pad_mask.unsqueeze(-1), 0
        )
        return (x + self.feed_forward_dropout(feed)).masked_fill(
            pad_mask.unsqueeze(-1), 0
        )


class SectionDinaTemporalEncoder(nn.Module):

    def __init__(
        self,
        *,
        dilations: tuple[int, ...] = (1, 4, 16, 64),
        n_heads: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
    ) -> None:
        super().__init__()
        if not isinstance(dilations, (tuple, list)) or not dilations:
            raise ValueError("dilations 必须为非空正整数序列")
        self.position_encoding = SinusoidalPE()
        self.blocks = nn.ModuleList(
            NeighborhoodAttentionBlock(
                dim=256,
                n_heads=n_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                ffn_dim=ffn_dim,
            )
            for dilation in dilations
        )
        self.norm = nn.LayerNorm(256)

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            block.set_attention_query_chunk_size(chunk_size)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 256 or x.shape[1] < 1:
            raise ValueError("x 必须为非空 B×Td×256 粗网格张量")
        if not torch.is_floating_point(x):
            raise ValueError("x 必须为浮点张量")
        if (
            pad_mask.shape != x.shape[:2]
            or pad_mask.dtype != torch.bool
            or pad_mask.device != x.device
        ):
            raise ValueError("pad_mask 必须为同设备 B×Td 布尔张量")
        if torch.any(pad_mask[:, :-1] & ~pad_mask[:, 1:]) or torch.any(pad_mask[:, 0]):
            raise ValueError("pad_mask 必须只标记右侧 padding，且每条序列至少一帧有效")
        x = self.position_encoding(x.masked_fill(pad_mask.unsqueeze(-1), 0))
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.norm(x).masked_fill(pad_mask.unsqueeze(-1), 0)


# Section boundary head: left-right context comparison.
class SectionBoundaryContextHead(nn.Module):

    def __init__(
        self,
        *,
        input_dim: int = 256,
        hidden_dim: int = 256,
        windows_native: tuple[int, ...] = (4, 16, 32),
        native_stride: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim < 1
        ):
            raise ValueError("input_dim 必须为正整数")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
        ):
            raise ValueError("hidden_dim 必须为正整数")
        if (
            isinstance(native_stride, bool)
            or not isinstance(native_stride, int)
            or native_stride < 1
        ):
            raise ValueError("native_stride 必须为正整数")
        if (
            not isinstance(windows_native, (tuple, list))
            or not windows_native
            or any(
                isinstance(w, bool) or not isinstance(w, int) or w < 1
                for w in windows_native
            )
            or len(set(windows_native)) != len(windows_native)
        ):
            raise ValueError("windows_native 必须为不重复的正整数序列")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout 必须在 [0,1) 范围内")

        self.input_dim = input_dim
        self.windows_native = tuple(windows_native)
        self.native_stride = native_stride
        feature_dim = input_dim + len(self.windows_native) * (3 * input_dim + 3)
        self.fusion = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def context_features(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim or x.shape[1] < 1:
            raise ValueError(f"x 必须为非空 B×T×{self.input_dim} 张量")
        if not torch.is_floating_point(x):
            raise ValueError("x 必须为浮点张量")
        if (
            pad_mask.shape != x.shape[:2]
            or pad_mask.dtype != torch.bool
            or pad_mask.device != x.device
        ):
            raise ValueError("pad_mask 必须为同设备 B×T 布尔张量")

        valid = ~pad_mask
        masked_x = x.masked_fill(pad_mask.unsqueeze(-1), 0)
        accumulator = (
            masked_x.float() if x.dtype in (torch.float16, torch.bfloat16) else masked_x
        )
        sums = F.pad(accumulator.cumsum(dim=1), (0, 0, 1, 0))
        counts = F.pad(valid.to(accumulator.dtype).cumsum(dim=1), (1, 0))
        time = x.shape[1]
        anchors = torch.arange(time, device=x.device)
        groups: list[torch.Tensor] = [masked_x]
        for native_width in self.windows_native:
            width = native_width * self.native_stride
            left_start = (anchors - width).clamp_min(0)
            right_end = (anchors + width).clamp_max(time)
            left_sum = sums.index_select(1, anchors) - sums.index_select(1, left_start)
            right_sum = sums.index_select(1, right_end) - sums.index_select(1, anchors)
            left_count = counts.index_select(1, anchors) - counts.index_select(
                1, left_start
            )
            right_count = counts.index_select(1, right_end) - counts.index_select(
                1, anchors
            )
            left = left_sum / left_count.clamp_min(1).unsqueeze(-1)
            right = right_sum / right_count.clamp_min(1).unsqueeze(-1)
            cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-6).unsqueeze(-1)
            left_exists = (left_count > 0).unsqueeze(-1).to(accumulator.dtype)
            right_exists = (right_count > 0).unsqueeze(-1).to(accumulator.dtype)
            groups.extend(
                (left, right, (left - right).abs(), cosine, left_exists, right_exists)
            )
        return (
            torch.cat(groups, dim=-1).to(x.dtype).masked_fill(pad_mask.unsqueeze(-1), 0)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        features = self.context_features(x, pad_mask)
        hidden = self.dropout(self.activation(self.fusion(features)))
        return self.output(hidden).masked_fill(pad_mask.unsqueeze(-1), 0)


# Phrase and beat branches: temporal convolutions.
class TemporalConvDecoder(nn.Module):

    def __init__(self, dilations: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        self.input_adapter = nn.Identity()
        self.blocks = nn.ModuleList(
            [SameRateDecoderBlock(256, dilation, 0.1) for dilation in dilations]
        )
        self.norm = nn.LayerNorm(256)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = _zero_padding(self.input_adapter(_zero_padding(x, pad_mask)), pad_mask)
        for block in self.blocks:
            x = _zero_padding(block(x, pad_mask), pad_mask)
        return _zero_padding(self.norm(x), pad_mask)


# Chord and section downsampling and grid restoration.
class DownsampleTransformerDecoder(nn.Module):

    def __init__(self, task_name: str) -> None:
        super().__init__()
        if task_name not in {"chord", "section"}:
            raise ValueError(f"未知私有任务: {task_name}")
        stride_blocks = 2 if task_name == "chord" else 3
        self.input_adapter = nn.Identity()
        self.downsample_blocks = nn.ModuleList(
            StrideDepthwiseSeparableBlock(channels=256) for _ in range(stride_blocks)
        )
        self.position_encoding = SinusoidalPE()
        if task_name == "section":
            self.dina_encoder = SectionDinaTemporalEncoder()
        else:
            self.conformer = ChordConformerEncoder()
        self.norm = nn.LayerNorm(256)
        self.scale = 2**stride_blocks

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, original_time: int
    ) -> torch.Tensor:
        original_mask = _pad_mask_from_lengths(lengths, x.shape[1])
        x = _zero_padding(
            self.input_adapter(_zero_padding(x, original_mask)), original_mask
        )
        current_lengths = lengths
        current_mask = original_mask
        for block in self.downsample_blocks:
            x = _zero_padding(x, current_mask)
            current_lengths = (current_lengths + 1) // 2
            current_mask = _pad_mask_from_lengths(
                current_lengths, (x.shape[1] + 1) // 2
            )
            x = _zero_padding(block(x, current_mask), current_mask)
        coarse_mask = _pad_mask_from_lengths(current_lengths, x.shape[1])
        if hasattr(self, "dina_encoder"):
            x = _zero_padding(self.dina_encoder(x, coarse_mask), coarse_mask)
        else:
            x = _zero_padding(
                self.position_encoding(_zero_padding(x, coarse_mask)), coarse_mask
            )
        if hasattr(self, "conformer"):
            x = _zero_padding(self.conformer(x, coarse_mask), coarse_mask)
        x = _zero_padding(self.norm(x), coarse_mask)
        restored = x.repeat_interleave(self.scale, dim=1)[:, :original_time]
        original_mask = _pad_mask_from_lengths(lengths, original_time)
        restored = _zero_padding(restored, original_mask)
        return restored


# Task prediction heads.
class LightDecoderHeads(nn.Module):

    def __init__(self, task_random_streams: TaskRandomStreams | None = None) -> None:
        super().__init__()

        def make_head(task: str, output_dim: int) -> MaskedDecoderHead:
            scope = (
                task_random_streams.initialization(task)
                if task_random_streams is not None
                else nullcontext()
            )
            with scope:
                return MaskedDecoderHead(256, output_dim)

        self.phrase_head = make_head("phrase", 1)
        self.section_function_head = make_head("section", 9)
        scope = (
            task_random_streams.initialization("section")
            if task_random_streams is not None
            else nullcontext()
        )
        with scope:
            self.section_boundary_context_head = SectionBoundaryContextHead()
        self.chord_root_head = make_head("chord", 12)
        self.chord_quality_head = make_head("chord", 16)
        self.tactus_head = make_head("beat", 1)
        self.downbeat_head = make_head("beat", 1)

    def forward_task(
        self, task: str, features: Mapping[str, torch.Tensor], pad_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        if task == "phrase":
            return {"phrase_logits": self.phrase_head(features["phrase"], pad_mask)}
        if task == "beat":
            tactus = self.tactus_head(features["beat"], pad_mask)
            downbeat = self.downbeat_head(features["beat"], pad_mask)
            return {
                "tactus_logits": tactus,
                "downbeat_logits": downbeat,
                # Enhanced tactus head: beat + downbeat.
                "tactus_enhanced_logits": _zero_padding(tactus + downbeat, pad_mask),
            }
        if task == "chord":
            return {
                "chord_root_logits": self.chord_root_head(features["chord"], pad_mask),
                "chord_quality_logits": self.chord_quality_head(
                    features["chord"], pad_mask
                ),
            }
        if task == "section":
            return {
                "section_logits": self.section_function_head(
                    features["section"], pad_mask
                ),
                "section_boundary_logits": self._section_boundary_logits(
                    features, pad_mask
                ),
            }
        raise ValueError(f"未知任务: {task}")

    def _section_boundary_logits(
        self, features: Mapping[str, torch.Tensor], pad_mask: torch.Tensor
    ) -> torch.Tensor:

        return self.section_boundary_context_head(features["section"], pad_mask)

    def forward(
        self, features: Mapping[str, torch.Tensor], pad_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        out["phrase_logits"] = self.phrase_head(features["phrase"], pad_mask)
        out["section_logits"] = self.section_function_head(
            features["section"], pad_mask
        )
        out["section_boundary_logits"] = self._section_boundary_logits(
            features, pad_mask
        )
        out["chord_root_logits"] = self.chord_root_head(features["chord"], pad_mask)
        out["chord_quality_logits"] = self.chord_quality_head(
            features["chord"], pad_mask
        )
        tactus = self.tactus_head(features["beat"], pad_mask)
        downbeat = self.downbeat_head(features["beat"], pad_mask)
        out["tactus_logits"] = tactus
        out["downbeat_logits"] = downbeat
        # Enhanced tactus head: beat + downbeat.
        out["tactus_enhanced_logits"] = _zero_padding(tactus + downbeat, pad_mask)
        return out
