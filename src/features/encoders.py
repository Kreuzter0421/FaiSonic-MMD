"""Load MuQ and MusicFM and extract hidden features."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class MuQProvider:
    def __init__(
        self,
        *,
        model: Callable[..., Any],
        layer: int,
        to_batch: Callable[[np.ndarray], Any],
        to_numpy: Callable[[Any], np.ndarray],
    ) -> None:
        self._model = model
        self._layer = layer
        self._to_batch = to_batch
        self._to_numpy = to_numpy

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        output = self._model(
            self._to_batch(np.asarray(waveform, dtype=np.float32)),
            output_hidden_states=True,
        )
        return np.asarray(
            self._to_numpy(output.hidden_states[self._layer])[0], dtype=np.float32
        )


def load_muq_provider(
    *, model_name: str, device: str, dtype: str, layer: int
) -> MuQProvider:
    import torch
    from muq import MuQ

    model = MuQ.from_pretrained(model_name).to(
        device=device, dtype=getattr(torch, dtype)
    )
    model.eval()

    def to_batch(waveform: np.ndarray) -> Any:
        return torch.from_numpy(waveform[None, :]).to(
            device=device, dtype=getattr(torch, dtype)
        )

    def to_numpy(tensor: Any) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def invoke(batch: Any, *, output_hidden_states: bool) -> Any:
        with torch.inference_mode():
            return model(batch, output_hidden_states=output_hidden_states)

    return MuQProvider(model=invoke, layer=layer, to_batch=to_batch, to_numpy=to_numpy)


def _to_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    on_cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    raw = on_cpu.numpy() if hasattr(on_cpu, "numpy") else on_cpu
    return np.asarray(raw)


def _normalize_musicfm_latent(latent: Any) -> np.ndarray:
    return np.asarray(_to_numpy(latent)[0], dtype=np.float32)


def build_musicfm_provider(
    model: Any, *, layer: int, device: str, dtype: str, torch_module: Any
) -> Callable[[np.ndarray], np.ndarray]:
    torch_dtype = getattr(torch_module, dtype)

    def provider(waveform: np.ndarray) -> np.ndarray:
        wav = (
            torch_module.from_numpy(
                np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
            )
            .unsqueeze(0)
            .to(device=device, dtype=torch_dtype)
        )
        with torch_module.no_grad():
            return _normalize_musicfm_latent(model.get_latent(wav, layer_ix=layer))

    return provider


def load_musicfm_provider(
    *,
    model_path: Path | str,
    stats_path: Path | str,
    layer: int,
    device: str,
    dtype: str,
) -> Callable[[np.ndarray], np.ndarray]:
    import torch
    from model.musicfm_25hz import MusicFM25Hz

    model = MusicFM25Hz(
        is_flash=False, stat_path=str(stats_path), model_path=str(model_path)
    )
    model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
    return build_musicfm_provider(
        model, layer=layer, device=device, dtype=dtype, torch_module=torch
    )
