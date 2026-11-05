"""Define the fixed four-task paper model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from src.models.branches import (
    DownsampleTransformerDecoder,
    LightDecoderHeads,
    TemporalConvDecoder,
)
from src.models.layers import SharedLightweightFrontEnd, TaskGate, _zero_padding

_LABELS = {"shared": 1, "phrase": 2, "beat": 3, "chord": 4, "section": 5}
_MODULUS = 2**63


def derived_seed(seed: int, label: str, purpose: str) -> int:

    if label not in _LABELS:
        raise ValueError(f"未知随机流标签: {label}")
    if purpose not in ("init", "train", "train_loader", "val_loader"):
        raise ValueError(f"未知随机流用途: {purpose}")
    purpose_id = {"init": 1, "train": 2, "train_loader": 3, "val_loader": 4}[purpose]
    return (int(seed) * 101 + purpose_id * 10 + _LABELS[label]) % _MODULUS


# Independent task random streams.
class TaskRandomStreams:

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._init_cpu = {
            label: torch.Generator(device="cpu")
            .manual_seed(derived_seed(seed, label, "init"))
            .get_state()
            for label in _LABELS
        }
        self._train_cpu = {
            label: torch.Generator(device="cpu")
            .manual_seed(derived_seed(seed, label, "train"))
            .get_state()
            for label in _LABELS
        }
        self._train_cuda: dict[str, dict[int, torch.Tensor]] = {
            label: {} for label in _LABELS
        }

    @contextmanager
    def initialization(self, label: str) -> Iterator[None]:

        if label not in _LABELS:
            raise ValueError(f"未知随机流标签: {label}")
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._init_cpu[label])
            try:
                yield
            finally:
                self._init_cpu[label] = torch.get_rng_state()

    @contextmanager
    def training(self, label: str, *, advance: bool) -> Iterator[None]:

        if label not in _LABELS:
            raise ValueError(f"未知随机流标签: {label}")
        devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._train_cpu[label])
            for device in devices:
                state = self._train_cuda[label].get(device)
                if state is None:
                    state = (
                        torch.Generator(device=f"cuda:{device}")
                        .manual_seed(derived_seed(self.seed, label, "train"))
                        .get_state()
                    )
                torch.cuda.set_rng_state(state, device=device)
            try:
                yield
            finally:
                if advance:
                    self._train_cpu[label] = torch.get_rng_state()
                    for device in devices:
                        self._train_cuda[label][device] = torch.cuda.get_rng_state(
                            device
                        )

    def state_dict(self) -> dict[str, Any]:

        return {
            "seed": self.seed,
            "init_cpu": self._init_cpu,
            "train_cpu": self._train_cpu,
            "train_cuda": self._train_cuda,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("续训总 seed 与 checkpoint 不一致")
        if set(state["train_cpu"]) != set(_LABELS):
            raise ValueError("checkpoint 任务随机流不完整")
        self._init_cpu = {
            name: tensor.clone() for name, tensor in state["init_cpu"].items()
        }
        self._train_cpu = {
            name: tensor.clone() for name, tensor in state["train_cpu"].items()
        }
        self._train_cuda = {
            name: {int(device): tensor.clone() for device, tensor in states.items()}
            for name, states in state["train_cuda"].items()
        }


class MultiTaskNetwork(nn.Module):

    def __init__(self, task_seed: int | None = None) -> None:
        super().__init__()
        self.task_random_streams = (
            TaskRandomStreams(task_seed) if task_seed is not None else None
        )
        with self._initialization_scope("shared"):
            # Shared encoder for fused MuQ and MusicFM features.
            self.frontend = SharedLightweightFrontEnd(2048)
        # Separate gates for the four task branches.
        self.task_gates = nn.ModuleDict()
        for task in ("phrase", "beat", "chord", "section"):
            with self._initialization_scope(task):
                self.task_gates[task] = TaskGate()
        with self._initialization_scope("phrase"):
            self.phrase_decoder = TemporalConvDecoder((1, 4, 16, 64))
        with self._initialization_scope("beat"):
            self.beat_decoder = TemporalConvDecoder((1, 2, 8, 32))
        with self._initialization_scope("chord"):
            self.chord_decoder = DownsampleTransformerDecoder("chord")
        with self._initialization_scope("section"):
            self.section_decoder = DownsampleTransformerDecoder("section")
        self.heads = LightDecoderHeads(task_random_streams=self.task_random_streams)

    def _initialization_scope(self, task: str):
        if self.task_random_streams is None:
            return nullcontext()
        return self.task_random_streams.initialization(task)

    def _training_scope(self, task: str):
        if self.task_random_streams is None:
            return nullcontext()
        return self.task_random_streams.training(task, advance=self.training)

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        self.chord_decoder.conformer.set_attention_query_chunk_size(chunk_size)
        self.section_decoder.dina_encoder.set_attention_query_chunk_size(chunk_size)

    def _task_input(
        self,
        task: str,
        shared: torch.Tensor,
        lengths: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:

        return self.task_gates[task](shared, lengths, pad_mask)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["feature"]
        lengths, pad_mask = (batch["lengths"], batch["pad_mask"])
        if (
            lengths.ndim != 1
            or torch.any(lengths <= 0)
            or torch.any(lengths > features.shape[1])
        ):
            raise ValueError("每首歌曲的真实长度必须在 1 到输入时间帧数之间")
        features = _zero_padding(features, pad_mask)
        with self._training_scope("shared"):
            shared = self.frontend(features, pad_mask)
        outputs: dict[str, torch.Tensor] = {}
        task_logits: dict[str, torch.Tensor] = {}
        # Phrase branch: TCN.
        with self._training_scope("phrase"):
            phrase_input = self._task_input("phrase", shared, lengths, pad_mask)
            phrase = self.phrase_decoder(phrase_input, pad_mask)
            outputs["phrase"] = phrase
            if self.task_random_streams is not None:
                task_logits.update(self.heads.forward_task("phrase", outputs, pad_mask))
        # Beat branch: TCN.
        with self._training_scope("beat"):
            beat_input = self._task_input("beat", shared, lengths, pad_mask)
            beat = self.beat_decoder(beat_input, pad_mask)
            outputs["beat"] = beat
            if self.task_random_streams is not None:
                task_logits.update(self.heads.forward_task("beat", outputs, pad_mask))
        # Chord branch: Conformer.
        with self._training_scope("chord"):
            chord_input = self._task_input("chord", shared, lengths, pad_mask)
            outputs["chord"] = self.chord_decoder(
                chord_input, lengths, chord_input.shape[1]
            )
            if self.task_random_streams is not None:
                task_logits.update(self.heads.forward_task("chord", outputs, pad_mask))
        # Section branch: DiNA.
        with self._training_scope("section"):
            section_input = self._task_input("section", shared, lengths, pad_mask)
            section = self.section_decoder(
                section_input, lengths, section_input.shape[1]
            )
            outputs["section"] = section
            if self.task_random_streams is not None:
                task_logits.update(
                    self.heads.forward_task("section", outputs, pad_mask)
                )
        return (
            task_logits
            if self.task_random_streams is not None
            else self.heads(outputs, pad_mask)
        )


class Model(nn.Module):

    def __init__(self, task_seed: int | None = None) -> None:
        super().__init__()
        self.n_section = 9
        self.n_chord_root = 12
        self.n_chord_quality = 16
        self.frozen_mert_light_decoders = MultiTaskNetwork(task_seed=task_seed)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.frozen_mert_light_decoders(batch)

    def set_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        self.frozen_mert_light_decoders.set_attention_query_chunk_size(chunk_size)


def build_model(task_seed: int | None = None) -> Model:
    return Model(task_seed=task_seed)
