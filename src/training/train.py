"""Train, validate, and manage optimizer and checkpoint state."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

if TYPE_CHECKING:
    from src.models.model import TaskRandomStreams


def configured_cosine_total_steps(train_config: Mapping[str, Any]) -> int:
    return int(train_config["cosine_total_steps"])


def cosine_lr_multiplier(
    step: int, warmup_steps: int, total_steps: int, *, min_lr_ratio: float = 0.0
) -> float:
    if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("余弦最低学习率比例必须在0到1之间")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:

    def scale(step: int) -> float:
        return cosine_lr_multiplier(
            step, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


_LOGICAL_GROUP_ORDER = (
    "mert_interface",
    "shared_backbone",
    "phrase",
    "beat",
    "chord_encoder",
    "chord_heads",
    "section",
)


def _logical_group_for_parameter(name: str) -> str:
    prefix = "frozen_mert_light_decoders."
    for task, logical_group in (
        ("phrase", "phrase"),
        ("beat", "beat"),
        ("chord", "chord_encoder"),
        ("section", "section"),
    ):
        if name.startswith(f"{prefix}task_gates.{task}."):
            return logical_group
    if name.startswith(f"{prefix}frontend.rms_norm."):
        return "mert_interface"
    if name.startswith(f"{prefix}frontend."):
        return "shared_backbone"
    if name.startswith(f"{prefix}phrase_decoder.") or name.startswith(
        f"{prefix}heads.phrase_head."
    ):
        return "phrase"
    if (
        name.startswith(f"{prefix}beat_decoder.")
        or name.startswith(f"{prefix}heads.tactus_head.")
        or name.startswith(f"{prefix}heads.downbeat_head.")
    ):
        return "beat"
    if name.startswith(f"{prefix}chord_decoder."):
        return "chord_encoder"
    if name.startswith(f"{prefix}heads.chord_root_head.") or name.startswith(
        f"{prefix}heads.chord_quality_head."
    ):
        return "chord_heads"
    if (
        name.startswith(f"{prefix}section_decoder.")
        or name.startswith(f"{prefix}heads.section_function_head.")
        or name.startswith(f"{prefix}heads.section_boundary_context_head.")
    ):
        return "section"
    raise ValueError(f"未识别的模型参数归组: {name}")


def _module_type(name: str, modules: Mapping[str, nn.Module]) -> str:
    parts = name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module = modules.get(".".join(parts[:index]))
        if module is not None:
            return type(module).__name__
    return ""


def build_optimizer_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    named_modules: Iterable[tuple[str, nn.Module]],
    *,
    lr: float,
    weight_decay: float,
    group_settings: Mapping[str, Any],
    loss_parameters: Iterable[nn.Parameter] = (),
) -> list[dict[str, Any]]:
    modules = dict(named_modules)
    aliases: dict[int, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        aliases.setdefault(id(parameter), []).append((name, parameter))
    group_order = _LOGICAL_GROUP_ORDER
    group_for_parameter = _logical_group_for_parameter
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {
        name: [] for name in group_order
    }
    for entries in aliases.values():
        name, parameter = min(entries, key=lambda item: item[0])
        grouped[group_for_parameter(name)].append((name, parameter))
    groups: list[dict[str, Any]] = []
    for logical_name in group_order:
        settings = group_settings.get(logical_name, {})
        group_lr = float(settings.get("lr", lr))
        group_weight_decay = float(settings.get("weight_decay", weight_decay))
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in sorted(grouped[logical_name], key=lambda item: item[0]):
            leaf = name.rsplit(".", 1)[-1]
            module_type = _module_type(name, modules)
            use_no_decay = (
                logical_name == "mert_interface"
                or leaf == "bias"
                or leaf.endswith("_bias")
                or (module_type == "LayerNorm")
                or module_type.endswith("RMSNorm")
            )
            (no_decay if use_no_decay else decay).append(parameter)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": group_lr,
                    "weight_decay": group_weight_decay,
                    "name": f"{logical_name}.decay",
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": group_lr,
                    "weight_decay": 0.0,
                    "name": f"{logical_name}.no_decay",
                }
            )
    trainable_loss_parameters = [
        parameter for parameter in loss_parameters if parameter.requires_grad
    ]
    if trainable_loss_parameters:
        groups.append(
            {
                "params": trainable_loss_parameters,
                "lr": lr,
                "weight_decay": 0.0,
                "name": "loss",
            }
        )
    return groups


def optimizer_step(
    *,
    total_loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    parameters: Iterable[nn.Parameter],
    grad_clip: float,
    clip_enabled: bool = True,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    if clip_enabled:
        torch.nn.utils.clip_grad_norm_(tuple(parameters), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


@dataclass(frozen=True)
class ResumeState:

    next_epoch: int
    global_step: int
    best_metric: float
    best_epoch: int | None
    resume_source: str


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_checkpoint_payload(
    *,
    checkpoint_role: str,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    next_epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int | None,
    resume_source: str | None,
    task_random_streams: TaskRandomStreams | None = None,
) -> dict[str, Any]:
    payload = {
        "checkpoint_role": checkpoint_role,
        "model_state_dict": model.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "rng_state": capture_rng_state(),
        "resume_source": resume_source,
    }
    if task_random_streams is not None:
        payload["task_random_state"] = task_random_streams.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), target)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    restore_rng: bool = True,
    task_random_streams: TaskRandomStreams | None = None,
) -> ResumeState:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if task_random_streams is not None and "task_random_state" not in payload:
        raise ValueError("Checkpoint is missing task RNG states required for exact resumption.")
    if task_random_streams is None and "task_random_state" in payload:
        raise ValueError("checkpoint 含任务随机流状态，续训配置须启用 task_randomness")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    loss_fn.load_state_dict(payload["loss_fn_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng and isinstance(payload.get("rng_state"), Mapping):
        restore_rng_state(payload["rng_state"])
    if task_random_streams is not None:
        task_random_streams.load_state_dict(payload["task_random_state"])
    return ResumeState(
        next_epoch=int(payload.get("next_epoch", 1)),
        global_step=int(payload.get("global_step", 0)),
        best_metric=float(payload.get("best_metric", float("inf"))),
        best_epoch=payload.get("best_epoch"),
        resume_source=str(checkpoint_path),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def _config_paths(
    train_path: Path, train_config: Mapping[str, Any]
) -> tuple[Path, Path]:
    config_dir = train_path.parent
    return (
        config_dir / str(train_config.get("model_config", "model.yaml")),
        config_dir / str(train_config.get("data_config", "data.yaml")),
    )


def _move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes:02d}:{secs:02d}"
    )


def _should_log_batch(batch_index: int, total_batches: int, interval: int) -> bool:
    return interval > 0 and (
        batch_index % interval == 0 or batch_index == total_batches
    )


def _print_epoch_line(prefix: str, total: float, per_head: Mapping[str, float]) -> None:
    labels = {
        "phrase": "phrase",
        "section_function": "section_function_loss",
        "section_boundary": "section_boundary_loss",
        "chord_root": "root",
        "chord_quality": "quality",
        "tactus": "tactus",
        "downbeat": "downbeat",
    }
    fields = " ".join(
        (
            f"{label}={per_head[head]:.4f}"
            for head, label in labels.items()
            if head in per_head
        )
    )
    print(f"{prefix}: total={total:.4f} | {fields}", flush=True)


def evaluate(
    model: nn.Module, loss_fn: nn.Module, loader: Any, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    from src.data.dataset import to_torch_batch

    model.eval()
    total_weighted = 0.0
    head_weighted: Dict[str, float] = defaultdict(float)
    total_samples = 0
    with torch.no_grad():
        for np_batch in loader:
            batch = _move_batch_to_device(to_torch_batch(np_batch), device)
            n = int(batch["feature"].shape[0])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(batch)
                total_loss, per_head = loss_fn(logits, batch)
            total_weighted += float(total_loss.item()) * n
            for name, value in per_head.items():
                head_weighted[name] += float(value.item()) * n
            total_samples += n
    model.train()
    denominator = max(total_samples, 1)
    return (
        total_weighted / denominator,
        {name: value / denominator for name, value in head_weighted.items()},
    )


def train(
    config_path: str | Path, resume: str | Path | None = None, run_id: str | None = None
) -> None:
    from torch.utils.data import DataLoader

    from src import INTERNAL_HEADS
    from src.data.dataset import (
        LengthBucketSampler,
        PhraseDataset,
        collate_fn,
        load_split,
        to_torch_batch,
    )
    from src.models.model import build_model, derived_seed
    from src.training.losses import MultiTaskLoss

    train_cfg_path = Path(config_path)
    train_cfg = _load_yaml(train_cfg_path)
    model_cfg_path, data_cfg_path = _config_paths(train_cfg_path, train_cfg)
    model_cfg = _load_yaml(model_cfg_path)
    data_cfg = _load_yaml(data_cfg_path)
    batch_size = int(train_cfg.get("batch_size", 1))
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    parameter_groups = train_cfg.get("parameter_groups", {})
    group_settings = parameter_groups.get("groups", {})
    clip_settings = parameter_groups.get("clip", {})
    clip_enabled = bool(clip_settings.get("enabled", True))
    task_randomness_enabled = bool(
        train_cfg.get("task_randomness", {}).get("enabled", False)
    )
    grad_clip = float(
        clip_settings.get("global_threshold", train_cfg.get("grad_clip", 1.0))
    )
    save_every = int(train_cfg.get("save_every", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    log_every_batches = int(train_cfg.get("log_every_batches", 10))
    seed = int(data_cfg.get("split", {}).get("seed", 42))
    torch.manual_seed(seed)
    active_internal_heads = INTERNAL_HEADS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[train] device={device}, active_heads={sorted(active_internal_heads)}",
        flush=True,
    )
    dataset_dir = str(data_cfg.get("sample_dir", "datasets/sample"))
    label_dir = str(Path(dataset_dir) / "labels")
    split_json = str(train_cfg.get("split_json", Path(dataset_dir) / "split.json"))
    train_paths, val_paths, _ = load_split(split_json, label_dir)
    train_ds = PhraseDataset(train_paths, data_cfg, active_internal_heads)
    val_ds = PhraseDataset(val_paths, data_cfg, active_internal_heads)
    train_sampler = LengthBucketSampler(
        train_ds.lengths, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_sampler = LengthBucketSampler(
        val_ds.lengths, batch_size=batch_size, shuffle=False, seed=seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **{
            "generator": torch.Generator().manual_seed(
                derived_seed(seed, "shared", "train_loader")
            )
        }
        if task_randomness_enabled
        else {},
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **{
            "generator": torch.Generator().manual_seed(
                derived_seed(seed, "shared", "val_loader")
            )
        }
        if task_randomness_enabled
        else {},
    )
    model = build_model(task_seed=seed if task_randomness_enabled else None).to(device)
    task_random_streams = model.frozen_mert_light_decoders.task_random_streams
    print("[train] FaiSonic MMD: MuQ + MusicFM, four-task model", flush=True)
    shift_tolerant = train_cfg.get("shift_tolerant", {})
    loss_fn = MultiTaskLoss(
        tol=int(shift_tolerant.get("tolerance", 1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(
            model.named_parameters(remove_duplicate=False),
            model.named_modules(remove_duplicate=False),
            lr=lr,
            weight_decay=weight_decay,
            group_settings=group_settings,
            loss_parameters=loss_fn.parameters(),
        )
    )
    total_steps = configured_cosine_total_steps(train_cfg)
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(train_cfg.get("cosine_min_lr_ratio", 0.0)),
    )
    run_id = run_id or train_cfg.get("run_id")
    explicit_run_id = run_id is not None
    run_id = run_id or datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%SZ")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id 必须是单个目录名")
    run_dir = Path("results") / run_id
    log_dir = Path(train_cfg.get("log_dir", "logs"))
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "model_checkpoints")) / run_id
    if (
        explicit_run_id
        and resume is not None
        and (Path(resume).resolve().parent != checkpoint_dir.resolve())
    ):
        raise ValueError("续训权重必须属于同一 run_id 的 checkpoint 目录")
    if explicit_run_id and resume is None and (checkpoint_dir / "last.pt").exists():
        raise FileExistsError("该 run_id 已有训练权重，请传 --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"train": train_cfg, "model": model_cfg, "data": data_cfg},
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    log_path = log_dir / f"{run_id}.jsonl"
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    best_epoch: int | None = None
    if resume is not None:
        restored = load_checkpoint(
            resume,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
            task_random_streams=task_random_streams,
        )
        start_epoch = restored.next_epoch
        global_step = restored.global_step
        best_metric = restored.best_metric
        best_epoch = restored.best_epoch
    print(
        f"[train] 开始训练：epochs={epochs}, batch_size={batch_size}, lr={lr}, grad_clip={('off' if not clip_enabled else grad_clip)}, task_randomness={task_randomness_enabled}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs + 1):
        train_sampler.set_epoch(epoch)
        if task_randomness_enabled:
            train_loader.generator.manual_seed(
                derived_seed(seed, "shared", "train_loader") + epoch
            )
        model.train()
        started_at = time.monotonic()
        total_weighted = 0.0
        head_weighted: Dict[str, float] = defaultdict(float)
        sample_count = 0
        for batch_index, np_batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(to_torch_batch(np_batch), device)
            n = int(batch["feature"].shape[0])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(batch)
                total_loss, per_head = loss_fn(logits, batch)
            optimizer_step(
                total_loss=total_loss,
                optimizer=optimizer,
                scheduler=scheduler,
                parameters=model.parameters(),
                grad_clip=grad_clip,
                clip_enabled=clip_enabled,
            )
            global_step += 1
            total_weighted += float(total_loss.item()) * n
            for name, value in per_head.items():
                head_weighted[name] += float(value.item()) * n
            sample_count += n
            if _should_log_batch(batch_index, len(train_loader), log_every_batches):
                average = total_weighted / max(sample_count, 1)
                elapsed = time.monotonic() - started_at
                print(
                    f"Epoch {epoch:04d} train | batch {batch_index}/{len(train_loader)} | loss={average:.4f} | 已用={_format_duration(elapsed)}",
                    flush=True,
                )
        train_total = total_weighted / max(sample_count, 1)
        train_heads = {
            name: value / max(sample_count, 1) for name, value in head_weighted.items()
        }
        val_sampler.set_epoch(epoch)
        if task_randomness_enabled:
            val_loader.generator.manual_seed(
                derived_seed(seed, "shared", "val_loader") + epoch
            )
        val_total, val_heads = evaluate(model, loss_fn, val_loader, device)
        _print_epoch_line(f"Epoch {epoch:04d} train", train_total, train_heads)
        _print_epoch_line(f"Epoch {epoch:04d} val  ", val_total, val_heads)
        current_metric = sum(
            (
                val_heads[name]
                for name in sorted(active_internal_heads)
                if name in val_heads
            )
        )
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
        payload = build_checkpoint_payload(
            checkpoint_role="last",
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            next_epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
            resume_source=None if resume is None else str(resume),
            task_random_streams=task_random_streams,
        )
        save_checkpoint(checkpoint_dir / "last.pt", payload)
        if best_epoch == epoch:
            best_payload = dict(payload)
            best_payload["checkpoint_role"] = "best"
            save_checkpoint(checkpoint_dir / "best.pt", best_payload)
        if save_every and epoch % save_every == 0:
            periodic_payload = dict(payload)
            periodic_payload["checkpoint_role"] = "periodic"
            save_checkpoint(checkpoint_dir / f"epoch_{epoch}.pt", periodic_payload)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_total_loss": train_total,
            "validation_total_loss": val_total,
            "train_losses": train_heads,
            "validation_losses": val_heads,
        }
        for name in ("section_function", "section_boundary"):
            if name in active_internal_heads:
                epoch_record[f"train_{name}_loss"] = train_heads.get(name, 0.0)
                epoch_record[f"validation_{name}_loss"] = val_heads.get(name, 0.0)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
    print(f"[train] 完成。日志 → {log_path}", flush=True)
