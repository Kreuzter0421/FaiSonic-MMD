"""Collect model logits from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_spec(root: Path, args) -> dict:
    train_path = root / args.config
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    model_path = train_path.parent / train.get("model_config", "model.yaml")
    data_path = train_path.parent / train.get("data_config", "data.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    run_id = args.run_id or train["run_id"]
    epoch = args.epoch if args.epoch is not None else int(train["epochs"])
    if epoch < 1:
        raise ValueError("轮次必须为正整数")
    checkpoint = args.checkpoint or str(
        Path(train["checkpoint_dir"]) / run_id / f"epoch_{epoch}.pt"
    )

    def project_path(path):
        path = (root / path).resolve()
        return (
            str(path.relative_to(root.resolve()))
            if path.is_relative_to(root.resolve())
            else str(path)
        )

    return {
        "run_id": run_id,
        "epoch": epoch,
        "checkpoint": project_path(checkpoint),
        "model_config": project_path(model_path),
        "data_config": project_path(data_path),
        "split": project_path(
            train.get("split_json", str(Path(data["sample_dir"]) / "split.json"))
        ),
        "labels": project_path(Path(data["sample_dir"]) / "labels"),
    }


def collect_logits(args) -> None:
    variant = "paper"
    run_id = args.run_id
    epoch = args.epoch
    out = args.out

    import torch

    from src import TASK_HEADS
    from src.data.dataset import FeatureDataset, collate_fn, to_torch_batch
    from src.models.model import build_model

    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.song_ids is None:
        split = json.loads((ROOT / args.split).read_text(encoding="utf-8"))
        song_ids = sorted(split[args.partition])
    else:
        song_ids = sorted(args.song_ids)
    if not song_ids:
        raise ValueError("推理曲目列表为空")
    data_config = yaml.safe_load((ROOT / args.data_config).read_text(encoding="utf-8"))
    data_config["project_root"] = str(ROOT)
    dataset = FeatureDataset(song_ids, data_config)
    model = build_model().cuda()
    checkpoint = torch.load(
        ROOT / args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    output = ROOT / out
    output.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(dataset, start=1):
        path = output / f"{item['song_id']}.npz"
        if path.exists():
            continue
        batch = to_torch_batch(collate_fn([item]))
        batch = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            logits = model(batch)
        arrays = {
            key: value[0, : item["length"]].float().cpu().numpy().astype(np.float32)
            for key, value in logits.items()
        }
        np.savez_compressed(path, **arrays)
        if index == 1 or index % 10 == 0:
            print(
                f"{variant} E{epoch} {args.partition} {index}/{len(dataset)}",
                flush=True,
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "variant": variant,
                "epoch": epoch,
                "checkpoint": args.checkpoint,
                "data_config": args.data_config,
                "active_concepts": sorted(TASK_HEADS),
                "split": f"{args.split}#{args.partition}",
                "song_ids": song_ids,
                "song_count": len(song_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
