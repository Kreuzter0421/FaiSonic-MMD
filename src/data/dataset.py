"""Load features and labels and assemble length-bucketed batches."""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np

from src.data.labels import ARRAY_DTYPES, _decode_npz_meta, read_manifest


def to_torch_batch(batch: dict) -> dict:
    import torch

    result = {}
    for k, v in batch.items():
        if isinstance(v, list):
            result[k] = v
        elif isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        else:
            result[k] = v
    return result


def _resolve(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def resolve_feature_path(
    song_id: str,
    config: Mapping[str, Any],
    *,
    project_root: Path | str,
    feature_index: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    feature_cfg = dict(config.get("feature_cache", config.get("mert", {})))
    index_value = feature_cfg.get("feature_index_path")
    if index_value:
        index = feature_index
        if index is None:
            index_path = _resolve(index_value, root)
            index = json.loads(index_path.read_text(encoding="utf-8"))
        song = index["songs"][song_id]
        cache_root = index["roots"][song["cache_root"]]
        return _resolve(cache_root["path"], root) / song["relative_path"]
    return _resolve(feature_cfg["cache_dir"], root) / f"{song_id}.npz"


_LABEL_FIELDS = {
    "phrase": ("phrase_boundary", "phrase_mask"),
    "section_function": ("section_label",),
    "section_boundary": ("section_boundary",),
    "chord_root": ("chord_root",),
    "chord_quality": ("chord_quality",),
    "tactus": ("beat_label", "beat_mask"),
    "downbeat": ("downbeat_label", "downbeat_mask"),
}


def _read_dual_feature(
    path: Path,
    song_id: str,
    source: str,
    *,
    expected_backbone: str = "muq",
    expected_layer: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            feature = np.asarray(archive["features"], dtype=np.float32)
            meta = _decode_npz_meta(archive["meta"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"song_id={song_id} {source} 缓存缺失: {path}") from exc
    except (
        OSError,
        EOFError,
        zipfile.BadZipFile,
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        raise ValueError(
            f"song_id={song_id} {source} 缓存无法读取: {path}: {exc}"
        ) from exc
    if feature.ndim != 2 or feature.shape[1] != 1024:
        raise ValueError(
            f"song_id={song_id} {source} 缓存 {path}: 期望 features [T,1024]，实际 {feature.shape}"
        )
    if not isinstance(meta, dict):
        raise ValueError(
            f"song_id={song_id} {source} 缓存 {path}: 期望特征元数据字典，实际 {meta!r}"
        )
    for key, expected in (
        ("backbone", expected_backbone),
        ("layer", expected_layer),
        ("frame_rate_hz", 25.0),
        ("T", len(feature)),
    ):
        actual = meta.get(key)
        if actual != expected:
            raise ValueError(
                f"song_id={song_id} {source} 缓存 {path}: 期望 meta.{key}={expected}，实际 {actual!r}"
            )
    return feature, meta


def _load_feature_pair(primary_path, secondary_path, song_id, stream_specs):
    primary, meta = _read_dual_feature(primary_path, song_id, "主路", **stream_specs[0])
    secondary, _ = _read_dual_feature(
        secondary_path, song_id, "secondary", **stream_specs[1]
    )
    if primary.shape[0] != secondary.shape[0]:
        raise ValueError(
            f"song_id={song_id} 双路缓存 T 不一致: "
            f"主路 {primary_path} T={primary.shape[0]}，"
            f"第二路 {secondary_path} T={secondary.shape[0]}；期望相同 T"
        )
    return np.concatenate((primary, secondary), axis=1), meta


class FeatureDataset:

    def __init__(self, song_ids: List[str], config: Mapping[str, Any]) -> None:
        self.song_ids = list(song_ids)
        self.config = config
        self.root = Path(config.get("project_root", ".")).resolve()
        feature = config["mert"]
        self.secondary_root = _resolve(feature["secondary_cache_dir"], self.root)
        self.stream_specs = [
            {
                "expected_backbone": feature.get(f"{name}_backbone", backbone),
                "expected_layer": feature.get(f"{name}_layer", layer),
            }
            for name, backbone, layer in (
                ("primary", "muq", 6),
                ("secondary", "musicfm", 7),
            )
        ]

    def __len__(self) -> int:
        return len(self.song_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        song_id = self.song_ids[index]
        primary = resolve_feature_path(song_id, self.config, project_root=self.root)
        feature, _ = _load_feature_pair(
            primary, self.secondary_root / f"{song_id}.npz", song_id, self.stream_specs
        )
        return {"feature": feature, "length": len(feature), "song_id": song_id}


class PhraseDataset:

    def __init__(
        self,
        json_paths: List[Any],
        config: Dict[str, Any],
        active_internal_heads: frozenset[str] | None = None,
    ) -> None:
        active = (
            frozenset(_LABEL_FIELDS)
            if active_internal_heads is None
            else frozenset(active_internal_heads)
        )
        unknown = active - _LABEL_FIELDS.keys()
        if not active or unknown:
            raise ValueError(f"数据集活动输出头为空或未知: {sorted(unknown)}")
        self._label_fields = tuple(
            field
            for head, fields in _LABEL_FIELDS.items()
            if head in active
            for field in fields
        )
        root = Path(config.get("project_root", ".")).resolve()
        label_root = Path(config["label_cache"]["cache_dir"])
        self._label_root = label_root if label_root.is_absolute() else root / label_root
        feature_cfg = dict(config.get("feature_cache", config.get("mert", {})))
        legacy_feature_cfg = config.get("mert", {})
        self._stream_specs = [
            {
                "expected_backbone": feature_cfg.get(
                    f"{name}_backbone",
                    legacy_feature_cfg.get(f"{name}_backbone", "muq"),
                ),
                "expected_layer": feature_cfg.get(
                    f"{name}_layer", legacy_feature_cfg.get(f"{name}_layer", 6)
                ),
            }
            for name in ("primary", "secondary")
        ]
        secondary_dir = feature_cfg.get(
            "secondary_cache_dir", config.get("mert", {}).get("secondary_cache_dir")
        )
        self._secondary_root = None
        if secondary_dir:
            secondary_path = Path(secondary_dir)
            self._secondary_root = (
                secondary_path
                if secondary_path.is_absolute()
                else root / secondary_path
            )
        index_value = feature_cfg.get("feature_index_path")
        self._index = None
        if index_value:
            index_path = Path(index_value)
            index_path = index_path if index_path.is_absolute() else root / index_path
            self._index = json.loads(index_path.read_text(encoding="utf-8"))
        self._entries: list[dict[str, Any]] = []
        manifest = read_manifest(self._label_root / "manifest.json")
        for raw_path in json_paths:
            song_id = Path(raw_path).stem
            record = manifest["songs"][song_id]
            feature_path = resolve_feature_path(
                song_id, config, project_root=root, feature_index=self._index
            )
            self._entries.append(
                {
                    "song_id": song_id,
                    "feature_path": feature_path,
                    "secondary_path": (
                        self._secondary_root / f"{song_id}.npz"
                        if self._secondary_root
                        else None
                    ),
                    "label_path": self._label_root / record["artifact"],
                    "T": int(record["T"]),
                }
            )
        self.lengths = [entry["T"] for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self._entries[index]
        if entry["secondary_path"] is None:
            with np.load(entry["feature_path"], allow_pickle=False) as archive:
                feature = np.asarray(archive["features"], dtype=np.float32)
                meta = _decode_npz_meta(archive["meta"])
        else:
            feature, meta = _load_feature_pair(
                entry["feature_path"],
                entry["secondary_path"],
                entry["song_id"],
                self._stream_specs,
            )
        with np.load(entry["label_path"], allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in self._label_fields}
        T = int(feature.shape[0])
        if meta.get("backbone") in {"muq", "musicfm"}:
            arrays = {key: value[:T] for key, value in arrays.items()}
        if entry["secondary_path"] is not None:
            for key, value in arrays.items():
                if value.ndim != 1 or len(value) != T:
                    raise ValueError(
                        f"song_id={entry['song_id']} 标签 {entry['label_path']} 的 {key}: "
                        f"期望 {T} 帧，实际形状 {value.shape}"
                    )
        return {"feature": feature, **arrays, "length": T, "song_id": entry["song_id"]}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("不能拼接空 batch")
    B = len(batch)
    lengths = np.asarray([item["length"] for item in batch], dtype=np.int64)
    T = int(lengths.max())
    feature_shape = np.asarray(batch[0]["feature"]).shape[1:]
    label_keys = tuple(key for key in ARRAY_DTYPES if key in batch[0])
    for item in batch[1:]:
        if tuple(key for key in ARRAY_DTYPES if key in item) != label_keys:
            raise ValueError("batch 中各歌曲的活动标签字段不一致")
    result = {
        "feature": np.zeros((B, T) + feature_shape, dtype=np.float32),
        "pad_mask": np.ones((B, T), dtype=bool),
    }
    result.update(
        {key: np.zeros((B, T), dtype=ARRAY_DTYPES[key]) for key in label_keys}
    )
    for index, item in enumerate(batch):
        length = int(item["length"])
        for key in ("feature", *label_keys):
            result[key][index, :length] = item[key]
        result["pad_mask"][index, :length] = False
    return {
        **result,
        "lengths": lengths,
        "song_id": [str(item["song_id"]) for item in batch],
    }


def original_song_ids(song_ids: List[str], dataset_dir: str | Path) -> List[str]:
    manifest_path = Path(dataset_dir) / "offline_augmentation_manifest.json"
    records = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("songs", {})
        if manifest_path.is_file()
        else {}
    )
    augmented = {
        song_id
        for song_id, record in records.items()
        if record.get("parent_song_id") and record["parent_song_id"] != song_id
    }
    pattern = r"^.+__(original|pitch_up|pitch_down|speed_up|speed_down)__.+__seed\d+$"
    return sorted(
        song_id
        for song_id in song_ids
        if song_id not in augmented and not re.match(pattern, song_id)
    )


def load_split(
    split_path: str | Path, label_dir: str | Path
) -> Tuple[List[str], List[str], List[str]]:
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少数据划分文件：{path}。请自行提供包含 train/val/test 曲目ID列表的JSON；程序不生成划分。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(label_dir)
    return tuple(
        [str(root / f"{song_id}.json") for song_id in payload[key]]
        for key in ("train", "val", "test")
    )


class LengthBucketSampler:

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        n = len(self.lengths)
        if n == 0:
            self._num_batches = 0
        else:
            full_batches = n // batch_size
            has_remainder = (n % batch_size) != 0
            if drop_last:
                self._num_batches = full_batches
            else:
                self._num_batches = full_batches + int(has_remainder)


    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.lengths)
        if n == 0:
            return

        sorted_indices: List[int] = sorted(range(n), key=lambda i: self.lengths[i])

        buckets: List[List[int]] = []
        for start in range(0, n, self.batch_size):
            bucket = sorted_indices[start : start + self.batch_size]
            buckets.append(bucket)

        if self.drop_last and buckets and len(buckets[-1]) < self.batch_size:
            buckets = buckets[:-1]

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        for bucket in buckets:
            yield bucket

    def __len__(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:

        self.epoch = int(epoch)
