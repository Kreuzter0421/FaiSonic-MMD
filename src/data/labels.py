"""Build and cache musical supervision labels."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ChordNormalization:
    valid: bool
    root: Optional[int]
    canonical_quality: Optional[str]
    quality_index: Optional[int]
    provenance: str
    issue: Optional[Dict[str, Any]] = None


def load_section_vocab(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_chord_vocab(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_chord(
    raw_root: Any,
    raw_quality: Any,
    chord_vocab: Dict[str, Any],
    *,
    allow_lossy: bool = False,
) -> ChordNormalization:
    if not isinstance(raw_root, int) or not isinstance(raw_quality, str):
        return ChordNormalization(False, None, None, None, "invalid")
    labels = chord_vocab["quality_label2idx"]
    if raw_quality in labels:
        return ChordNormalization(
            True,
            raw_root,
            raw_quality,
            labels[raw_quality],
            "no_chord" if raw_quality == "N" else "standard",
        )
    aliases = chord_vocab.get("alias2standard", {})
    if raw_quality in aliases:
        quality = aliases[raw_quality]
        return ChordNormalization(True, raw_root, quality, labels[quality], "alias")
    lossy = chord_vocab.get("unmapped_aliases", {}).get(raw_quality)
    if lossy:
        return ChordNormalization(
            allow_lossy, raw_root, lossy["fallback"], lossy["fallback_idx"], "lossy"
        )
    return ChordNormalization(False, raw_root, None, None, "invalid")


Meter = Tuple[int, int]


def _meter(entry: Mapping[str, Any]) -> Meter:
    return int(entry["numerator"]), int(entry["denominator"])


def _coordinate(entry: Mapping[str, Any]) -> int | float:
    value = entry.get("measure_index", entry.get("measure"))
    return int(value) if isinstance(value, float) and value.is_integer() else value


def normalize_meter_by_bar(
    audio_config: Mapping[str, Any], n_bars: int, source: str = ""
) -> list[Meter]:
    entries = audio_config.get("time_signatures")
    if entries is None or entries == []:
        overrides: dict[int, Meter] = {0: _meter(audio_config)}
    else:
        in_range = [
            (_coordinate(entry), entry)
            for entry in entries
            if 0 <= _coordinate(entry) < n_bars
        ]
        integers = sorted(
            (item for item in in_range if isinstance(item[0], int)),
            key=lambda item: item[0],
        )
        overrides = {}
        for coordinate, entry in integers:
            overrides.setdefault(coordinate, _meter(entry))

    current = overrides[0]
    result: list[Meter] = []
    for measure_index in range(n_bars):
        if measure_index in overrides:
            current = overrides[measure_index]
        result.append(current)
    return result


def _is_compound(numerator: int, denominator: int) -> bool:
    return denominator == 8 and numerator in {6, 9, 12}


def _tactus_count(numerator: int, denominator: int) -> int:
    return numerator // 3 if _is_compound(numerator, denominator) else numerator


def build_beat_truth_times(
    bar_timestamps_ms: List[int],
    audio_config: Dict,
    offset_ms: float,
    precision: str = "half_beat",
    channel: str = "tactus",
    source: str = "annotation=<unknown>",
) -> List[float]:
    bt_ms = list(bar_timestamps_ms)
    n_bars = len(bt_ms) - 1

    if channel == "downbeat":
        downbeat_times_s: List[float] = []
        for index in range(n_bars):
            t_s = (bt_ms[index] - offset_ms) / 1000.0
            downbeat_times_s.append(t_s)
        return downbeat_times_s

    meter_by_bar = normalize_meter_by_bar(audio_config, n_bars, source)
    points_per_tactus = 1 if precision == "beat" else 2
    truth_times_s: List[float] = []
    for bar_idx in range(n_bars):
        numerator, denominator = meter_by_bar[bar_idx]
        tactus_count = _tactus_count(numerator, denominator)
        bar_start_ms = bt_ms[bar_idx]
        bar_end_ms = bt_ms[bar_idx + 1]
        bar_duration_ms = bar_end_ms - bar_start_ms
        sub_duration_ms = bar_duration_ms / (tactus_count * points_per_tactus)
        total_points = tactus_count * points_per_tactus
        for point in range(total_points):
            absolute_ms = bar_start_ms + point * sub_duration_ms
            truth_times_s.append((absolute_ms - offset_ms) / 1000.0)
    return truth_times_s


def build_beat_head_labels(
    grid_times_s: Sequence[float],
    beat_truth_times_s: Sequence[float],
    sigma_grid: float,
    beat_coverage_intervals: Optional[List[Tuple[float, float]]] = None,
    label_mode: str = "hard",
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(grid_times_s, dtype=np.float64)
    values = np.zeros(len(grid), dtype=np.float32)
    mask = np.zeros(len(grid), dtype=np.float32)
    if beat_coverage_intervals is None:
        mask[:] = 1.0
    else:
        for start_s, end_s in beat_coverage_intervals:
            start_index = int(np.argmin(np.abs(grid - start_s)))
            end_index = int(np.argmin(np.abs(grid - end_s)))
            mask[start_index : end_index + 1] = 1.0

    grid_start = float(grid[0])
    grid_end = float(grid[-1])
    for truth_s in beat_truth_times_s:
        if truth_s < grid_start or truth_s > grid_end:
            continue
        index = int(np.argmin(np.abs(grid - truth_s)))
        if label_mode == "gaussian":
            half_width = math.ceil(4 * sigma_grid)
            lo, hi = max(0, index - half_width), min(len(grid) - 1, index + half_width)
            for point in range(lo, hi + 1):
                distance = point - index
                weight = math.exp(
                    -(distance * distance) / (2.0 * sigma_grid * sigma_grid)
                )
                if weight > values[point]:
                    values[point] = float(weight)
        else:
            values[index] = 1.0
    return (values * mask).astype(np.float32), mask


CACHE_SCHEMA_VERSION = 1

MANIFEST_SCHEMA_VERSION = 1

ARRAY_DTYPES = {
    "phrase_boundary": "float32",
    "phrase_mask": "float32",
    "section_label": "int64",
    "section_boundary": "float32",
    "chord_root": "int64",
    "chord_quality": "int64",
    "beat_label": "float32",
    "beat_mask": "float32",
    "downbeat_label": "float32",
    "downbeat_mask": "float32",
}

_META_KEY = "__meta__"


def _decode_npz_meta(raw: np.ndarray) -> Any:
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def read_feature_info(path: Path | str) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        features = archive["features"]
        shape, meta = features.shape, _decode_npz_meta(archive["meta"])
    result = {"T": int(shape[0]), "meta": meta}
    if isinstance(meta, Mapping) and "audio_duration_s" in meta:
        result["audio_duration_s"] = float(meta["audio_duration_s"])
    return result


def build_song_key_map(song_ids: Sequence[str]) -> dict[str, str]:
    return {
        song_id: song_id.replace("/", "-").replace("\\", "-") for song_id in song_ids
    }


def write_label_artifact(
    path: Path | str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez(
            handle,
            **{key: np.asarray(value) for key, value in arrays.items()},
            **{
                _META_KEY: np.bytes_(
                    json.dumps(dict(meta), ensure_ascii=False).encode("utf-8")
                )
            },
        )
    return {"path": str(target), "meta": dict(meta)}


def read_manifest(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"schema_version": MANIFEST_SCHEMA_VERSION, "songs": {}}
    return json.loads(target.read_text(encoding="utf-8"))


def merge_manifest_record(
    path: Path | str, song_id: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(target)
    songs = dict(manifest.get("songs", {}))
    songs[song_id] = dict(record)
    merged = {"schema_version": MANIFEST_SCHEMA_VERSION, "songs": songs}
    target.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    return merged


def resolve_section_segments(
    structures: Sequence[Dict[str, Any]],
    *,
    total_duration_s: float | None = None,
    background_label: Any = "<background>",
) -> List[Tuple[float, float, Any]]:
    entries = []
    for index, item in enumerate(structures):
        if item.get("level") == "section":
            start = float(item["audio_start_ms"]) / 1000.0
            end = start + float(item["duration_ms"]) / 1000.0
            entries.append((start, end, index, item.get("text", "")))
    boundaries = sorted(
        {value for start, end, _, _ in entries for value in (start, end)}
    )
    if total_duration_s is not None:
        boundaries = sorted(set(boundaries + [0.0, total_duration_s]))
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [entry for entry in entries if entry[0] <= start and entry[1] >= end]
        label = (
            max(active, key=lambda item: (item[0], item[2]))[3]
            if active
            else background_label
        )
        result.append((start, end, label))
    return result


def build_section_labels(
    structures: List[Dict[str, Any]],
    grid_times_s: Sequence[float],
    section_vocab: Dict[str, int],
) -> np.ndarray:
    grid = np.asarray(grid_times_s)
    output = np.zeros(len(grid), dtype=np.int64)
    for start, end, label in resolve_section_segments(structures):
        output[(grid >= start) & (grid < end)] = section_vocab.get(str(label), 0)
    return output


def build_section_instance_boundary(
    structures: Sequence[Dict[str, Any]],
    grid_times_s: Sequence[float],
    *,
    native_stride: int = 8,
) -> np.ndarray:

    grid = np.asarray(grid_times_s)
    boundary = np.zeros(len(grid), dtype=np.float32)
    endpoints = sorted(
        {
            value
            for item in structures
            for value in (
                float(item["audio_start_ms"]) / 1000.0,
                (float(item["audio_start_ms"]) + float(item["duration_ms"])) / 1000.0,
            )
        }
    )
    if len(endpoints) < 3:
        return boundary

    anchors = np.arange(0, len(grid), native_stride, dtype=np.int64)
    for endpoint_s in endpoints[1:-1]:
        frame_index = int(np.abs(grid - endpoint_s).argmin())
        native_index = min(
            (frame_index + native_stride // 2) // native_stride,
            len(anchors) - 1,
        )
        boundary[anchors[native_index]] = 1.0
    return boundary


def build_phrase_boundary_and_mask(
    phrases: List[Dict[str, Any]], grid_times_s: Sequence[float], sigma_grid: float
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(grid_times_s)
    boundary, mask = (
        np.zeros(len(grid), dtype=np.float32),
        np.zeros(len(grid), dtype=np.float32),
    )
    items = [item for item in phrases if item.get("level", "phrase") == "phrase"]
    if not items:
        return boundary, mask
    start = min(float(item["audio_start_ms"]) / 1000.0 for item in items)
    end = max(
        (float(item["audio_start_ms"]) + float(item["duration_ms"])) / 1000.0
        for item in items
    )
    start_index = int(np.argmin(np.abs(grid - start)))
    end_index = int(np.argmin(np.abs(grid - end)))
    mask[start_index : end_index + 1] = 1.0
    radius = int(math.ceil(4 * sigma_grid))
    for item in items:
        index = int(np.abs(grid - float(item["audio_start_ms"]) / 1000.0).argmin())
        lo, hi = max(0, index - radius), min(len(grid), index + radius + 1)
        offsets = np.arange(lo, hi) - index
        boundary[lo:hi] = np.maximum(
            boundary[lo:hi],
            np.exp(-(offsets * offsets) / (2 * sigma_grid * sigma_grid)),
        )
    return (boundary * mask).astype(np.float32), mask


@dataclass(frozen=True)
class LabelBuildResult:
    arrays: Dict[str, np.ndarray]
    status: str
    issues: list[Dict[str, Any]]


def _grid(
    T: int, frame_rate_hz: float, grid_times_s: Optional[Sequence[float]]
) -> np.ndarray:
    if grid_times_s is None:
        return np.arange(T, dtype=np.float64) / float(frame_rate_hz)
    return np.asarray(grid_times_s, dtype=np.float64)


def _clip_intervals(
    items: Sequence[Mapping[str, Any]], audio_duration_s: float
) -> list[Dict[str, Any]]:
    clipped: list[Dict[str, Any]] = []
    for raw in items:
        start_ms = raw.get("audio_start_ms")
        duration_ms = raw.get("duration_ms")
        raw_start_s = float(start_ms) / 1000.0
        raw_end_s = float(start_ms + duration_ms) / 1000.0
        if raw_end_s <= 0.0 or raw_start_s >= audio_duration_s:
            continue
        start_s = max(0.0, raw_start_s)
        end_s = min(audio_duration_s, raw_end_s)
        if end_s <= start_s:
            continue
        item = dict(raw)
        item["audio_start_ms"] = start_s * 1000.0
        item["duration_ms"] = (end_s - start_s) * 1000.0
        clipped.append(item)
    return clipped


def _in_duration(values: Sequence[float], audio_duration_s: float) -> list[float]:
    return [value for value in values if 0.0 <= value < audio_duration_s]


def _build_chords(
    chords: Sequence[Mapping[str, Any]],
    grid: np.ndarray,
    chord_vocab: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    root = np.zeros(len(grid), dtype=np.int64)
    quality = np.zeros(len(grid), dtype=np.int64)
    for item in sorted(chords, key=lambda value: value["audio_start_ms"]):
        normalized = normalize_chord(
            item.get("root"), item.get("quality"), dict(chord_vocab)
        )
        if not normalized.valid:
            continue
        start_s = float(item["audio_start_ms"]) / 1000.0
        end_s = start_s + float(item["duration_ms"]) / 1000.0
        start_idx = int(np.argmin(np.abs(grid - start_s)))
        end_idx = int(np.argmin(np.abs(grid - end_s)))
        quality[start_idx : end_idx + 1] = normalized.quality_index
        root[start_idx : end_idx + 1] = normalized.root
    return root, quality


def build_song_labels(
    annotation: Mapping[str, Any],
    *,
    T: int,
    frame_rate_hz: float,
    audio_duration_s: float,
    section_vocab: Mapping[str, Any],
    chord_vocab: Mapping[str, Any],
    sigma_grid: float = 1.0,
    beat_precision: str = "beat",
    beat_label_mode: str = "hard",
    grid_times_s: Optional[Sequence[float]] = None,
    section_native_stride: int = 8,
    source: str = "annotation=<in-memory>",
    include_phrase: bool = True,
) -> LabelBuildResult:

    grid = _grid(T, frame_rate_hz, grid_times_s)
    audio_duration_s = float(audio_duration_s)
    structures = list(annotation.get("structures", []) or [])
    section_raw = [item for item in structures if item.get("level") == "section"]
    phrase_raw = (
        [item for item in structures if item.get("level", "phrase") == "phrase"]
        if include_phrase
        else []
    )
    sections = _clip_intervals(section_raw, audio_duration_s)
    phrases = _clip_intervals(phrase_raw, audio_duration_s)

    section_map = section_vocab.get("label2idx", section_vocab)
    aliases = {
        alias.casefold(): target
        for alias, target in section_vocab.get("alias2standard", {}).items()
    }
    safe_sections: list[Dict[str, Any]] = []
    for item in sections:
        raw_text = str(item.get("text", "")).strip()
        if raw_text in {"Section", "Section 2"}:
            continue
        base_name = re.sub(r"\s+\d+$", "", raw_text)
        canonical = (
            base_name
            if base_name in section_map
            else aliases.get(raw_text.casefold(), aliases.get(base_name.casefold()))
        )
        if canonical in section_map:
            normalized_item = dict(item)
            normalized_item["text"] = canonical
            safe_sections.append(normalized_item)

    safe_chords = _clip_intervals(
        list(annotation.get("chords", []) or []), audio_duration_s
    )
    chord_root, chord_quality = _build_chords(safe_chords, grid, chord_vocab)
    section_label = build_section_labels(safe_sections, grid, dict(section_map))
    section_boundary = build_section_instance_boundary(
        sections, grid, native_stride=section_native_stride
    )
    if include_phrase:
        phrase_boundary, phrase_mask = build_phrase_boundary_and_mask(
            phrases, grid, sigma_grid
        )

    audio_config = dict(annotation.get("audio_config", {}) or {})
    bar_timestamps = list(audio_config.get("bar_timestamps", []) or [])
    offset_ms = float(audio_config.get("offset", 0.0)) * 1000.0
    if len(bar_timestamps) >= 2:
        tactus = build_beat_truth_times(
            bar_timestamps, audio_config, offset_ms, beat_precision, "tactus", source
        )
        downbeats = build_beat_truth_times(
            bar_timestamps,
            audio_config,
            offset_ms,
            channel="downbeat",
            source=source,
        )
    else:
        tactus, downbeats = [], []
    tactus_in = _in_duration(tactus, audio_duration_s)
    downbeats_in = _in_duration(downbeats, audio_duration_s)
    coverage = [(downbeats_in[0], downbeats_in[-1])] if downbeats_in else []
    beat_label, beat_mask = build_beat_head_labels(
        grid, tactus_in, sigma_grid, coverage, beat_label_mode
    )
    downbeat_label, downbeat_mask = build_beat_head_labels(
        grid, downbeats_in, sigma_grid, coverage, beat_label_mode
    )
    arrays = {
        "section_label": section_label.astype(np.int64, copy=False),
        "section_boundary": section_boundary,
        "chord_root": chord_root,
        "chord_quality": chord_quality,
        "beat_label": beat_label,
        "beat_mask": beat_mask,
        "downbeat_label": downbeat_label,
        "downbeat_mask": downbeat_mask,
    }
    if include_phrase:
        arrays["phrase_boundary"] = phrase_boundary.astype(np.float32, copy=False)
        arrays["phrase_mask"] = phrase_mask.astype(np.float32, copy=False)
    return LabelBuildResult(arrays=arrays, status="ready", issues=[])


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _semantic_inputs(
    config: Mapping[str, Any], project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    vocab = config["vocab"]
    return load_section_vocab(
        _resolve(vocab["section"], project_root)
    ), load_chord_vocab(_resolve(vocab["chord"], project_root))


def run_label_preprocess(
    config: Mapping[str, Any],
    *,
    mode: str,
    project_root: Path | str = ".",
    label_paths_override: list[str] | None = None,
) -> dict[str, Any]:
    from src.data.dataset import original_song_ids, resolve_feature_path

    root = Path(project_root).resolve()
    cache_dir = _resolve(config["label_cache"]["cache_dir"], root)
    labels_dir = _resolve(config["sample_dir"], root) / "labels"
    paths = (
        [Path(path) for path in label_paths_override]
        if label_paths_override is not None
        else sorted(labels_dir.glob("*.json"))
    )
    if label_paths_override is None:
        selected = set(
            original_song_ids([path.stem for path in paths], labels_dir.parent)
        )
        paths = [path for path in paths if path.stem in selected]
    section_vocab, chord_vocab = _semantic_inputs(config, root)
    keys = build_song_key_map([path.stem for path in paths])
    manifest = read_manifest(cache_dir / "manifest.json")
    built, reused, failed = [], [], []
    for label_path in paths:
        song_id = label_path.stem
        artifact = cache_dir / "artifacts" / f"{keys[song_id]}.npz"
        old_artifact = manifest.get("songs", {}).get(song_id, {}).get("artifact")
        if isinstance(old_artifact, str):
            existing = cache_dir / old_artifact
            if existing.is_file():
                artifact = existing
        if mode == "build" and artifact.is_file():
            reused.append(song_id)
            continue
        try:
            feature = read_feature_info(
                resolve_feature_path(song_id, config, project_root=root)
            )
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            rate = float(config.get("frame_rate_hz", 75.0)) / int(
                config.get("downsample_n", 1)
            )
            result = build_song_labels(
                annotation,
                T=feature["T"],
                frame_rate_hz=rate,
                audio_duration_s=float(
                    feature.get("audio_duration_s", feature["T"] / rate)
                ),
                section_vocab=section_vocab,
                chord_vocab=chord_vocab,
                sigma_grid=float(config.get("gaussian_sigma_ms", 70.0)) * rate / 1000.0,
                beat_precision=str(config.get("beat_precision", "beat")),
                beat_label_mode=str(config.get("beat_label_mode", "hard")),
                section_native_stride=8,
                include_phrase=True,
            )
            meta = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "song_id": song_id,
                "song_key": keys[song_id],
                "T": feature["T"],
            }
            write_label_artifact(artifact, result.arrays, meta)
            merge_manifest_record(
                cache_dir / "manifest.json",
                song_id,
                {"artifact": str(artifact.relative_to(cache_dir)), "T": feature["T"]},
            )
            built.append(song_id)
        except Exception as exc:
            failed.append({"song_id": song_id, "error": str(exc)})
    return {"built": built, "reused": reused, "failed": failed}


def report_exit_code(report: Mapping[str, Any]) -> int:
    return int(bool(report["failed"]))
