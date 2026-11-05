"""Extract ground truth and convert chord symbols."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mir_eval.chord as mir_eval_chord
import numpy as np

from src.data.labels import (
    build_beat_truth_times,
    load_chord_vocab,
    normalize_chord,
    resolve_section_segments,
)

__all__ = [
    "extract_gt_phrase",
    "extract_gt_section",
    "extract_gt_chord",
    "extract_gt_downbeat",
    "extract_gt_tactus",
    "root_quality_to_mir_eval_symbol",
    "QUALITY_TO_MIR_EVAL_SPEC",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHORD_VOCAB_PATH = (
    PROJECT_ROOT / "configs" / "vocab" / "chord.json"
)

DEFAULT_DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

DEFAULT_BACKGROUND_LABEL = "<background>"


QUALITY_TO_MIR_EVAL_SPEC: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "major": ("maj", ()),
    "minor": ("min", ()),
    "diminished": ("dim", ()),
    "augmented": ("aug", ()),
    "major7": ("maj7", ()),
    "minor7": ("min7", ()),
    "dominant7": ("7", ()),
    "half-dim7": ("hdim7", ()),
    "dim7": ("dim7", ()),
    "major6": ("maj6", ()),
    "minor6": ("min6", ()),
    "sus2": ("sus2", ()),
    "sus4": ("sus4", ()),
    "major9": ("maj9", ()),
    "minor9": ("min9", ()),
    "dominant9": ("9", ()),
    "11": ("11", ()),
    "13": ("13", ()),
    "minor-major7": ("minmaj7", ()),
    "add9": ("maj", ("9",)),
    "add11": ("maj", ("11",)),
    "add13": ("maj", ("13",)),
    "altered": ("7", ("#5", "*5", "b9")),
    "power": ("5", ()),
    "minor-add9": ("min", ("9",)),
    "minor-add11": ("min", ("11",)),
    "minor-add13": ("min", ("13",)),
    "dominant7sus4": ("sus4", ("b7",)),
    "dominant7sus2": ("sus2", ("b7",)),
    "dominant9sus4": ("sus4", ("b7", "9")),
    "minor11": ("min11", ()),
    "major11": ("maj7", ("9", "11")),
    "major69": ("maj6", ("9",)),
    "major13": ("maj13", ()),
    "minor13": ("min13", ()),
}


def _root_idx_to_pitch_name(root_idx: int, pitch_class_map: Dict[str, str]) -> str:
    raw = pitch_class_map[str(root_idx)]
    return raw.split("/")[0]


def root_quality_to_mir_eval_symbol(
    root_idx: int,
    quality_idx: int,
    quality_idx2label: Dict[str, str],
    pitch_class_map: Dict[str, str],
) -> str:
    if quality_idx == 0:
        return mir_eval_chord.NO_CHORD

    quality_name = quality_idx2label[str(quality_idx)]
    if quality_name == "X":
        return mir_eval_chord.X_CHORD
    quality_base, extensions = QUALITY_TO_MIR_EVAL_SPEC[quality_name]
    root_name = _root_idx_to_pitch_name(root_idx, pitch_class_map)
    extensions_list: Optional[List[str]] = list(extensions) if extensions else None
    return mir_eval_chord.join(root_name, quality_base, extensions_list)


def _load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(json_path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _load_chord_vocab_cached(path_str: str) -> Dict[str, Any]:
    return load_chord_vocab(path_str)


@lru_cache(maxsize=None)
def _default_beat_precision() -> str:
    try:
        import yaml

        with open(DEFAULT_DATA_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("beat_precision", "beat"))
    except FileNotFoundError:
        return "beat"


def _infer_audio_duration_s(data: Dict[str, Any]) -> float:
    ac = data.get("audio_config", {}) or {}

    num_samples = ac.get("num_samples", None)
    sample_rate = ac.get("sample_rate", None)
    if num_samples is not None and sample_rate and float(sample_rate) > 0:
        ns_duration_s = float(num_samples) / float(sample_rate)
    else:
        ns_duration_s = 0.0

    dur_field = ac.get("duration", None)
    duration_s = float(dur_field) if dur_field is not None else 0.0

    structures = data.get("structures", []) or []
    chords = data.get("chords", []) or []
    max_event_end_ms = 0.0
    for s in structures:
        end_ms = s.get("audio_start_ms", 0) + s.get("duration_ms", 0)
        if end_ms > max_event_end_ms:
            max_event_end_ms = float(end_ms)
    for c in chords:
        end_ms = c.get("audio_start_ms", 0) + c.get("duration_ms", 0)
        if end_ms > max_event_end_ms:
            max_event_end_ms = float(end_ms)
    event_duration_s = max_event_end_ms / 1000.0

    audio_end_s = max(ns_duration_s, duration_s, event_duration_s) + 0.1
    if audio_end_s <= 0.1:
        audio_end_s = 1.0
    return audio_end_s


def extract_gt_phrase(json_path: Union[str, Path]) -> np.ndarray:
    data = _load_json(json_path)
    structures = data.get("structures", []) or []
    phrase_items = [s for s in structures if s.get("level") == "phrase"]

    starts_s = sorted({item["audio_start_ms"] / 1000.0 for item in phrase_items})
    arr = np.array(starts_s, dtype=float)

    return arr


def extract_gt_section(
    json_path: Union[str, Path],
    background_label: Any = DEFAULT_BACKGROUND_LABEL,
) -> Tuple[np.ndarray, List[Any]]:
    data = _load_json(json_path)
    duration_s = _infer_audio_duration_s(data)

    structures = data.get("structures", []) or []
    section_items = [
        s
        for s in structures
        if s.get("level") == "section" and float(s.get("duration_ms", 0.0)) > 0.0
    ]

    resolved = resolve_section_segments(
        section_items, total_duration_s=duration_s, background_label=background_label
    )
    intervals = np.asarray([[start, end] for start, end, _ in resolved], dtype=float)
    labels = [label for _, _, label in resolved]
    return intervals, labels


def extract_gt_chord(
    json_path: Union[str, Path],
    chord_vocab_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _load_json(json_path)
    chords = data.get("chords", []) or []

    if not chords:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
        )

    vocab_path = (
        str(chord_vocab_path) if chord_vocab_path else str(DEFAULT_CHORD_VOCAB_PATH)
    )
    chord_vocab = _load_chord_vocab_cached(vocab_path)

    chords_sorted = sorted(chords, key=lambda c: c["audio_start_ms"])
    n = len(chords_sorted)

    starts_s: List[float] = []
    ends_s: List[float] = []
    roots: List[int] = []
    qualities: List[int] = []

    for i, chord in enumerate(chords_sorted):
        start_ms = chord["audio_start_ms"]
        start_s = start_ms / 1000.0

        if i < n - 1:
            end_ms = chords_sorted[i + 1]["audio_start_ms"]
        else:
            end_ms = start_ms + chord["duration_ms"]
        end_s = end_ms / 1000.0

        starts_s.append(start_s)
        ends_s.append(end_s)
        normalized = normalize_chord(
            chord.get("root"),
            chord.get("quality"),
            chord_vocab,
        )
        roots.append(normalized.root)
        qualities.append(normalized.quality_index)

    intervals = np.array(list(zip(starts_s, ends_s)), dtype=float)
    root_idx = np.array(roots, dtype=int)
    quality_idx = np.array(qualities, dtype=int)

    return intervals, root_idx, quality_idx


def extract_gt_downbeat(json_path: Union[str, Path]) -> np.ndarray:
    data = _load_json(json_path)
    ac = data.get("audio_config", {}) or {}

    bar_timestamps_ms: List[int] = ac.get("bar_timestamps", []) or []
    offset_s = float(ac.get("offset", 0))
    offset_ms = offset_s * 1000.0

    duration_s = _infer_audio_duration_s(data)

    real_s = [(float(bt) - offset_ms) / 1000.0 for bt in bar_timestamps_ms]
    filtered = np.array([t for t in real_s if 0.0 <= t <= duration_s], dtype=float)

    return filtered


def extract_gt_tactus(
    json_path: Union[str, Path],
    precision: Optional[str] = None,
) -> np.ndarray:
    data = _load_json(json_path)
    ac = data.get("audio_config", {}) or {}

    bar_timestamps_ms: List[int] = ac.get("bar_timestamps", []) or []
    offset_s = float(ac.get("offset", 0))
    offset_ms = offset_s * 1000.0

    duration_s = _infer_audio_duration_s(data)

    if precision is None:
        precision = _default_beat_precision()

    truth_times_s: List[float] = build_beat_truth_times(
        bar_timestamps_ms,
        ac,
        offset_ms,
        precision=precision,
        channel="tactus",
        source=f"annotation={Path(json_path)}",
    )

    filtered = np.array(
        [t for t in truth_times_s if 0.0 <= t <= duration_s], dtype=float
    )

    return filtered
