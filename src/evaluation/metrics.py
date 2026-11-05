"""Compute boundary metrics and convert chord labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import mir_eval.beat
import mir_eval.chord
import mir_eval.segment
import mir_eval.util
import numpy as np

from src.evaluation.gt_extract import (
    extract_gt_phrase,
    extract_gt_section,
    root_quality_to_mir_eval_symbol,
)


def _as_float(value: Any) -> float:
    return float(np.asarray(value).item() if np.asarray(value).shape == () else value)


def _duration_from_gt(gt_json: Path) -> float:
    intervals, _ = extract_gt_section(gt_json)
    if intervals.size == 0:
        return 0.0
    return float(intervals[-1, 1])


def _events_to_intervals(events_s: np.ndarray, duration_s: float) -> np.ndarray:
    events = np.asarray(events_s, dtype=float)
    if duration_s <= 0:
        return np.zeros((0, 2), dtype=float)
    inner = events[(events > 0.0) & (events < duration_s)]
    boundaries = np.unique(np.concatenate([[0.0], inner, [duration_s]])).astype(float)
    if boundaries.size < 2:
        boundaries = np.array([0.0, duration_s], dtype=float)
    return mir_eval.util.boundaries_to_intervals(boundaries)


def _score_phrase(
    gt_json: Path,
    pred: Mapping[str, np.ndarray],
    windows: Sequence[float],
) -> Dict[str, Any]:
    duration_s = _duration_from_gt(gt_json)
    ref_intervals = _events_to_intervals(extract_gt_phrase(gt_json), duration_s)
    est_intervals = _events_to_intervals(
        np.asarray(pred["phrase_boundary_times_s"]), duration_s
    )
    out: Dict[str, Any] = {}
    for w in windows:
        precision, recall, f_measure = mir_eval.segment.detection(
            ref_intervals,
            est_intervals,
            window=float(w),
            trim=True,
        )
        out[f"window_{w:g}"] = {
            "precision": _as_float(precision),
            "recall": _as_float(recall),
            "f_measure": _as_float(f_measure),
        }
    return out


def _chord_labels(
    root_idx: Sequence[int],
    quality_idx: Sequence[int],
    chord_vocab: Mapping[str, Any],
) -> List[str]:
    q_i2l = chord_vocab["quality_idx2label"]
    pc_map = chord_vocab["root_info"]["pitch_class_map"]
    labels: List[str] = []
    for root, quality in zip(root_idx, quality_idx):
        labels.append(
            root_quality_to_mir_eval_symbol(int(root), int(quality), q_i2l, pc_map)
        )
    return labels
