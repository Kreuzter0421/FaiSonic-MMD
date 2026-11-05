"""Decode phrase, section, beat, and chord predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import mir_eval
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.special import expit, log_softmax, logit, softmax
from src.evaluation.metrics import _chord_labels

from src.evaluation.gt_extract import (
    extract_gt_chord,
    extract_gt_downbeat,
    extract_gt_tactus,
)

ROOT = Path(__file__).resolve().parents[2]

KEYS = ("root", "majmin", "thirds", "triads", "sevenths", "tetrads", "mirex")

VOCAB_PATH = ROOT / "configs/vocab/chord.json"

VOCAB = json.loads(VOCAB_PATH.read_text())

SECTION_FEATURE_NAMES = (
    "boundary_logit",
    "log_peak_width_s",
    "peak_prominence",
    "local_boundary_mean",
    "wide_boundary_min",
    "section_change",
    "phrase_support",
    "downbeat_support",
    "root_change",
)


def rle_compress(frame_labels: np.ndarray) -> List[Tuple[int, int, Any]]:
    frame_labels = np.asarray(frame_labels)
    T = frame_labels.shape[0]
    if T == 0:
        return []

    def _label_at(idx: int) -> Any:
        row = frame_labels[idx]
        if frame_labels.ndim == 1:
            return row.item() if hasattr(row, "item") else row
        return tuple(row.tolist())

    segments: List[Tuple[int, int, Any]] = []
    start = 0
    cur_label = _label_at(0)
    for i in range(1, T):
        lbl = _label_at(i)
        if lbl != cur_label:
            segments.append((start, i, cur_label))
            start = i
            cur_label = lbl
    segments.append((start, T, cur_label))
    return segments


# Beat and downbeat decoding.
def decode_frozen_beat(logits, grid_times_s, head, parameters):
    values = np.asarray(logits).reshape(-1)
    grid = np.asarray(grid_times_s, dtype=float)
    p = parameters["heads"][head]
    peaks, _ = find_peaks(
        expit(values), distance=parameters["candidate_min_distance_frames"]
    )
    peaks = peaks[expit(values[peaks]) >= parameters["candidate_probability_floor"]]
    scores = expit(
        (values[peaks].astype(float) - p["mean"]) / p["scale"] * p["coefficient"]
        + p["intercept"]
    )
    kept = []
    for j in np.argsort(-scores, kind="stable"):
        if scores[j] < p["threshold"]:
            break
        if all(
            (abs(peaks[j] - peaks[k]) >= p["min_peak_distance_frames"] for k in kept)
        ):
            kept.append(j)
    return grid[np.sort(peaks[kept])]


def _root_viterbi(z, penalty):
    z = np.asarray(z, float)
    n, k = z.shape
    if n == 0:
        return np.zeros(0, int)
    dp = z[0].copy()
    back = np.empty((n, k), np.int32)
    classes = np.arange(k)
    for t in range(1, n):
        best = np.argsort(dp)[-2:]
        other = np.where(classes == best[-1], best[-2], best[-1])
        sw = dp[other] - penalty[t - 1]
        take = sw > dp
        back[t] = np.where(take, other, classes)
        dp = z[t] + np.where(take, sw, dp)
    out = np.empty(n, int)
    out[-1] = dp.argmax()
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def expected_f1_threshold(scores: np.ndarray) -> float:
    probabilities = np.asarray(scores, dtype=float).reshape(-1)
    probabilities = probabilities[np.isfinite(probabilities)]
    if not len(probabilities) or float(probabilities.sum()) <= 0.0:
        return 0.5

    def residual(threshold: float) -> float:
        return float(
            np.maximum(probabilities - threshold, 0.0).sum()
            - threshold * probabilities.sum()
        )

    return float(brentq(residual, 0.0, 1.0))


# Phrase candidates and sequence decoding.
def extract_phrase_only_features(
    phrase_logits: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(phrase_logits, dtype=float).reshape(-1)
    probabilities = expit(values)
    peaks, _ = find_peaks(
        probabilities, distance=int(parameters["candidate_distance_frames"])
    )
    floor = float(parameters["candidate_probability_floor"])
    peaks = peaks[probabilities[peaks] >= floor]
    if not len(peaks):
        return (peaks.astype(int), np.zeros((0, 5), dtype=float))
    prominence = peak_prominences(probabilities, peaks, wlen=101)[0]
    frame_rate_hz = float(parameters.get("frame_rate_hz", 25.0))
    width = peak_widths(probabilities, peaks, rel_height=0.5)[0] / frame_rate_hz
    features = np.asarray(
        [
            [
                float(values[time]),
                float(np.log(width[index] + 0.001)),
                float(prominence[index]),
                float(np.mean(probabilities[max(0, time - 5) : time + 6])),
                float(np.min(probabilities[max(0, time - 25) : time + 26])),
            ]
            for index, time in enumerate(peaks)
        ],
        dtype=float,
    )
    return (peaks.astype(int), features)


def phrase_candidate_scores(
    phrase_logits: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    peaks, features = extract_phrase_only_features(phrase_logits, parameters)
    if not len(peaks):
        return (peaks, np.zeros(0, dtype=float))
    standardized = (
        features - np.asarray(parameters["scaler_mean"], dtype=float)
    ) / np.asarray(parameters["scaler_scale"], dtype=float)
    scores = expit(
        standardized @ np.asarray(parameters["coefficients"], dtype=float)
        + float(parameters["intercept"])
    )
    return (peaks, scores)


def select_phrase_candidates(
    peaks: np.ndarray, scores: np.ndarray, *, threshold: float, min_distance_frames: int
) -> np.ndarray:
    kept: list[int] = []
    for candidate in np.argsort(-np.asarray(scores), kind="stable"):
        if float(scores[candidate]) < float(threshold):
            break
        if all(
            (
                abs(int(peaks[candidate]) - int(peaks[other])) >= min_distance_frames
                for other in kept
            )
        ):
            kept.append(int(candidate))
    return (
        np.sort(peaks[np.asarray(kept, dtype=int)]) if kept else np.zeros(0, dtype=int)
    )


def decode_phrase_only(
    phrase_logits: np.ndarray, parameters: Mapping[str, object]
) -> np.ndarray:
    peaks, scores = phrase_candidate_scores(phrase_logits, parameters)
    selected = select_phrase_candidates(
        peaks,
        scores,
        threshold=float(parameters["phrase_threshold"]),
        min_distance_frames=int(parameters["event_min_distance_frames"]),
    )
    return selected.astype(float) / float(parameters.get("frame_rate_hz", 25.0))


def _match_count(reference: np.ndarray, estimated: np.ndarray, window_s: float) -> int:
    ref = np.sort(np.asarray(reference, dtype=float).reshape(-1))
    est = np.sort(np.asarray(estimated, dtype=float).reshape(-1))
    left = right = matches = 0
    while left < len(ref) and right < len(est):
        delta = est[right] - ref[left]
        if abs(delta) <= window_s:
            matches += 1
            left += 1
            right += 1
        elif delta < -window_s:
            right += 1
        else:
            left += 1
    return matches


def event_metrics(
    reference_times_s: np.ndarray,
    estimated_times_s: np.ndarray,
    *,
    windows_s: Sequence[float] = (0.5, 3.0),
) -> dict[str, dict[str, float | int]]:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    estimated = np.asarray(estimated_times_s, dtype=float).reshape(-1)
    output: dict[str, dict[str, float | int]] = {}
    for window in windows_s:
        matches = _match_count(reference, estimated, float(window))
        precision = (
            matches / len(estimated) if len(estimated) else float(not len(reference))
        )
        recall = (
            matches / len(reference) if len(reference) else float(not len(estimated))
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[f"window_{float(window):g}"] = {
            "matches": matches,
            "reference_count": len(reference),
            "estimated_count": len(estimated),
            "precision": precision,
            "recall": recall,
            "f_measure": f1,
        }
    return output


def section_feature_names():
    return SECTION_FEATURE_NAMES


def candidate_features(song: dict, task: str, section_stride: int = 8):
    width = 9
    if task == "phrase":
        peaks, base = extract_phrase_only_features(
            song["phrase_logits"],
            {
                "candidate_distance_frames": 12,
                "candidate_probability_floor": 0.05,
                "frame_rate_hz": 25,
            },
        )
    else:
        values = np.asarray(song["section_boundary_logits"]).reshape(-1)[
            ::section_stride
        ]
        native, _ = find_peaks(values)
        peaks = native * section_stride
        if not len(peaks):
            return (peaks, np.empty((0, width)))
        probabilities = expit(values)
        width = (
            peak_widths(probabilities, native, rel_height=0.5)[0] * section_stride / 25
        )
        near = max(1, round(8 / section_stride))
        wide = max(1, round(24 / section_stride))
        base = np.column_stack(
            (
                values[native],
                np.log(width + 0.001),
                peak_prominences(probabilities, native)[0],
                [
                    np.mean(probabilities[max(0, t - near) : t + near + 1])
                    for t in native
                ],
                [
                    np.min(probabilities[max(0, t - wide) : t + wide + 1])
                    for t in native
                ],
            )
        )
    if not len(peaks):
        return (peaks, np.empty((0, width)))
    sec = softmax(song["section_logits"], axis=1)
    root = softmax(song["chord_root_logits"], axis=1)
    cross = []
    for t in peaks:
        a, b = (max(0, t - 16), min(len(sec) - 1, t + 16))
        lo, hi = (max(0, t - 12), min(len(sec), t + 13))
        downbeat = expit(song["downbeat_logits"].reshape(-1)[lo:hi]).max()
        section_change = np.abs(sec[a] - sec[b]).sum() / 2
        root_change = np.abs(root[a] - root[b]).sum() / 2
        support = expit(
            song[
                "section_boundary_logits" if task == "phrase" else "phrase_logits"
            ].reshape(-1)[lo:hi]
        ).max()
        cross.append((section_change, support, downbeat, root_change))
    return (peaks, np.column_stack((base, np.asarray(cross))))


def f1_pair(song, times, task):
    times = np.unique(np.asarray(times, dtype=float))
    if task == "phrase":
        times = times[(times > 0) & (times < song["duration"])]
    result = event_metrics(
        song["phrase_ref" if task == "phrase" else "section_ref"], times
    )
    return np.asarray([result[f"window_{w:g}"]["f_measure"] for w in (0.5, 3.0)])


def score_model(x, model):
    return expit(
        (x - model["mean"]) / model["scale"] @ np.asarray(model["coef"])
        + model["intercept"]
    )


def duration_cost(times_s: np.ndarray, profile: Mapping[str, object]) -> np.ndarray:
    times = np.asarray(times_s, float).reshape(-1)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("候选时间必须有限且严格递增")
    gaps = np.maximum(times[:, None] - times[None, :], 1e-06)
    if profile["kind"] == "lognormal":
        costs = (
            np.log(gaps / float(profile["median_s"])) / float(profile["log_scale"])
        ) ** 2
    else:
        raise ValueError(f"未知时长约束：{profile['kind']}")
    return np.tril(costs, -1)


def select_soft_phrase_sequence(
    times_s: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    weight: float,
    costs: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times_s, float).reshape(-1)
    probabilities = np.asarray(probabilities, float).reshape(-1)
    costs = np.asarray(costs, float)
    n = len(times)
    if probabilities.shape != times.shape or costs.shape != (n, n):
        raise ValueError("时间、概率与转移矩阵形状不一致")
    if not 0 < threshold < 1 or weight < 0:
        raise ValueError("阈值必须在(0,1)，惩罚权重必须非负")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("候选概率必须有限且在[0,1]")
    if not n:
        return times.copy()
    rewards = logit(np.clip(probabilities, 1e-06, 1 - 1e-06)) - logit(threshold)
    best = rewards.copy()
    parents = np.full(n, -1, int)
    for current in range(1, n):
        previous = best[:current] - weight * costs[current, :current]
        parent = int(previous.argmax())
        if previous[parent] > 0:
            best[current] += previous[parent]
            parents[current] = parent
    endpoint = int(best.argmax())
    if best[endpoint] <= 0:
        return np.zeros(0, float)
    path = [endpoint]
    while parents[path[-1]] >= 0:
        path.append(int(parents[path[-1]]))
    return times[path[::-1]]


def predict_phrase_boundaries(peaks, scores, config, profile):
    times = np.asarray(peaks) / 25
    costs = duration_cost(times, {**profile, "kind": config["kind"]})
    return select_soft_phrase_sequence(
        times,
        scores,
        threshold=config["threshold"],
        weight=config["weight"],
        costs=costs,
    )


# Section function decoding.
@dataclass(frozen=True)
class SectionDecodeResult:

    frame_labels: np.ndarray
    segments: list[tuple[int, int, int]]
    diagnostics: Mapping[str, Any]


def _as_logits_and_grid(
    logits: np.ndarray, grid_times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=float)
    grid = np.asarray(grid_times_s, dtype=float).reshape(-1)
    return (values, grid)


def _smooth_scores(scores: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames == 1 or scores.shape[0] == 0:
        return scores.copy()
    left = window_frames // 2
    right = window_frames - left - 1
    padded = np.pad(scores, ((left, right), (0, 0)), mode="edge")
    kernel = np.full(window_frames, 1.0 / window_frames)
    return np.stack(
        [
            np.convolve(padded[:, index], kernel, mode="valid")
            for index in range(scores.shape[1])
        ],
        axis=1,
    )


def _phrase_prior(
    values: np.ndarray | None, length: int, weight: float
) -> tuple[np.ndarray, bool, str | None]:
    if values is None:
        return (np.zeros(length, dtype=float), False, "missing_phrase_prior")
    prior = np.asarray(values, dtype=float).reshape(-1)
    if np.all(prior == 0.0) or weight == 0.0:
        return (np.zeros(length, dtype=float), False, "empty_phrase_prior")
    return (prior * float(weight), True, None)


def _best_other(previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if previous.size == 1:
        return (np.full(1, -np.inf), np.full(1, -1, dtype=int))
    first = int(np.argmax(previous))
    masked = previous.copy()
    masked[first] = -np.inf
    second = int(np.argmax(masked))
    best_values = np.where(
        np.arange(previous.size) == first, previous[second], previous[first]
    )
    best_indices = np.where(np.arange(previous.size) == first, second, first)
    return (best_values, best_indices.astype(int))


def _viterbi(
    scores: np.ndarray, switch_penalty: float, switch_discount: np.ndarray
) -> np.ndarray:
    time_steps, classes = scores.shape
    if time_steps == 0:
        return np.zeros(0, dtype=int)
    dp = scores[0].copy()
    parents = np.full((time_steps, classes), -1, dtype=int)
    for frame in range(1, time_steps):
        other_values, other_indices = _best_other(dp)
        stay = dp
        switch = other_values - max(0.0, switch_penalty - float(switch_discount[frame]))
        choose_switch = switch > stay
        parents[frame] = np.where(choose_switch, other_indices, np.arange(classes))
        dp = scores[frame] + np.maximum(stay, switch)
    labels = np.empty(time_steps, dtype=int)
    labels[-1] = int(np.argmax(dp))
    for frame in range(time_steps - 1, 0, -1):
        labels[frame - 1] = parents[frame, labels[frame]]
    return labels


def decode_section_logits(
    logits: np.ndarray,
    grid_times_s: np.ndarray,
    *,
    switch_penalty: float = 0.0,
    phrase_boundary_probabilities: np.ndarray | None = None,
    phrase_soft_weight: float = 0.0,
) -> SectionDecodeResult:
    values, grid = _as_logits_and_grid(logits, grid_times_s)
    scores = _smooth_scores(values, 1)
    discount, prior_used, fallback_reason = _phrase_prior(
        phrase_boundary_probabilities, values.shape[0], phrase_soft_weight
    )
    labels = _viterbi(scores, float(switch_penalty), discount)
    segments = [
        (int(start), int(end), int(label)) for start, end, label in rle_compress(labels)
    ]
    return SectionDecodeResult(
        frame_labels=labels,
        segments=segments,
        diagnostics={
            "algorithm": "viterbi",
            "class_count": int(values.shape[1]),
            "smoothing_window_frames": 1,
            "switch_penalty": float(switch_penalty),
            "min_segment_frames": 1,
            "max_segment_frames": None,
            "phrase_prior_used": prior_used,
            "fallback_reason": fallback_reason,
            "complexity": "O(T*C)",
        },
    )


def decode_boundary_supported_section(
    logits,
    boundary_times,
    boundary_scores,
    *,
    penalty,
    inside_scale,
    boundary_scale,
    width_s,
    stride=8,
    frame_rate=25.0,
):
    values = np.asarray(logits)
    native = values[::stride] * (stride / 8.0)
    grid = np.arange(len(native)) * stride / frame_rate
    support = np.zeros(len(native))
    for time, score in zip(boundary_times, boundary_scores):
        support = np.maximum(
            support, float(score) * np.exp(-0.5 * ((grid - time) / width_s) ** 2)
        )
    scale = inside_scale - (inside_scale - boundary_scale) * support
    base = float(penalty) * inside_scale
    result = decode_section_logits(
        native,
        grid,
        switch_penalty=base,
        phrase_boundary_probabilities=base - float(penalty) * scale,
        phrase_soft_weight=1.0,
    )
    return np.repeat(result.frame_labels, stride)[: len(values)]


# Section boundary segmentation.
def segment_dispersion(probabilities, native_peaks, stride=8):
    p = np.asarray(probabilities, float)
    anchors = np.r_[0, np.asarray(native_peaks, int), len(p)]
    prefix = np.vstack([np.zeros(p.shape[1]), p.cumsum(axis=0)])
    squared = np.r_[0.0, (p * p).sum(axis=1).cumsum()]
    counts = anchors[:, None] - anchors[None, :]
    sums = prefix[anchors][:, None, :] - prefix[anchors][None, :, :]
    error = squared[anchors][:, None] - squared[anchors][None, :]
    error -= (sums * sums).sum(axis=2) / np.maximum(1, counts)
    return np.tril(np.maximum(0.0, error) * stride / 25, -1)


def segment_sequence(
    times, probabilities, *, threshold, weight, costs, segment_costs, segment_weight
):
    times = np.asarray(times, float)
    n = len(times)
    if not n:
        return times.copy()
    rewards = logit(np.clip(probabilities, 1e-06, 1 - 1e-06)) - logit(threshold)
    best = rewards - segment_weight * segment_costs[1 : n + 1, 0]
    parents = np.full(n, -1, int)
    for i in range(1, n):
        transitions = (
            best[:i]
            - weight * costs[i, :i]
            - segment_weight * segment_costs[i + 1, 1 : i + 1]
            + rewards[i]
        )
        parent = int(transitions.argmax())
        if transitions[parent] > best[i]:
            best[i] = transitions[parent]
            parents[i] = parent
    empty = -segment_weight * segment_costs[-1, 0]
    final = best - segment_weight * segment_costs[-1, 1 : n + 1]
    endpoint = int(final.argmax())
    if final[endpoint] <= empty:
        return np.zeros(0)
    path = [endpoint]
    while parents[path[-1]] >= 0:
        path.append(int(parents[path[-1]]))
    return times[path[::-1]]


def decode_section_boundaries(candidates, config, single_costs, segments=None):
    peaks, scores = candidates
    costs = single_costs
    if config["family"] == "segment_consistency":
        return segment_sequence(
            peaks / 25,
            scores,
            threshold=config["threshold"],
            weight=config["weight"],
            costs=costs,
            segment_costs=segments,
            segment_weight=config["segment_weight"],
        )
    return select_soft_phrase_sequence(
        peaks / 25,
        scores,
        threshold=config["threshold"],
        weight=config["weight"],
        costs=costs,
    )


def baseline_joint_emissions(root, quality):
    r = log_softmax(root, axis=1)
    q = log_softmax(quality, axis=1)
    return np.column_stack(
        (q[:, 0], (r[:, :, None] + q[:, None, 1:]).reshape(len(r), -1))
    )


def chord_frames(song, config):
    stride = config["stride"]
    window = config["window"]
    penalty = config["penalty"]
    q = uniform_filter1d(
        song["chord_quality_logits"].astype(float),
        size=window,
        axis=0,
        mode="nearest",
    )[::stride]
    r = uniform_filter1d(
        song["chord_root_logits"].astype(float),
        size=window,
        axis=0,
        mode="nearest",
    )[::stride]
    scores = baseline_joint_emissions(r, q) * (stride / 4)
    states = _root_viterbi(scores, np.full(max(0, len(scores) - 1), penalty))
    states = np.repeat(states, stride)[: len(song["chord_quality_logits"])]
    quality = np.where(states == 0, 0, (states - 1) % (q.shape[1] - 1) + 1)
    roots = np.where(states == 0, -1, (states - 1) // (q.shape[1] - 1))
    return (np.where(quality == 0, -1, roots), quality)


def chord_prediction(song, config):
    roots, quality = chord_frames(song, config)
    starts = np.r_[
        0, np.flatnonzero((roots[1:] != roots[:-1]) | (quality[1:] != quality[:-1])) + 1
    ]
    ends = np.r_[starts[1:], len(roots) - 1]
    return {
        "chord_intervals_s": np.column_stack((starts, ends)) / 25,
        "chord_root_idx": roots[starts],
        "chord_quality_idx": quality[starts],
    }


def prepare_reference_songs(songs):
    for s in songs:
        intervals, roots, quality = extract_gt_chord(s["gt"], VOCAB_PATH)
        intervals = np.maximum(np.asarray(intervals, float), 0)
        keep = intervals[:, 1] > intervals[:, 0]
        s["chord_reference"] = (
            intervals[keep],
            _chord_labels(np.asarray(roots)[keep], np.asarray(quality)[keep], VOCAB),
        )
        s["tactus_ref"] = extract_gt_tactus(s["gt"])
        s["downbeat_ref"] = extract_gt_downbeat(s["gt"])


def chord_scores(song, config):
    ref, labels = song["chord_reference"]
    pred = chord_prediction(song, config)
    est = pred["chord_intervals_s"]
    keep = est[:, 1] > est[:, 0]
    elabels = _chord_labels(
        pred["chord_root_idx"][keep], pred["chord_quality_idx"][keep], VOCAB
    )
    est, elabels = mir_eval.util.adjust_intervals(
        est[keep], elabels, float(ref.min()), float(ref.max()), "N", "N"
    )
    keep = est[:, 1] > est[:, 0]
    est = est[keep]
    elabels = [v for v, k in zip(elabels, keep) if k]
    merged, a, b = mir_eval.util.merge_labeled_intervals(ref, labels, est, elabels)
    weights = mir_eval.util.intervals_to_durations(merged)
    return {
        k: float(
            mir_eval.chord.weighted_accuracy(getattr(mir_eval.chord, k)(a, b), weights)
        )
        for k in KEYS
    }


def beat_times(song, head, params):
    z = song[
        "tactus_enhanced_logits" if head == "tactus" else "downbeat_logits"
    ].reshape(-1)
    return decode_frozen_beat(z, np.arange(len(z)) / 25, head, params["beat"])


# Rhythm-supported 181-state chord decoding.
def peak_support(logits, *, stride=1, width_s=0.16, distance=2):
    values = np.asarray(logits, dtype=float).reshape(-1)
    support = np.zeros(len(values))
    if not len(values):
        return support
    probabilities = expit(values[::stride])
    peaks, _ = find_peaks(probabilities, height=0.05, distance=distance)
    sigma = width_s * 25
    radius = int(np.ceil(3 * sigma))
    for native in peaks:
        frame = native * stride
        lo = max(0, frame - radius)
        hi = min(len(values), frame + radius + 1)
        evidence = probabilities[native] * np.exp(
            -0.5 * ((np.arange(lo, hi) - frame) / sigma) ** 2
        )
        support[lo:hi] = np.maximum(support[lo:hi], evidence)
    return support


def prediction_support(song, config):
    width = config.get("beat_width_s", 0.08)
    return {
        "beat": peak_support(song["tactus_enhanced_logits"], width_s=width),
        "downbeat": peak_support(song["downbeat_logits"], width_s=width),
    }


def transition_penalties(streams, config, chord_stride=4):
    beat = np.asarray(streams["beat"])[chord_stride::chord_stride]
    down = np.asarray(streams["downbeat"])[chord_stride::chord_stride]
    multiplier = 1 + config.get("offbeat_boost", 0.0) * (1 - np.maximum(beat, down))
    for name in ("beat", "downbeat"):
        weight = config.get(name + "_weight", 0.0)
        if not weight:
            continue
        if name not in streams:
            raise ValueError(f"和弦解码启用 {name}_weight，但预测流缺少 {name}")
        multiplier = (
            multiplier - weight * np.asarray(streams[name])[chord_stride::chord_stride]
        )
    return float(config["penalty"]) * np.clip(multiplier, 0.1, 3.0)


def joint_emissions(root_logits, quality_logits, stride=4):
    r = log_softmax(np.asarray(root_logits, float)[::stride], axis=1)
    q = log_softmax(np.asarray(quality_logits, float)[::stride], axis=1)
    return np.column_stack(
        (q[:, 0], (r[:, :, None] + q[:, None, 1:]).reshape(len(r), -1))
    ) * (stride / 4)


def predict_chords(song, config):
    chord_stride = config.get("chord_native_stride", 4)
    key = config.get("beat_width_s", 0.08)
    if key not in song["support_cache"]:
        song["support_cache"][key] = prediction_support(song, config)
    penalty = transition_penalties(
        song["support_cache"][key], config, chord_stride=chord_stride
    )
    if (
        len(song["emissions"])
        != (len(song["chord_root_logits"]) + chord_stride - 1) // chord_stride
    ):
        raise ValueError("和弦发射时间网格与冻结chord_native_stride不一致")
    states = np.repeat(_root_viterbi(song["emissions"], penalty), chord_stride)[
        : len(song["chord_root_logits"])
    ]
    quality = np.where(states == 0, 0, (states - 1) % 15 + 1)
    roots = np.where(states == 0, -1, (states - 1) // 15)
    starts = np.r_[
        0, np.flatnonzero((roots[1:] != roots[:-1]) | (quality[1:] != quality[:-1])) + 1
    ]
    ends = np.r_[starts[1:], len(roots) - 1]
    return {
        "chord_intervals_s": np.column_stack((starts, ends)) / 25,
        "chord_root_idx": roots[starts],
        "chord_quality_idx": quality[starts],
    }


def score_chords(song, pred):
    ref, labels = song["chord_reference"]
    est = pred["chord_intervals_s"]
    keep = est[:, 1] > est[:, 0]
    names = _chord_labels(
        pred["chord_root_idx"][keep], pred["chord_quality_idx"][keep], VOCAB
    )
    est, names = mir_eval.util.adjust_intervals(
        est[keep], names, float(ref.min()), float(ref.max()), "N", "N"
    )
    keep = est[:, 1] > est[:, 0]
    est = est[keep]
    names = [s for s, k in zip(names, keep) if k]
    merged, a, b = mir_eval.util.merge_labeled_intervals(ref, labels, est, names)
    durations = mir_eval.util.intervals_to_durations(merged)
    return {
        k: float(
            mir_eval.chord.weighted_accuracy(
                getattr(mir_eval.chord, k)(a, b), durations
            )
        )
        for k in KEYS
    }


def section_distance_steps(old_steps: int, old_stride: int, new_stride: int) -> int:
    return max(1, round(old_steps * old_stride / new_stride))
