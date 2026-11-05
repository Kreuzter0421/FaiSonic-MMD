"""Fit and freeze decoder parameters on validation data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import find_peaks
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from src.decoding.decoders import (
    KEYS,
    candidate_features,
    chord_scores,
    decode_boundary_supported_section,
    decode_section_boundaries,
    decode_phrase_only,
    duration_cost,
    expected_f1_threshold,
    extract_phrase_only_features,
    f1_pair,
    joint_emissions,
    predict_chords,
    prepare_reference_songs,
    score_chords,
    score_model,
    section_distance_steps,
    segment_dispersion,
    select_soft_phrase_sequence,
)
from src.evaluation.metrics import _duration_from_gt

from src.evaluation.gt_extract import (
    extract_gt_downbeat,
    extract_gt_phrase,
    extract_gt_tactus,
)

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)

WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

ROOT = Path(__file__).resolve().parents[2]

DECODERS = ROOT / "configs/decoders.yaml"

FIELDS = ("phrase", "section", "section_boundary", "beat", "chord")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def read_song_output(path: Path, label_context: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        song = {key: archive[key] for key in archive.files}
    required = {"section_label", "section_boundary", "chord_root", "chord_quality"}
    if required <= song.keys():
        return song
    if "label_root" not in label_context:
        config = yaml.safe_load(
            (ROOT / label_context["data_config"]).read_text(encoding="utf-8")
        )
        label_root = ROOT / config["label_cache"]["cache_dir"]
        label_context["label_root"] = label_root
        label_context["index"] = json.loads(
            (label_root / "manifest.json").read_text(encoding="utf-8")
        )["songs"]
    from src.data.labels import ARRAY_DTYPES

    artifact = (
        label_context["label_root"] / label_context["index"][path.stem]["artifact"]
    )
    length = len(song["section_logits"])
    with np.load(artifact, allow_pickle=False) as labels:
        for key in ARRAY_DTYPES:
            values = labels[key][:length]
            if values.ndim != 1 or len(values) != length:
                raise ValueError(f"song_id={path.stem} 标签{key}与logits帧数不一致")
            song[key] = values
    return song


def load_songs(folder: Path, labels_dir: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    songs = []
    label_context = {"data_config": manifest.get("data_config")}
    for sid in manifest["song_ids"]:
        song = read_song_output(folder / f"{sid}.npz", label_context)
        song["id"] = sid
        song["gt"] = labels_dir / f"{sid}.json"
        song["duration"] = _duration_from_gt(song["gt"])
        ref = extract_gt_phrase(song["gt"])
        song["phrase_ref"] = np.unique(ref[(ref > 0) & (ref < song["duration"])])
        song["section_ref"] = (
            np.flatnonzero(song["section_boundary"].reshape(-1) > 0.5) / 25
        )
        songs.append(song)
    return songs


def fit_boundary_model(x, y, c):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-06] = 1
    model = LogisticRegression(C=c, max_iter=500, solver="lbfgs").fit(
        (x - mean) / scale, y
    )
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def calibrate_candidates(songs, task, n_features, c, section_stride=8):
    data = []
    for s in songs:
        peaks, x = candidate_features(s, task, section_stride=section_stride)
        ref = s["phrase_ref" if task == "phrase" else "section_ref"]
        y = (
            (np.abs(peaks[:, None] / 25 - ref[None, :]).min(1) <= 0.5).astype(int)
            if len(ref)
            else np.zeros(len(peaks), int)
        )
        data.append((peaks, x[:, :n_features], y))
    oof = [None] * len(songs)
    for fold in range(5):
        train = [d for i, d in enumerate(data) if i % 5 != fold]
        model = fit_boundary_model(
            np.concatenate([d[1] for d in train]),
            np.concatenate([d[2] for d in train]),
            c,
        )
        for i, d in enumerate(data):
            if i % 5 == fold:
                oof[i] = (d[0], score_model(d[1], model))
    fitted = fit_boundary_model(
        np.concatenate([d[1] for d in data]), np.concatenate([d[2] for d in data]), c
    )
    return (oof, fitted)


def load_templates() -> dict:
    return yaml.safe_load(DECODERS.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _near(peaks: np.ndarray, events_s: np.ndarray, tolerance_s: float) -> np.ndarray:
    if not len(peaks) or not len(events_s):
        return np.zeros(len(peaks), dtype=int)
    distances = np.min(np.abs(peaks[:, None] / 25.0 - events_s[None, :]), axis=1)
    return (distances <= tolerance_s).astype(int)


def _fit_logistic(
    features: list[np.ndarray], labels: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    x = np.concatenate(features).astype(float)
    y = np.concatenate(labels).astype(int)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit((x - mean) / scale, y)
    return (mean, scale, model)


def _metadata(
    data: dict,
    *,
    epoch: int,
    count: int,
    task: str,
    fit_split: str = "datasets/FaiSonicMMD/split.json#val",
) -> dict:
    data.pop("template_only", None)
    data.pop("requires_validation_fit", None)
    data.update(
        fit_split=fit_split,
        fit_epoch=epoch,
        fit_song_count=count,
        task_input=task,
        test_used_for_fit=False,
        fit_method="regularized_logistic_or_empirical_transition_prior; no_grid_search",
    )
    return data


def fit_peaks(args) -> None:
    if args.section_native_stride < 1 or args.chord_native_stride < 1:
        raise ValueError("解码原生倍率必须是正整数")
    section_stride = args.section_native_stride
    chord_stride = args.chord_native_stride
    folder = ROOT / args.logits
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    song_ids = manifest["song_ids"]
    epoch = manifest["epoch"]
    tasks = FIELDS
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    config_out = ROOT / args.config_out if args.config_out else out
    config_out.mkdir(parents=True, exist_ok=True)
    templates = load_templates()
    source = {task: templates[task] for task in tasks}
    phrase_x, phrase_y = ([], [])
    boundary_x, boundary_y = ([], [])
    beat_x = {"tactus": [], "downbeat": []}
    beat_y = {"tactus": [], "downbeat": []}
    section_switch = section_total = 0
    label_context = {"data_config": manifest.get("data_config")}
    for song_id in song_ids:
        gt = ROOT / args.label_dir / f"{song_id}.json"
        song = read_song_output(folder / f"{song_id}.npz", label_context)
        phrase_peaks, features = extract_phrase_only_features(
            song["phrase_logits"], source["phrase"]
        )
        phrase_x.append(features)
        phrase_y.append(_near(phrase_peaks, np.asarray(extract_gt_phrase(gt)), 0.5))
        boundary_logits = song["section_boundary_logits"].reshape(-1)[::section_stride]
        peaks, _ = find_peaks(boundary_logits)
        boundary_x.append(boundary_logits[peaks, None])
        boundary_ref = np.flatnonzero(song["section_boundary"].reshape(-1) > 0.5) / 25.0
        boundary_y.append(_near(peaks * section_stride, boundary_ref, 0.5))
        for head, logit_key, getter in (
            ("tactus", "tactus_enhanced_logits", extract_gt_tactus),
            ("downbeat", "downbeat_logits", extract_gt_downbeat),
        ):
            values = song[logit_key].reshape(-1)
            candidate, _ = find_peaks(expit(values), distance=2)
            candidate = candidate[expit(values[candidate]) >= 0.02]
            beat_x[head].append(values[candidate, None])
            beat_y[head].append(_near(candidate, np.asarray(getter(gt)), 0.07))
        native = song["section_label"].reshape(-1)[::section_stride]
        valid = (native[:-1] != 0) & (native[1:] != 0)
        section_switch += int(((native[:-1] != native[1:]) & valid).sum())
        section_total += int(valid.sum())
    phrase = source["phrase"]
    mean, scale, model = _fit_logistic(phrase_x, phrase_y)
    phrase.update(
        scaler_mean=mean.tolist(),
        scaler_scale=scale.tolist(),
        coefficients=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]),
        phrase_threshold=expected_f1_threshold(
            model.predict_proba((np.concatenate(phrase_x) - mean) / scale)[:, 1]
        ),
    )
    _write(
        config_out / "phrase.json",
        _metadata(
            phrase,
            epoch=epoch,
            count=len(song_ids),
            task="phrase_logits",
            fit_split=manifest["split"],
        ),
    )
    section = source["section"]
    observed = section_switch / section_total
    reference_switch = 1 - (1 - observed) ** (8 / section_stride)
    with np.load(folder / f"{song_ids[0]}.npz", allow_pickle=False) as first:
        class_count = first["section_logits"].shape[1]
    section.update(
        section_native_stride=section_stride,
        section_penalty=float(
            np.log((1 - reference_switch) * (class_count - 1) / reference_switch)
        ),
        transition_count=section_switch,
        valid_native_edges=section_total,
    )
    _write(
        config_out / "section.json",
        _metadata(
            section,
            epoch=epoch,
            count=len(song_ids),
            task="section_logits",
            fit_split=manifest["split"],
        ),
    )
    boundary = source["section_boundary"]
    source_stride = int(boundary["native_stride"])
    boundary["native_stride"] = section_stride
    boundary["min_distance_steps"] = section_distance_steps(
        int(boundary["min_distance_steps"]), source_stride, section_stride
    )
    mean, scale, model = _fit_logistic(boundary_x, boundary_y)
    probabilities = model.predict_proba((np.concatenate(boundary_x) - mean) / scale)[
        :, 1
    ]
    calibrated_threshold = expected_f1_threshold(probabilities)
    raw_threshold = float(
        mean[0]
        + scale[0]
        * (
            np.log(calibrated_threshold / (1 - calibrated_threshold))
            - model.intercept_[0]
        )
        / model.coef_[0, 0]
    )
    boundary.update(
        logit_threshold=raw_threshold,
        calibration={
            "mean": float(mean[0]),
            "scale": float(scale[0]),
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "expected_f1_threshold": calibrated_threshold,
        },
    )
    _write(
        config_out / "section_boundary.json",
        _metadata(
            boundary,
            epoch=epoch,
            count=len(song_ids),
            task="section_boundary_logits",
            fit_split=manifest["split"],
        ),
    )
    beat = source["beat"]
    for head in ("tactus", "downbeat"):
        mean, scale, model = _fit_logistic(beat_x[head], beat_y[head])
        probabilities = model.predict_proba(
            (np.concatenate(beat_x[head]) - mean) / scale
        )[:, 1]
        p = beat["heads"][head]
        p.update(
            mean=float(mean[0]),
            scale=float(scale[0]),
            coefficient=float(model.coef_[0, 0]),
            intercept=float(model.intercept_[0]),
            threshold=expected_f1_threshold(probabilities),
        )
    _write(
        config_out / "beat.json",
        _metadata(
            beat,
            epoch=epoch,
            count=len(song_ids),
            task="tactus_enhanced_logits;downbeat_logits",
            fit_split=manifest["split"],
        ),
    )
    chord = source["chord"]
    chord["native_stride"] = chord_stride
    _write(
        config_out / "chord.json",
        _metadata(
            chord,
            epoch=epoch,
            count=len(song_ids),
            task="chord_root_logits;chord_quality_logits",
            fit_split=manifest["split"],
        ),
    )
    _write(
        out / "fit_summary.json",
        {
            "run_id": manifest["run_id"],
            "epoch": epoch,
            "songs": len(song_ids),
            "fit_split": manifest["split"],
            "tasks": list(tasks),
            "method": "regularized logistic calibration and empirical Section transition prior; no exhaustive search",
            "section_switches": section_switch,
            "section_valid_native_edges": section_total,
            "samples": {
                **(
                    {
                        "phrase": [
                            int(np.concatenate(phrase_y).sum()),
                            len(np.concatenate(phrase_y)),
                        ]
                    }
                ),
                **(
                    {
                        "section_boundary": [
                            int(np.concatenate(boundary_y).sum()),
                            len(np.concatenate(boundary_y)),
                        ]
                    }
                ),
                **(
                    {
                        head: [
                            int(np.concatenate(beat_y[head]).sum()),
                            len(np.concatenate(beat_y[head])),
                        ]
                        for head in beat_y
                    }
                ),
            },
        },
    )
    print((out / "fit_summary.json").read_text(encoding="utf-8"), flush=True)


def fit_phrase_duration(songs):
    intervals = np.concatenate([np.diff(s["phrase_ref"]) for s in songs])
    intervals = intervals[intervals > 0]
    logs = np.log(intervals)
    center = np.median(logs)
    return {
        "short_s": float(np.quantile(intervals, 0.1)),
        "median_s": float(np.median(intervals)),
        "log_scale": float(max(0.15, 1.4826 * np.median(np.abs(logs - center)))),
        "fit_interval_count": len(intervals),
    }


def fit_soft_phrase(
    validation_logits: Path, parameters: Path, labels_dir: Path
) -> None:
    import numpy as np

    metadata = json.loads((validation_logits / "manifest.json").read_text())
    songs = load_songs(validation_logits, labels_dir)
    oof, model = calibrate_candidates(songs, "phrase", 9, 1.0)
    profiles = [
        fit_phrase_duration([s for i, s in enumerate(songs) if i % 5 != fold])
        for fold in range(5)
    ]
    costs = [
        duration_cost(peaks / 25, {**profiles[i % 5], "kind": "lognormal"})
        for i, (peaks, _) in enumerate(oof)
    ]
    baseline = json.loads((parameters / "original_decoders/phrase.json").read_text())
    reference = np.mean(
        [
            f1_pair(
                s,
                decode_phrase_only(s["phrase_logits"], baseline),
                "phrase",
            )
            for s in songs
        ],
        axis=0,
    )
    trials = []
    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            pairs = []
            for song, (peaks, scores), cost in zip(songs, oof, costs):
                times = select_soft_phrase_sequence(
                    peaks / 25, scores, threshold=threshold, weight=weight, costs=cost
                )
                pairs.append(f1_pair(song, times, "phrase"))
            pair = np.mean(pairs, axis=0)
            trials.append(
                {
                    "config": {
                        "method": "soft_dp",
                        "scorer": "cross",
                        "kind": "lognormal",
                        "threshold": threshold,
                        "weight": weight,
                    },
                    "val_f1": pair.tolist(),
                    "objective": float(pair.mean()),
                }
            )
    eligible = [r for r in trials if min(np.asarray(r["val_f1"]) - reference) >= -1e-10]
    selected = max(eligible or trials, key=lambda r: r["objective"])
    write_json(
        parameters / "phrase_validation_search.json",
        {
            "reference_val_f1": reference.tolist(),
            "fold_duration_profiles": profiles,
            "trials": trials,
            "eligible_count": len(eligible),
        },
    )
    frozen = {
        "selected": selected,
        "calibration_models": {"cross": model},
        "scorers": {"cross": {"n_features": 9, "C": 1.0}},
        "duration_profile": fit_phrase_duration(songs),
        "fit_epoch": metadata["epoch"],
        "fit_run_id": metadata["run_id"],
        "fit_split": metadata["split"],
        "fit_song_count": len(songs),
        "selection_rule": "Prefer candidates meeting both reference F1 thresholds; maximize mean out-of-fold song F1.",
        "eligible_count": len(eligible),
        "no_dominating_candidate_fallback": not bool(eligible),
        "test_used_for_fit": False,
        "strategy": "cross nine-feature logistic C=1; song-fivefold OOF; lognormal soft-duration DP",
        "learned_parameters_source": "Validation logits and labels from the specified checkpoint.",
    }
    write_json(parameters / "phrase_soft_duration.json", frozen)
    print(
        "own epoch phrase fitted", metadata["epoch"], json.dumps(selected), flush=True
    )


def fit_section_duration(songs):
    gaps = np.concatenate([np.diff(s["section_ref"]) for s in songs])
    gaps = gaps[gaps > 0]
    center = np.median(np.log(gaps))
    return {
        "median_s": float(np.median(gaps)),
        "short_s": float(np.quantile(gaps, 0.1)),
        "log_scale": float(
            max(0.15, 1.4826 * np.median(np.abs(np.log(gaps) - center)))
        ),
        "fit_interval_count": len(gaps),
    }


def calibrate_section_candidates(songs, section_stride=8):
    data = []
    for song in songs:
        peaks, x = candidate_features(
            song, "section_boundary", section_stride=section_stride
        )
        ref = song["section_ref"]
        y = (
            (np.abs(peaks[:, None] / 25 - ref[None, :]).min(axis=1) <= 0.5).astype(int)
            if len(ref)
            else np.zeros(len(peaks), int)
        )
        data.append((peaks, x, y))
    return data


def fit_section_oof(data, c):
    oof = [None] * len(data)
    for fold in range(5):
        train = [d for i, d in enumerate(data) if i % 5 != fold]
        model = fit_boundary_model(
            np.concatenate([d[1] for d in train]),
            np.concatenate([d[2] for d in train]),
            c,
        )
        for i, (peaks, x, _) in enumerate(data):
            if i % 5 == fold:
                oof[i] = (peaks, score_model(x, model))
    full = fit_boundary_model(
        np.concatenate([d[1] for d in data]), np.concatenate([d[2] for d in data]), c
    )
    return (oof, full)


def fit_section_boundary(songs, job, previous, output):
    section_stride = job.get("decoder_native_strides", {}).get("section", 8)
    data = calibrate_section_candidates(songs, section_stride=section_stride)
    feature_names = job.get("section_candidate_feature_names")
    feature_count = len(feature_names) if feature_names is not None else 9
    if any((features.shape[1] != feature_count for _, features, _ in data)):
        raise ValueError("Section 校准列数与活动任务声明不一致")
    streams = {}
    models = {}
    streams["base"], models["base"] = fit_section_oof(data, 1.0)
    baseline = previous["section_boundary"]["selected"]["config"]
    own_data = [(p, x[:, : baseline["n_features"]], y) for p, x, y in data]
    streams["own_baseline"], models["own_baseline"] = fit_section_oof(own_data, 1.0)
    folds = [
        fit_section_duration([s for i, s in enumerate(songs) if i % 5 != fold])
        for fold in range(5)
    ]
    costs = [
        duration_cost(p / 25, {**folds[i % 5], "kind": "lognormal"})
        for i, (p, _, _) in enumerate(data)
    ]
    segments = [
        segment_dispersion(
            softmax(s["section_logits"][::section_stride], axis=1),
            p // section_stride,
            stride=section_stride,
        )
        for s, (p, _, _) in zip(songs, data)
    ]
    configs = [
        {
            "family": "current",
            "scorer": "own_baseline",
            "threshold": baseline["threshold"],
            "weight": baseline["weight"],
        }
    ]
    configs += [
        {
            "family": "segment_consistency",
            "scorer": "base",
            "threshold": threshold,
            "weight": weight,
            "segment_weight": beta,
        }
        for threshold in (0.2, 0.3, 0.4)
        for weight in (0.05, 0.1, 0.25)
        for beta in (0.05, 0.1, 0.25, 0.5, 1.0)
    ]
    trials = []
    for config in configs:
        predictions = [
            decode_section_boundaries(cand, config, costs[i], segments=segments[i])
            for i, cand in enumerate(streams[config["scorer"]])
        ]
        pair = np.mean(
            [f1_pair(s, t, "section_boundary") for s, t in zip(songs, predictions)],
            axis=0,
        )
        trials.append(
            {
                "config": config,
                "val_f1": pair.tolist(),
                "objective": float(pair.mean()),
                "event_count": sum((len(t) for t in predictions)),
            }
        )
    reference = np.asarray(trials[0]["val_f1"])
    eligible = [
        r for r in trials[1:] if np.min(np.asarray(r["val_f1"]) - reference) >= -1e-10
    ]
    winner = {
        **max(eligible or trials[1:], key=lambda r: r["objective"]),
        "eligible": bool(eligible),
    }
    frozen = {
        "epoch": job["epoch"],
        "frontend": "conv_task_gate",
        "fit_split": job["fit_split"],
        "fit_song_count": len(songs),
        "section_native_stride": section_stride,
        "models": models,
        "duration_profile": fit_section_duration(songs),
        "fold_duration_profiles": folds,
        "family_winners": {
            "current": {**trials[0], "eligible": True},
            "segment_consistency": winner,
        },
        "selected": winner,
        "reference": trials[0],
        "validation_eligible": bool(eligible),
        "selection_rule": f"{feature_count}-feature calibration with five-fold song-level validation; prefer candidates meeting both reference F1 thresholds, then maximize mean F1.",
        "learned_parameters_source": "Validation logits and labels from the specified checkpoint.",
    }
    if feature_names is not None:
        frozen.update(
            section_candidate_feature_names=feature_names,
            section_candidate_feature_count=feature_count,
        )
    write_json(
        output / "validation_search.json",
        {"trials": trials, "eligible_count": len(eligible), "selected": winner},
    )
    write_json(output / "frozen_decoder.json", frozen)
    print("section fitted", job["epoch"], json.dumps(winner), flush=True)


def boundary_inputs(songs, frozen):
    config = frozen["selected"]["config"]
    section_stride = frozen.get("section_native_stride", 8)
    streams, _ = fit_section_oof(
        calibrate_section_candidates(songs, section_stride=section_stride),
        1.0,
    )
    profiles = [
        fit_section_duration([s for i, s in enumerate(songs) if i % 5 != fold])
        for fold in range(5)
    ]
    results = []
    for i, (song, (peaks, probabilities)) in enumerate(zip(songs, streams)):
        costs = duration_cost(peaks / 25, {**profiles[i % 5], "kind": "lognormal"})
        segments = segment_dispersion(
            softmax(song["section_logits"][::section_stride], axis=1),
            peaks // section_stride,
            stride=section_stride,
        )
        times = decode_section_boundaries(
            (peaks, probabilities), config, costs, segments=segments
        )
        lookup = {int(p): float(v) for p, v in zip(peaks, probabilities)}
        scores = np.asarray([lookup[int(round(t * 25))] for t in times])
        results.append((times, scores))
    return results


def score_section_function(songs, boundaries, params, config):
    correct = valid = 0
    for song, (times, scores) in zip(songs, boundaries):
        labels = decode_boundary_supported_section(
            song["section_logits"],
            times,
            scores,
            penalty=params["section_penalty"],
            stride=params["section_native_stride"],
            **config,
        )
        target = song["section_label"].reshape(-1)
        mask = target != 0
        n = int(mask.sum())
        c = int(((labels == target) & mask).sum())
        correct += c
        valid += n
    return correct / valid


def evaluate_chord_validation(songs, config):
    rows = []
    for song in songs:
        pred = predict_chords(song, config)
        row = score_chords(song, pred)
        rows.append(row)
    result = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    return {
        "config": config,
        "metrics": result,
        "objective": float(np.mean(list(result.values()))),
    }


def unique(configs):
    seen = set()
    out = []
    for c in configs:
        if not c.get("phrase_weight", 0) and (not c.get("section_weight", 0)):
            c.pop("structure_width_s", None)
        if not any(
            (c.get(k, 0) for k in ("beat_weight", "downbeat_weight", "offbeat_boost"))
        ):
            c.pop("beat_width_s", None)
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def rhythm_configs(base):
    return unique(
        [base.copy()]
        + [
            {
                **base,
                "beat_width_s": w,
                "offbeat_boost": off,
                "beat_weight": b,
                "downbeat_weight": d,
            }
            for w in (0.08, 0.16)
            for off in (0.0, 0.5)
            for b in (0.0, 0.5, 1.0)
            for d in (0.0, 0.5, 1.0)
        ]
    )


def fit_formal(job):
    import numpy as np

    output = ROOT / job["output_dir"]
    grid = job.get("decoder_native_strides", {"section": 8, "chord": 4})
    songs = load_songs(ROOT / job["validation_logits"], ROOT / job["labels"])
    prepare_reference_songs(songs)
    trials = []
    for penalty in (0.0, 1.0, 2.0, 4.0, 8.0):
        config = {
            "method": "joint",
            "stride": grid["chord"],
            "window": 1,
            "penalty": penalty,
        }
        values = [chord_scores(s, config) for s in songs]
        scores = {k: float(np.mean([v[k] for v in values])) for k in KEYS}
        trials.append(
            {
                "config": config,
                "validation": scores,
                "objective": float(np.mean(list(scores.values()))),
            }
        )
    write_json(
        output / "chord_frozen.json",
        {
            "selected": max(trials, key=lambda r: r["objective"]),
            "trials": trials,
            "fit_epoch": job["epoch"],
            "fit_run_id": job["item"]["run_id"],
            "test_used_for_selection": False,
            "chord_native_stride": grid["chord"],
        },
    )
    data = calibrate_section_candidates(songs, section_stride=grid["section"])
    feature_names = job.get("section_candidate_feature_names")
    feature_count = len(feature_names) if feature_names is not None else 9
    if any((features.shape[1] != feature_count for _, features, _ in data)):
        raise ValueError("Section 候选特征维数与任务声明不一致")
    streams, _ = fit_section_oof(data, 1.0)
    folds = [
        fit_section_duration([s for i, s in enumerate(songs) if i % 5 != fold])
        for fold in range(5)
    ]
    costs = [
        duration_cost(p / 25, {**folds[i % 5], "kind": "lognormal"})
        for i, (p, _, _) in enumerate(data)
    ]
    reference_trials = []
    for threshold in (0.2, 0.3, 0.4):
        for weight in (0.05, 0.1, 0.25):
            config = {
                "family": "current",
                "scorer": "base",
                "threshold": threshold,
                "weight": weight,
            }
            predictions = [
                decode_section_boundaries(c, config, costs[i])
                for i, c in enumerate(streams)
            ]
            pair = np.mean(
                [f1_pair(s, t, "section_boundary") for s, t in zip(songs, predictions)],
                axis=0,
            )
            reference_trials.append(
                {
                    "config": {**config, "n_features": feature_count},
                    "val_f1": pair.tolist(),
                    "objective": float(pair.mean()),
                }
            )
    reference = max(reference_trials, key=lambda r: r["objective"])
    write_json(
        output / "section_reference_validation.json",
        {"trials": reference_trials, "selected": reference},
    )
    fit_section_boundary(
        songs, job, {"section_boundary": {"selected": reference}}, output / "section"
    )
    frozen_path = output / "section/frozen_decoder.json"
    frozen = json.loads(frozen_path.read_text())
    frozen.update(
        frontend=job["item"]["shared_backbone_kind"], fit_run_id=job["item"]["run_id"]
    )
    if feature_names is not None:
        frozen.update(
            section_candidate_feature_names=feature_names,
            section_candidate_feature_count=feature_count,
        )
    write_json(frozen_path, frozen)


# Validation-only fitting and parameter freezing.
def fit(
    root: Path, spec: dict, validation_logits: str | Path, output: str | Path
) -> dict:
    from src.decoding.pipeline import (
        FIELDS,
        _read,
        _resolve,
        _write,
        active_protocol,
        decoder_native_strides,
        validate_logits_manifest,
    )

    model_config = yaml.safe_load(
        _resolve(root, spec["model_config"]).read_text(encoding="utf-8")
    )
    grid = decoder_native_strides(spec, model_config)
    protocol = active_protocol(root, spec)
    folder = _resolve(root, validation_logits)
    manifest = _read(folder / "manifest.json")
    validate_logits_manifest(manifest, spec, "val", root)
    out = _resolve(root, output)
    if (out / "formal_bundle.json").exists():
        raise FileExistsError(f"冻结包已存在，拒绝覆盖：{out}")
    work = out / "work"
    original_dir = work / "decoders/original_decoders"
    from types import SimpleNamespace

    fit_peaks(
        SimpleNamespace(
            logits=str(folder),
            out=str(original_dir),
            config_out=None,
            label_dir=str(_resolve(root, spec["labels"])),
            section_native_stride=grid["section"],
            chord_native_stride=grid["chord"],
        )
    )
    fit_soft_phrase(folder, work / "decoders", _resolve(root, spec["labels"]))
    shared_backbone_kind = model_config.get("architecture_kind", "conv_task_gate")
    job = {
        "item": {
            "run_id": spec["run_id"],
            "shared_backbone_kind": shared_backbone_kind,
        },
        "epoch": spec["epoch"],
        "fit_split": spec["split"] + "#val",
        "output_dir": str(work.relative_to(root)),
        "parameters_dir": str((work / "decoders").relative_to(root)),
        "validation_logits": str(folder.relative_to(root)),
        "labels": str(_resolve(root, spec["labels"])),
        "decoder_native_strides": grid,
        "active_concepts": list(protocol["active_concepts"]),
        "section_candidate_feature_names": list(
            protocol["section_candidate_feature_names"]
        ),
    }
    fit_formal(job)
    songs = load_songs(folder, _resolve(root, spec["labels"]))
    base = {key: _read(original_dir / f"{key}.json") for key in protocol["fields"]}
    section_frozen = _read(work / "section/frozen_decoder.json")
    boundaries = boundary_inputs(songs, section_frozen)
    identity = {"inside_scale": 1.0, "boundary_scale": 1.0, "width_s": 0.32}
    configs = [identity] + [
        {"inside_scale": a, "boundary_scale": b, "width_s": width}
        for a in (1.0, 1.5, 2.0, 3.0)
        for b in (0.0, 0.25, 0.5, 0.75, 1.0)
        for width in (0.32, 0.64, 1.28)
        if not a == b == 1.0
    ]
    trials = [
        {
            "config": config,
            "validation_ACC": score_section_function(
                songs, boundaries, base["section"], config
            ),
        }
        for config in configs
    ]
    _write(
        work / "section_function_support.json",
        {
            "selected": max(trials, key=lambda row: row["validation_ACC"]),
            "trials": trials,
            "fit_song_count": len(songs),
            "epoch": spec["epoch"],
            "fit_run_id": spec["run_id"],
            "test_used_for_selection": False,
            "section_native_stride": grid["section"],
        },
    )
    prepare_reference_songs(songs)
    for song in songs:
        song["emissions"] = joint_emissions(
            song["chord_root_logits"],
            song["chord_quality_logits"],
            stride=grid["chord"],
        )
        song["support_cache"] = {}
    chord_base = _read(work / "chord_frozen.json")["selected"]["config"]
    trials = [
        evaluate_chord_validation(songs, config)
        for config in rhythm_configs(
            {
                "penalty": chord_base["penalty"],
                "chord_native_stride": grid["chord"],
                "section_native_stride": grid["section"],
            }
        )
    ]
    _write(
        work / "chord_rhythm.json",
        {
            "selected": max(trials, key=lambda row: row["objective"]),
            "trials": trials,
            "fit_song_count": len(songs),
            "epoch": spec["epoch"],
            "fit_run_id": spec["run_id"],
            "test_used_for_selection": False,
            "chord_native_stride": grid["chord"],
        },
    )
    components = {key: f"original_decoders/{key}.json" for key in protocol["fields"]}
    components["phrase_soft_duration"] = "phrase_soft_duration.json"
    components.update(
        section_boundary_formal="section_boundary_formal.json",
        section_function_support="section_function_support.json",
        chord_rhythm="chord_rhythm.json",
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "original_decoders").mkdir(exist_ok=True)
    for key, relative in components.items():
        source = (
            original_dir / f"{key}.json"
            if key in FIELDS
            else {
                "phrase_soft_duration": work / "decoders/phrase_soft_duration.json",
                "section_boundary_formal": work / "section/frozen_decoder.json",
                "section_function_support": work / "section_function_support.json",
                "chord_rhythm": work / "chord_rhythm.json",
            }[key]
        )
        shutil.copyfile(source, out / relative)
    frozen = {
        key: spec[key]
        for key in ("run_id", "epoch", "checkpoint", "split", "labels", "data_config")
    }
    frozen.update(
        fit_split=spec["split"] + "#val",
        fit_song_count=len(songs),
        test_used_for_fit=False,
        components=components,
        protocol="faisonic_mmd_paper",
        decoder_native_strides=grid,
        active_concepts=list(protocol["active_concepts"]),
        active_internal_heads=list(protocol["active_internal_heads"]),
        metric_names=list(protocol["metric_names"]),
        section_candidate_feature_names=list(
            protocol["section_candidate_feature_names"]
        ),
        section_candidate_feature_count=len(
            protocol["section_candidate_feature_names"]
        ),
    )
    _write(out / "formal_bundle.json", frozen)
    return frozen
