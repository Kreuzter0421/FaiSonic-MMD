"""Load frozen decoders, generate predictions, and compute scores."""

from __future__ import annotations

import json
from pathlib import Path

import mir_eval
import numpy as np
import yaml
from scipy.special import softmax
from src.decoding.decoders import (
    beat_times,
    candidate_features,
    decode_boundary_supported_section,
    decode_section_boundaries,
    duration_cost,
    f1_pair,
    joint_emissions,
    predict_chords,
    predict_phrase_boundaries,
    prepare_reference_songs,
    score_chords,
    score_model,
    section_feature_names,
    segment_dispersion,
)
from src.decoding.fitting import fit as fit
from src.decoding.fitting import load_songs

FIELDS = ("phrase", "section", "section_boundary", "beat", "chord")
METRICS = (
    "ACC",
    "HR.5F1",
    "HR3.F1",
    "SectionBoundary.5F1",
    "SectionBoundary3.F1",
    "tactusF1",
    "downbeatF1",
    "root",
    "majmin",
    "thirds",
    "triads",
    "sevenths",
    "tetrads",
    "mirex",
)
NATIVE_STRIDES = {"section": 8, "chord": 4}


def active_protocol(root: Path, spec: dict) -> dict:
    from src import INTERNAL_HEADS, TASK_HEADS

    model_config = yaml.safe_load(
        _resolve(root, spec["model_config"]).read_text(encoding="utf-8")
    )
    heads = INTERNAL_HEADS
    concepts = tuple(sorted(TASK_HEADS))
    fields = FIELDS
    metrics = METRICS
    return {
        "active_concepts": concepts,
        "active_internal_heads": tuple(sorted(heads)),
        "fields": fields,
        "metric_names": metrics,
        "section_candidate_feature_names": section_feature_names(),
        "model_config": model_config,
    }


def decoder_native_strides(
    spec: dict, model_config: dict | None = None
) -> dict[str, int]:
    values = spec.get("decoder_native_strides")
    if values is None:
        return dict(NATIVE_STRIDES)
    if (
        not isinstance(values, dict)
        or set(values) != {"section", "chord"}
        or any((type(value) is not int or value < 1 for value in values.values()))
    ):
        raise ValueError("decoder_native_strides 必须包含正整数 section 与 chord")
    return dict(values)


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"正式解码缺少冻结文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _same_path(a: str | Path, b: str | Path, root: Path) -> bool:
    return _resolve(root, a).resolve() == _resolve(root, b).resolve()


def validate_logits_manifest(
    manifest: dict, spec: dict, partition: str, root: Path
) -> None:
    for key in ("run_id", "epoch", "checkpoint", "data_config"):
        if key not in manifest:
            raise ValueError(f"logits manifest 缺少 {key}")
        if key in ("checkpoint", "data_config"):
            equal = _same_path(manifest[key], spec[key], root)
        else:
            equal = manifest[key] == spec[key]
        if not equal:
            raise ValueError(
                f"logits {key} 与冻结模型不符：{manifest[key]} != {spec[key]}"
            )
    split = _read(_resolve(root, spec["split"]))
    expected = sorted(split[partition])
    actual = manifest.get("song_ids", [])
    if actual != expected or manifest.get("song_count") != len(expected):
        raise ValueError(f"logits {partition} 曲目身份/数量与划分不符")
    protocol = active_protocol(root, spec)
    recorded = manifest.get("active_concepts")
    if recorded is None and "phrase" in protocol["active_concepts"]:
        return
    if recorded is None or tuple(sorted(recorded)) != protocol["active_concepts"]:
        raise ValueError("logits active_concepts 与模型活动任务不符")


def validate_frozen_identity(frozen: dict, spec: dict) -> None:
    root = Path.cwd()
    for key in ("run_id", "epoch", "checkpoint", "split", "labels", "data_config"):
        if key not in frozen:
            raise ValueError(f"正式冻结包缺少 {key}")
        equal = (
            _same_path(frozen[key], spec[key], root)
            if key in ("checkpoint", "split", "labels", "data_config")
            else frozen[key] == spec[key]
        )
        if not equal:
            raise ValueError(f"正式冻结包 {key} 与当前模型不符")
    if frozen.get("fit_split") != spec["split"] + "#val":
        raise ValueError("正式冻结包 fit_split 不是本模型 val")
    if frozen.get("test_used_for_fit") is not False:
        raise ValueError("正式冻结包未证明 test_used_for_fit=false")
    model_config = (
        yaml.safe_load(_resolve(root, spec["model_config"]).read_text(encoding="utf-8"))
        if "model_config" in spec
        else None
    )
    if decoder_native_strides(frozen) != decoder_native_strides(spec, model_config):
        raise ValueError("正式冻结包 decoder_native_strides 与 job 不一致")
    protocol = active_protocol(root, spec)
    identity = (
        "active_concepts",
        "active_internal_heads",
        "metric_names",
        "section_candidate_feature_names",
    )
    if "phrase" not in protocol["active_concepts"] or any(
        (name in frozen for name in identity)
    ):
        for name in identity:
            if list(frozen.get(name, [])) != list(protocol[name]):
                raise ValueError(f"正式冻结包 {name} 与模型活动协议不一致")
        count = len(protocol["section_candidate_feature_names"])
        if frozen.get("section_candidate_feature_count") != count:
            raise ValueError("正式冻结包 Section 候选特征维数不一致")
        expected_components = set(protocol["fields"]) | {
            "section_boundary_formal",
            "section_function_support",
            "chord_rhythm",
        }
        expected_components.add("phrase_soft_duration")
        if set(frozen.get("components", {})) != expected_components:
            raise ValueError("正式冻结包 components 与活动任务不一致")


def component_paths(
    root: Path, spec: dict, bundle: str | Path | None = None
) -> dict[str, Path]:
    if bundle is not None:
        folder = _resolve(root, bundle)
        frozen = _read(folder / "formal_bundle.json")
        validate_frozen_identity(frozen, spec)
        return {key: folder / path for key, path in frozen["components"].items()}
    raise ValueError("请先在本模型验证集拟合，并显式提供 --bundle")


def load_components(root: Path, spec: dict, bundle: str | Path | None = None) -> dict:
    paths = component_paths(root, spec, bundle)
    params = {key: _read(path) for key, path in paths.items()}
    model_config = (
        yaml.safe_load(_resolve(root, spec["model_config"]).read_text(encoding="utf-8"))
        if "model_config" in spec
        else None
    )
    grid = decoder_native_strides(spec, model_config)
    checks = (
        ("section", "section_native_stride", "section"),
        ("section_boundary", "native_stride", "section"),
        ("section_boundary_formal", "section_native_stride", "section"),
        ("section_function_support", "section_native_stride", "section"),
        ("chord", "native_stride", "chord"),
        ("chord_rhythm", "chord_native_stride", "chord"),
    )
    for component, field, task in checks:
        if component not in params:
            continue
        value = params[component].get(field)
        if value is not None and value != grid[task]:
            raise ValueError(f"{component}.{field} 与冻结 decoder_native_strides 冲突")
    for component, value in params.items():
        config = value.get("selected", {}).get("config", {})
        if not isinstance(config, dict):
            continue
        for field, task in (
            ("section_native_stride", "section"),
            ("chord_native_stride", "chord"),
        ):
            if field in config and config[field] != grid[task]:
                raise ValueError(
                    f"{component}.selected.config.{field} 与冻结 decoder_native_strides 冲突"
                )
    params["_decoder_native_strides"] = grid
    protocol = active_protocol(root, spec)
    params["_active_concepts"] = protocol["active_concepts"]
    params["_metric_names"] = protocol["metric_names"]
    params["_section_candidate_feature_names"] = protocol[
        "section_candidate_feature_names"
    ]
    return params


def load_logits(
    root: Path, folder: str | Path, spec: dict, partition: str
) -> list[dict]:
    path = _resolve(root, folder)
    manifest = _read(path / "manifest.json")
    validate_logits_manifest(manifest, spec, partition, root)
    return load_songs(path, _resolve(root, spec["labels"]))


def predict_song(song: dict, params: dict) -> dict:
    # Phrase boundaries.
    phrase = params["phrase_soft_duration"]
    phrase_config = phrase["selected"]["config"]
    peaks, features = candidate_features(song, "phrase")
    scorer = phrase_config["scorer"]
    probabilities = score_model(
        features[:, : phrase["scorers"][scorer]["n_features"]],
        phrase["calibration_models"][scorer],
    )
    phrase_times = predict_phrase_boundaries(
        peaks, probabilities, phrase_config, phrase["duration_profile"]
    )
    # Section boundaries and functions.
    section = params["section_boundary_formal"]
    section_config = section["selected"]["config"]
    grid = params.get("_decoder_native_strides", NATIVE_STRIDES)
    section_stride = grid["section"]
    chord_stride = grid["chord"]
    boundary_peaks, boundary_features = candidate_features(
        song, "section_boundary", section_stride=section_stride
    )
    if boundary_features.shape[1] != len(params["_section_candidate_feature_names"]):
        raise ValueError("Section 候选特征列与冻结任务协议不一致")
    model = section["models"][section_config["scorer"]]
    boundary_probabilities = score_model(
        boundary_features[:, : len(model["coef"])], model
    )
    costs = duration_cost(
        boundary_peaks / 25, {**section["duration_profile"], "kind": "lognormal"}
    )
    segments = segment_dispersion(
        softmax(song["section_logits"][::section_stride], axis=1),
        boundary_peaks // section_stride,
        stride=section_stride,
    )
    boundary_times = decode_section_boundaries(
        (boundary_peaks, boundary_probabilities),
        section_config,
        costs,
        segments=segments,
    )
    lookup = {int(p): float(v) for p, v in zip(boundary_peaks, boundary_probabilities)}
    boundary_scores = np.asarray([lookup[int(round(t * 25))] for t in boundary_times])
    function_config = params["section_function_support"]["selected"]["config"]
    function_labels = decode_boundary_supported_section(
        song["section_logits"],
        boundary_times,
        boundary_scores,
        penalty=params["section"]["section_penalty"],
        stride=params["section"]["section_native_stride"],
        **function_config,
    )
    # Chords with predicted rhythm support.
    song["emissions"] = joint_emissions(
        song["chord_root_logits"], song["chord_quality_logits"], stride=chord_stride
    )
    song["support_cache"] = {}
    chord_config = params["chord_rhythm"]["selected"]["config"]
    chords = predict_chords(
        song,
        {
            **chord_config,
            "chord_native_stride": chord_stride,
            "section_native_stride": section_stride,
        },
    )
    # Beats and downbeats.
    beat_params = {"beat": params["beat"]}
    tactus = beat_times(song, "tactus", beat_params)
    downbeat = beat_times(song, "downbeat", beat_params)
    return {
        "phrase_boundary_times_s": phrase_times,
        "section_boundary_times_s": boundary_times,
        "section_function_labels": function_labels,
        "tactus_times_s": tactus,
        "downbeat_times_s": downbeat,
        **chords,
    }


def infer(
    root: Path, spec: dict, logits: str | Path, output: str | Path, bundle: str | Path
) -> dict:
    params = load_components(root, spec, bundle)
    folder = _resolve(root, logits)
    manifest = _read(folder / "manifest.json")
    for key in ("run_id", "epoch"):
        if manifest[key] != spec[key]:
            raise ValueError(f"推理输出 {key} 与解码包不符")
    if not _same_path(manifest["checkpoint"], spec["checkpoint"], root):
        raise ValueError("推理输出checkpoint与解码包不符")
    out = _resolve(root, output)
    for song_id in manifest["song_ids"]:
        with np.load(folder / f"{song_id}.npz", allow_pickle=False) as archive:
            song = {
                key: archive[key] for key in archive.files if key.endswith("logits")
            }
        predictions = predict_song(song, params)
        _write(
            out / "songs" / f"{song_id}.json",
            {
                "song_id": song_id,
                "frame_rate_hz": 25.0,
                "predictions": {
                    key: value.tolist() for key, value in predictions.items()
                },
            },
        )
    result = {
        "run_id": spec["run_id"],
        "epoch": spec["epoch"],
        "checkpoint": spec["checkpoint"],
        "bundle": str(bundle),
        "song_ids": manifest["song_ids"],
        "song_count": manifest["song_count"],
    }
    _write(out / "manifest.json", result)
    return result


def score(
    root: Path,
    spec: dict,
    logits: str | Path,
    output: str | Path,
    bundle: str | Path | None = None,
) -> dict:
    import numpy as np

    params = load_components(root, spec, bundle)
    songs = load_logits(root, logits, spec, "test")
    prepare_reference_songs(songs)
    out = _resolve(root, output)
    rows = []
    for song in songs:
        pred = predict_song(song, params)
        phrase_metrics = {}
        from src.evaluation.metrics import _score_phrase

        phrase = _score_phrase(
            song["gt"],
            {"phrase_boundary_times_s": pred["phrase_boundary_times_s"]},
            [0.5, 3.0],
        )
        phrase_metrics = {
            "HR.5F1": phrase["window_0.5"]["f_measure"],
            "HR3.F1": phrase["window_3"]["f_measure"],
        }
        section_pair = f1_pair(
            song, pred["section_boundary_times_s"], "section_boundary"
        )
        chord_scores = score_chords(song, pred)
        tactus_ref = np.asarray(song["tactus_ref"])
        tactus_pred = pred["tactus_times_s"]
        metrics = {
            **phrase_metrics,
            "SectionBoundary.5F1": float(section_pair[0]),
            "SectionBoundary3.F1": float(section_pair[1]),
            "tactusF1": float(
                mir_eval.beat.f_measure(
                    tactus_ref[tactus_ref >= 5], tactus_pred[tactus_pred >= 5]
                )
            ),
            "downbeatF1": float(
                mir_eval.beat.f_measure(
                    np.asarray(song["downbeat_ref"]), pred["downbeat_times_s"]
                )
            ),
            **chord_scores,
        }
        target = song["section_label"].reshape(-1)
        valid = target != 0
        correct = int(((pred["section_function_labels"] == target) & valid).sum())
        row = {
            "song_id": song["id"],
            "metrics": metrics,
            "section_correct": correct,
            "section_valid": int(valid.sum()),
            "predictions": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in pred.items()
            },
        }
        _write(out / "songs" / f"{song['id']}.json", row)
        rows.append(row)
    if not rows or not sum((row["section_valid"] for row in rows)):
        raise ValueError("test 没有可评分曲目或有效 Section 帧")
    summary = {
        "songs": len(rows),
        "epoch": spec["epoch"],
        "run_id": spec["run_id"],
        "checkpoint": spec["checkpoint"],
        "split": spec["split"] + "#test",
        "labels": spec["labels"],
        "decoder": "formal",
        "baseline_metric_count": len(params["_metric_names"]),
        "active_concepts": list(params["_active_concepts"]),
        "metric_names": list(params["_metric_names"]),
        "test_used_for_fit": False,
        "ACC": sum((row["section_correct"] for row in rows))
        / sum((row["section_valid"] for row in rows)),
        "metrics": {
            key: float(np.mean([row["metrics"][key] for row in rows]))
            for key in params["_metric_names"]
            if key != "ACC"
        },
    }
    _write(out / "summary.json", summary)
    return summary
