"""Extract, concatenate, and cache full-song encoder features."""

from __future__ import annotations

import gc
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SAMPLE_RATE_HZ = 24000

DEFAULT_FRAME_RATE_HZ = 25.0

AUDIO_EXTENSIONS = frozenset({".flac", ".wav", ".mp3", ".ogg", ".aac", ".m4a"})


@dataclass(frozen=True)
class CacheRunSummary:
    ready: int
    reused: int
    failed: list[str]
    failure_reasons: dict[str, str]

    @property
    def exit_code(self) -> int:
        return int(bool(self.failed))


def build_song_id_map(audio_paths: Sequence[Path | str]) -> dict[str, Path]:
    return {Path(path).stem: Path(path) for path in audio_paths}


def collect_audio_paths(
    audio_dir: Path | str, *, limit: int | None = None
) -> list[Path]:
    paths = sorted(
        (
            path
            for path in Path(audio_dir).iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    )
    return paths if limit is None else paths[:limit]


def load_audio_waveform(
    path: Path | str, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> tuple[np.ndarray, float]:
    import soundfile as sf

    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if int(source_rate) != sample_rate_hz:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=int(source_rate), target_sr=sample_rate_hz
        )
    return (waveform, float(len(waveform) / sample_rate_hz))


def prepare_single_layer_features(features: np.ndarray | Any) -> np.ndarray:
    return np.asarray(features, dtype=np.float16)


def build_single_layer_meta(
    *,
    backbone: str,
    layer: int,
    T: int,
    audio_duration_s: float,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
) -> dict[str, Any]:
    return {
        "backbone": backbone,
        "layer": int(layer),
        "T": int(T),
        "audio_duration_s": float(audio_duration_s),
        "frame_rate_hz": float(frame_rate_hz),
    }


def write_single_layer_cache(
    path: Path | str, features: np.ndarray, meta: dict[str, Any]
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez(
            handle,
            features=prepare_single_layer_features(features),
            meta=np.bytes_(json.dumps(meta, ensure_ascii=False)),
        )
    return {"path": str(target), "status": "written"}


def _infer_waveform(
    waveform: np.ndarray,
    *,
    provider: Callable[[np.ndarray], np.ndarray],
    chunk_sec: float,
    sample_rate: int,
) -> np.ndarray:
    chunk_samples = int(round(chunk_sec * sample_rate))
    if chunk_samples <= 0:
        return provider(waveform)
    return np.concatenate(
        [
            provider(waveform[start : start + chunk_samples])
            for start in range(0, len(waveform), chunk_samples)
        ],
        axis=0,
    )


def run_single_layer_cache(
    audio_paths: Sequence[Path | str],
    out_dir: Path | str,
    *,
    backbone: str,
    layer: int,
    provider: Callable[[np.ndarray], np.ndarray],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    audio_loader: Callable[..., tuple[np.ndarray, float]] = load_audio_waveform,
    sample_rate: int = SAMPLE_RATE_HZ,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    chunk_sec: float = 0,
) -> CacheRunSummary:
    ready = reused = 0
    output = Path(out_dir)
    for song_id, audio_path in build_song_id_map(audio_paths).items():
        target = output / f"{song_id}.npz"
        if target.is_file():
            reused += 1
            if on_progress:
                on_progress(
                    {"status": "skipped", "song_id": song_id, "cache_path": str(target)}
                )
            continue
        waveform, duration = audio_loader(audio_path, sample_rate)
        array = prepare_single_layer_features(
            _infer_waveform(
                waveform,
                provider=provider,
                chunk_sec=chunk_sec,
                sample_rate=sample_rate,
            )
        )
        write_single_layer_cache(
            target,
            array,
            build_single_layer_meta(
                backbone=backbone,
                layer=layer,
                T=array.shape[0],
                audio_duration_s=duration,
                frame_rate_hz=frame_rate_hz,
            ),
        )
        ready += 1
        if on_progress:
            on_progress(
                {"status": "ready", "song_id": song_id, "cache_path": str(target)}
            )
    return CacheRunSummary(ready=ready, reused=reused, failed=[], failure_reasons={})


def run_prefetched_cache(
    audio_paths: Sequence[Path | str],
    out_dir: Path | str,
    *,
    backbone: str,
    layer: int,
    provider: Callable[[np.ndarray], np.ndarray],
    workers: int = 8,
    audio_loader: Callable[..., tuple[np.ndarray, float]] = load_audio_waveform,
    chunk_sec: float = 0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> CacheRunSummary:
    songs = build_song_id_map(audio_paths)
    remaining, pending = (iter(songs.items()), deque())
    ready = reused = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit() -> None:
            item = next(remaining, None)
            if item is not None:
                song_id, path = item
                target = Path(out_dir) / f"{song_id}.npz"
                future = (
                    None
                    if target.is_file()
                    else pool.submit(audio_loader, path, SAMPLE_RATE_HZ)
                )
                pending.append((song_id, path, future))

        for _ in range(workers):
            submit()
        while pending:
            song_id, path, future = pending.popleft()
            if future is None:
                reused += 1
                status = "reused"
            else:
                waveform, duration = future.result()
                result = run_single_layer_cache(
                    [path],
                    out_dir,
                    backbone=backbone,
                    layer=layer,
                    provider=provider,
                    audio_loader=lambda _path, _sample_rate: (waveform, duration),
                    chunk_sec=chunk_sec,
                )
                ready += result.ready
                reused += result.reused
                status = "ready"
            if on_progress:
                on_progress(
                    {
                        "song_id": song_id,
                        "status": status,
                        "ready": ready,
                        "reused": reused,
                        "failed": 0,
                        "total": len(songs),
                    }
                )
            submit()
    return CacheRunSummary(ready=ready, reused=reused, failed=[], failure_reasons={})


def _extract_stream(paths, output, log_dir, *, backbone, layer, provider, workers):
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "progress.jsonl").open("a", encoding="utf-8", buffering=1) as log:

        def report(event):
            line = json.dumps(event, ensure_ascii=False)
            print(line, flush=True)
            log.write(line + "\n")

        result = run_prefetched_cache(
            paths,
            output,
            backbone=backbone,
            layer=layer,
            provider=provider,
            workers=workers,
            chunk_sec=30.0,
            on_progress=report,
        )
    (log_dir / "summary.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result.exit_code


def run_feature_extraction(
    config,
    *,
    project_root,
    log_dir,
    workers=8,
    torch_threads=4,
    muq_model="OpenMuQ/MuQ-large-msd-iter",
    musicfm_model="assets/musicfm/pretrained_msd.pt",
    musicfm_stats="assets/musicfm/msd_stats.json",
):
    import torch
    from src.features.encoders import load_muq_provider, load_musicfm_provider

    root = Path(project_root)
    paths = collect_audio_paths(root / config["sample_dir"] / "audios")
    if not paths:
        raise ValueError("未找到音频，请准备数据配置对应的 audios/ 目录")
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cache = config["mert"]
    for backbone, layer, destination in (
        ("muq", 6, cache["cache_dir"]),
        ("musicfm", 7, cache["secondary_cache_dir"]),
    ):
        print(f"[数据处理] {backbone}第{layer}层：30秒分窗提取并按时间拼接", flush=True)
        if backbone == "muq":
            provider = load_muq_provider(
                model_name=muq_model, device="cuda", dtype="float32", layer=layer
            )
        else:
            provider = load_musicfm_provider(
                model_path=root / musicfm_model,
                stats_path=root / musicfm_stats,
                device="cuda",
                dtype="float32",
                layer=layer,
            )
        status = _extract_stream(
            paths,
            root / destination,
            Path(log_dir) / backbone,
            backbone=backbone,
            layer=layer,
            provider=provider,
            workers=workers,
        )
        del provider
        gc.collect()
        torch.cuda.empty_cache()
        if status:
            return status
    return 0
