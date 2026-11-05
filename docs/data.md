# FaiSonic MMD annotations

English | [中文](data_zh.md) · [Code guide](../README.md)

The annotation archive contains labels for 500 songs and the train/validation/test split of 400/50/50 songs. Extract it into `datasets/` to create `datasets/FaiSonicMMD/`.

```text
labels/<song_id>.json   # Song metadata and annotations.
split.json             # Train, validation, and test song IDs.
```

The song ID is the JSON filename without its extension. Audio recordings and feature caches must use the same ID. The three split lists are disjoint.

## Annotation fields

Each JSON has four top-level objects: `metadata`, `structures`, `chords`, and `audio_config`.

| Object | Field | Meaning |
|---|---|---|
| `metadata` | `title`, `artist` | Song title and performing artist |
| `metadata` | `duration` | Recording duration in seconds |
| `structures[]` | `level` | `phrase` or `section` |
| `structures[]` | `text` | Phrase label or section function |
| `structures[]` | `audio_start_ms`, `duration_ms` | Interval start and duration in milliseconds |
| `chords[]` | `root` | Pitch class: 0=C, 1=C♯, …, 11=B |
| `chords[]` | `quality` | Chord quality |
| `chords[]` | `audio_start_ms`, `duration_ms` | Interval start and duration in milliseconds |
| `audio_config` | `bpm` | Global tempo in beats per minute |
| `audio_config` | `numerator`, `denominator` | Default time signature |
| `audio_config` | `offset` | Bar-grid offset in seconds |
| `audio_config` | `bar_timestamps` | Bar-grid anchors in milliseconds |
| `audio_config` | `time_signatures[]` | Time-signature changes: `measure_index`, `numerator`, `denominator` |
| `audio_config` | `bpm_automation` | Tempo automation: `enabled`, `points` |
| `audio_config.bpm_automation.points[]` | `measure_index`, `beat_index`, `bpm`, `curve` | Tempo control-point positions and values |

Time-signature changes and tempo automation are optional. Measure indices are zero-based.

Letter-based phrase IDs use uppercase letters and ASCII apostrophes (`A`, `A'`, `A''`). Phrase IDs are local to each song.

## Time conventions

All annotation layers refer to the same recording timeline, with time zero at the recording's start. Interval end time is `audio_start_ms + duration_ms`. Phrase and section boundaries are independent. Adjacent sections with the same function remain separate instances.

A bar-grid anchor maps to recording time as:

```text
audio_time_seconds = bar_timestamp / 1000 - offset
```

Consecutive anchors delimit a bar. Beats are interpolated within each bar using its time signature; compound meters 6/8, 9/8, and 12/8 use 2, 3, and 4 tactus beats respectively. Use the provided label utilities to generate beat and downbeat events. Only events within the recording's duration are valid; retain the grid anchors needed for interpolation.

## Model labels

The JSON annotations retain their chord-quality names. Model targets are generated using `configs/vocab/chord.json` and `configs/vocab/section.json`. Chord targets use 12 roots and 16 quality classes, including `N` and `X`; section targets use eight functions plus background. Vocabulary mappings define how annotation labels map to model classes.

Place matching audio in `audios/` and run preprocessing as described in the [code guide](../README.md) to create encoder features and label caches. Audio, Mel files, and generated caches are not included in this directory.
