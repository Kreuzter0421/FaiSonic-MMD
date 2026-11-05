# FaiSonic MMD标签

[English](data.md) | 中文 · [代码使用说明](../README_zh.md)

标签压缩包包含500首歌曲的标注，以及训练400／验证50／测试50的数据划分。解压到`datasets/`后，数据位于`datasets/FaiSonicMMD/`。

```text
labels/<song_id>.json   # Song metadata and annotations.
split.json             # Train, validation, and test song IDs.
```

歌曲ID为JSON文件名去掉扩展名后的部分，音频和特征缓存须使用相同ID。三个划分列表互不重叠。

## 标注字段

每份JSON包含`metadata`、`structures`、`chords`、`audio_config`四个顶层对象。

| 对象 | 字段 | 含义 |
|---|---|---|
| `metadata` | `title`, `artist` | 歌名与演唱者 |
| `metadata` | `duration` | 录音时长，单位为秒 |
| `structures[]` | `level` | `phrase`（乐句）或`section`（段落） |
| `structures[]` | `text` | 乐句标签或段落功能 |
| `structures[]` | `audio_start_ms`, `duration_ms` | 区间起点与时长，单位为毫秒 |
| `chords[]` | `root` | 根音音高类别：0=C、1=C♯、…、11=B |
| `chords[]` | `quality` | 和弦性质 |
| `chords[]` | `audio_start_ms`, `duration_ms` | 区间起点与时长，单位为毫秒 |
| `audio_config` | `bpm` | 全局速度，单位为每分钟拍数 |
| `audio_config` | `numerator`, `denominator` | 默认拍号 |
| `audio_config` | `offset` | 小节网格偏移，单位为秒 |
| `audio_config` | `bar_timestamps` | 小节网格锚点，单位为毫秒 |
| `audio_config` | `time_signatures[]` | 变拍号：`measure_index`、`numerator`、`denominator` |
| `audio_config` | `bpm_automation` | 速度自动化：`enabled`、`points` |
| `audio_config.bpm_automation.points[]` | `measure_index`, `beat_index`, `bpm`, `curve` | 速度控制点位置与数值 |

变拍号和速度自动化字段为可选项，小节索引从0开始。

字母型乐句ID使用大写字母与ASCII单引号（`A`、`A'`、`A''`），乐句ID仅在各自歌曲内表示对应关系。

## 时间约定

各层标注共用录音时间轴，0秒对应录音起点。区间终点为`audio_start_ms + duration_ms`。乐句与段落边界独立，相邻同功能段落仍作为独立实例。

小节网格锚点换算到录音时间：

```text
audio_time_seconds = bar_timestamp / 1000 - offset
```

相邻锚点界定一个小节，节拍按该小节拍号插值生成；6/8、9/8、12/8复合拍分别对应2、3、4个tactus拍。可使用随包标签工具生成节拍与小节线事件。仅录音时长范围内的事件有效，插值所需的网格锚点应保留。

## 模型标签

JSON保留和弦性质名称。模型监督通过`configs/vocab/chord.json`和`configs/vocab/section.json`生成：和弦使用12类根音与包含`N`、`X`的16类性质，段落使用8类功能及背景类。标注名称与模型类别的对应关系以词表映射为准。

将匹配音频放入`audios/`，再按[代码使用说明](../README_zh.md)运行预处理，生成编码器特征和标签缓存。本目录不附带音频、Mel文件或生成的缓存。
