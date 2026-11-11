# FaiSonic MMD

[English](README.md) | 中文

[论文（arXiv:2611.13503）](https://arxiv.org/abs/2611.13503) · [PDF](https://arxiv.org/pdf/2611.13503) · [Hugging Face 数据集](https://huggingface.co/datasets/FaiSonic/FaiSonic-MMD)

FaiSonic MMD 收录了 500 首流行歌曲，提供由专家标注并独立复核的乐句、段落、和弦和节奏标签。四类标注共用同一录音时间轴。本仓库提供使用这些标注进行联合训练与评估的多任务基线。

以下命令均在仓库根目录运行。

## 1. 准备环境

使用配备 NVIDIA GPU 的 Linux 环境，Python 版本不低于 3.10。先安装与 CUDA 环境匹配的 PyTorch 和 torchaudio，再安装其余依赖。

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

准备 MusicFM 源码、预训练权重 `pretrained_msd.pt` 和统计文件 `msd_stats.json`。将权重和统计文件放入 `assets/musicfm/`，并将下面的路径替换为包含 `model/musicfm_25hz.py` 的目录。

```bash
export MUSICFM_SOURCE="/path/to/musicfm"
```

## 2. 准备数据

从 GitHub Releases 下载全部 `FaiSonicMMD.tar.gz.part*` 分卷及 `merge_parts.py`，放入仓库根目录后合并解压。

```bash
python merge_parts.py
python -m tarfile -e FaiSonicMMD.tar.gz datasets
```

将对应录音放入 `datasets/FaiSonicMMD/audios/`，音频与标注文件使用相同的文件名主体。在 `configs/data.yaml` 中设置数据与缓存路径，在 `configs/train.yaml` 中设置划分文件路径 `split_json`。标注格式见[数据说明](docs/data_zh.md)。

## 3. 数据预处理

```bash
PYTHONPATH=".:${MUSICFM_SOURCE}" python scripts/preprocess.py --config configs/data.yaml
```

## 4. 训练

```bash
python scripts/train.py --config configs/train.yaml --run-id faisonic_mmd_paper_seed42
```

恢复训练：

```bash
python scripts/train.py --config configs/train.yaml --run-id faisonic_mmd_paper_seed42 --resume model_checkpoints/paper/faisonic_mmd_paper_seed42/last.pt
```

## 5. 验证集与测试集推理

```bash
python scripts/infer.py --config configs/train.yaml --out results/inference/paper_e12
```

## 6. 评估

```bash
python scripts/evaluate.py --config configs/train.yaml --validation-logits results/inference/paper_e12/val_logits --test-logits results/inference/paper_e12/test_logits --out results/evaluation/paper_e12
```

## 查看命令行参数

```bash
python scripts/preprocess.py --help
python scripts/train.py --help
python scripts/infer.py --help
python scripts/evaluate.py --help
```

## 引用

```bibtex
@article{zhang2026faisonicmmd,
  title = {{FaiSonic MMD}: A Popular Music Dataset with Temporally Aligned Multidimensional Annotations for Multitask Music Analysis},
  author = {Zhang, Tieyao and Liu, Yuke and Yu, Jiaxing and Wu, Xinda and Chen, Genfang},
  journal = {arXiv preprint arXiv:2611.13503},
  year = {2026},
  doi = {10.48550/arXiv.2611.13503}
}
```
