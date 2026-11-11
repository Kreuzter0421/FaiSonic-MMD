# FaiSonic MMD

English | [中文](README_zh.md)

[Paper (arXiv:2611.13503)](https://arxiv.org/abs/2611.13503) · [PDF](https://arxiv.org/pdf/2611.13503) · [Dataset on Hugging Face](https://huggingface.co/datasets/FaiSonic/FaiSonic-MMD)

Run the following commands from the repository root.

## 1. Set up the environment

Use Python 3.10 or later on Linux with an NVIDIA GPU. Install PyTorch and torchaudio builds compatible with your CUDA environment, then install the remaining dependencies.

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

Obtain the MusicFM source code, `pretrained_msd.pt`, and `msd_stats.json`. Place the checkpoint and statistics file in `assets/musicfm/`. Replace the path below with the directory containing `model/musicfm_25hz.py`.

```bash
export MUSICFM_SOURCE="/path/to/musicfm"
```

## 2. Prepare the data

Download all `FaiSonicMMD.tar.gz.part*` files and `merge_parts.py` from GitHub Releases into the repository root, then merge and extract them.

```bash
python merge_parts.py
python -m tarfile -e FaiSonicMMD.tar.gz datasets
```

Place the corresponding recordings in `datasets/FaiSonicMMD/audios/`, using the same filename stems as the annotation files. Set the data and cache paths in `configs/data.yaml`, and the split path (`split_json`) in `configs/train.yaml`. See the [data guide](docs/data.md) for the annotation format.

## 3. Preprocess

```bash
PYTHONPATH=".:${MUSICFM_SOURCE}" python scripts/preprocess.py --config configs/data.yaml
```

## 4. Train

```bash
python scripts/train.py --config configs/train.yaml --run-id faisonic_mmd_paper_seed42
```

Resume training:

```bash
python scripts/train.py --config configs/train.yaml --run-id faisonic_mmd_paper_seed42 --resume model_checkpoints/paper/faisonic_mmd_paper_seed42/last.pt
```

## 5. Run inference on the validation and test sets

```bash
python scripts/infer.py --config configs/train.yaml --out results/inference/paper_e12
```

## 6. Evaluate

```bash
python scripts/evaluate.py --config configs/train.yaml --validation-logits results/inference/paper_e12/val_logits --test-logits results/inference/paper_e12/test_logits --out results/evaluation/paper_e12
```

## Command-line options

```bash
python scripts/preprocess.py --help
python scripts/train.py --help
python scripts/infer.py --help
python scripts/evaluate.py --help
```
