"""Run model training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse


def main():
    parser = argparse.ArgumentParser(description="训练 FaiSonic MMD 四任务模型")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    from src.training.train import train

    train(args.config, resume=args.resume, run_id=args.run_id)


if __name__ == "__main__":
    main()
