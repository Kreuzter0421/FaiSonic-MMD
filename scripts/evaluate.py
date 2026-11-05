"""Fit decoders and evaluate test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="训练主配置，自动读取模型、数据和划分路径",
    )
    parser.add_argument("--run-id", help="默认使用训练配置的运行名")
    parser.add_argument(
        "--epoch", type=int, help="默认评估训练配置的最后一轮，论文为12轮"
    )
    parser.add_argument("--checkpoint", help="可选：显式指定该运行、轮次的权重路径")
    parser.add_argument("--out", help="本次评估目录；包含冻结包和测试结果")
    parser.add_argument(
        "--validation-logits", required=True, help="推理入口生成的验证集logits目录"
    )
    parser.add_argument(
        "--test-logits", required=True, help="推理入口生成的测试集logits目录"
    )
    args = parser.parse_args(argv)
    from src.inference.inference import run_spec

    spec = run_spec(ROOT, args)
    from src.data.dataset import load_split

    load_split(ROOT / spec["split"], ROOT / spec["labels"])
    from src.decoding.pipeline import fit, score

    output = ROOT / (
        args.out or f"results/evaluation/{spec['run_id']}_e{spec['epoch']}"
    )
    bundle = output / "bundle"
    if (bundle / "formal_bundle.json").exists():
        raise FileExistsError(f"评估目录已有冻结包，请使用新的 --out：{output}")

    validation = ROOT / args.validation_logits
    print("[评估] 仅使用验证集拟合并冻结解码参数", flush=True)
    fit(ROOT, spec, validation, bundle)
    test = ROOT / args.test_logits
    print("[评估] 使用冻结参数解码测试集并计算指标", flush=True)
    result = score(ROOT, spec, test, output / "test", bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
