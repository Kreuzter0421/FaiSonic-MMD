"""Run model inference and optional frozen decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", help="训练主配置")
    parser.add_argument("--run-id", help="默认使用训练配置的运行名")
    parser.add_argument("--epoch", type=int, help="默认使用训练配置的最后一轮")
    parser.add_argument("--checkpoint", help="可选：显式指定本次模型权重")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--partition", choices=("val", "test", "both"), help="默认推理val和test"
    )
    selection.add_argument(
        "--song-ids", nargs="+", help="直接指定歌曲ID，不需要划分或标注"
    )
    parser.add_argument("--bundle", help="已有的验证集冻结包；提供后额外输出最终预测")
    parser.add_argument("--out", help="推理输出目录，包含logits及可选的最终预测")
    args = parser.parse_args(argv)
    from src.inference.inference import collect_logits, run_spec

    spec = run_spec(ROOT, args)
    output = ROOT / (args.out or f"results/inference/{spec['run_id']}_e{spec['epoch']}")
    if args.bundle:
        from src.decoding.pipeline import load_components

        load_components(ROOT, spec, args.bundle)
    if args.song_ids is not None:
        partitions = ("custom",)
    else:
        from src.data.dataset import load_split

        load_split(ROOT / spec["split"], ROOT / spec["labels"])
        partitions = (
            ("val", "test") if args.partition in (None, "both") else (args.partition,)
        )
    for partition in partitions:
        logits = output / f"{partition}_logits"
        collect_logits(
            SimpleNamespace(
                **spec, partition=partition, song_ids=args.song_ids, out=str(logits)
            )
        )
        if args.bundle:
            from src.decoding.pipeline import infer

            result = infer(
                ROOT, spec, logits, output / f"{partition}_predictions", args.bundle
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[推理] {partition} 原始输出已写入 {logits}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
