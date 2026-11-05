"""Extract full-song features and cache labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--log-dir", default="logs/preprocessing")
    parser.add_argument("--workers", type=int, default=8, help="音频预读线程数")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--muq-model", default="OpenMuQ/MuQ-large-msd-iter")
    parser.add_argument("--musicfm-model", default="assets/musicfm/pretrained_msd.pt")
    parser.add_argument("--musicfm-stats", default="assets/musicfm/msd_stats.json")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    from src.features.cache import run_feature_extraction

    status = run_feature_extraction(
        config,
        project_root=ROOT,
        log_dir=ROOT / args.log_dir,
        workers=args.workers,
        torch_threads=args.torch_threads,
        muq_model=args.muq_model,
        musicfm_model=ROOT / args.musicfm_model,
        musicfm_stats=ROOT / args.musicfm_stats,
    )
    if status:
        return status
    from src.data.labels import report_exit_code, run_label_preprocess

    print("[数据处理] 将原始JSON标注转换为标签缓存", flush=True)
    report = run_label_preprocess(config, mode="build", project_root=ROOT)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    target = ROOT / args.log_dir / "labels_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
