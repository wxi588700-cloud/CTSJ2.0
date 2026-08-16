#!/usr/bin/env python3
"""trop2_cis-dimer_inhibitor - 一键设计入口 (design.py).

等价于 CLI 的 prepare + generate + evaluate 阶段 (PRD M01-M09):
靶标标准化 -> R87-T88 裂解态构建 -> 表位分析 -> 候选生成 ->
序列设计 -> 正/负状态评估 -> 机制与可开发性评分。

最终排序与候选清单独由 predict.py / run.sh (全流程) 完成。

用法:
    python design.py                      # 全新设计运行
    python design.py --run-id run_xxx     # 复用已有 run 目录续跑
    python design.py --config configs/trop2_v1.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/trop2_v1.yaml")
    parser.add_argument("--tools", default="configs/tools.yaml")
    parser.add_argument("--run-id", default=None,
                        help="reuse an existing outputs/<run_id> directory")
    args = parser.parse_args()

    from trop2_design.cli import run as cli_run

    cli_run(
        project=ROOT / args.config,
        tools=ROOT / args.tools if Path(args.tools).exists() else None,
        run_id=args.run_id,
        stages="prepare,generate,evaluate",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
