#!/usr/bin/env python3
"""trop2_cis-dimer_inhibitor - 一键生成最终结果文件 (predict.py).

按大赛《代码提交要求》第四节输出标准化候选清单 results/results.csv
(UTF-8)，必填字段：
    候选编号 / 所属赛道 / 候选序列 / 关键预测指标 /
    对应模型与运行版本 / 结构文件名 / 备注

数据来源: outputs/ 下最新 (或 --run-id 指定) 的 M10 排序结果；
若没有任何已完成运行，则先自动执行完整流水线 (M01-M10)。

用法:
    python predict.py                    # 使用最新 run, 没有则先全流程运行
    python predict.py --run-id run_xxx   # 指定 run 目录
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

TRACK = "生命科学挑战赛-蛋白质设计 (TROP2 cis-dimer inhibitor)"
PLATFORM_VERSION = "trop2_cis-dimer_inhibitor v1.0.0"

# 关键预测指标列 (原始值 + 方向在 docs/acceptance_mapping.md 附录A)
METRIC_COLS = [
    "robust_selectivity",
    "positive_state_pass_rate",
    "t88_terminal_contact",
    "t88_contact_occupancy",
    "complex_iptm",
    "intact_trop2_risk",
    "epcam_risk",
    "cis_block_score",
    "trans_occlusion_score",
    "fold_plddt",
    "aggregation_risk",
    "solubility_score",
    "uncertainty",
]


def latest_run(outputs: Path) -> Path | None:
    runs = sorted(p for p in outputs.glob("run_*") if (p / "candidate_metrics.csv").exists())
    return runs[-1] if runs else None


def run_project_name(run_dir: Path) -> str | None:
    """Project name recorded in a run's resolved_config.yaml (None if absent)."""
    rc = run_dir / "resolved_config.yaml"
    if not rc.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(rc.read_text(encoding="utf-8")) or {}
        return (data.get("project") or {}).get("name")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="generate standardised results.csv")
    parser.add_argument("--run-id", default=None, help="outputs/<run_id> to package")
    args = parser.parse_args()

    outputs = ROOT / "outputs"
    run_dir = outputs / args.run_id if args.run_id else latest_run(outputs)

    # audit fix: auto-picking "latest run" previously packaged SMOKE runs into
    # results.csv without any warning - now the run's project name must match
    # the production config (explicit --run-id bypasses the check on purpose)
    if not args.run_id and run_dir is not None:
        try:
            import yaml
            default_project = ((yaml.safe_load(
                (ROOT / "configs" / "trop2_v1.yaml").read_text(encoding="utf-8")) or {})
                .get("project") or {}).get("name")
        except Exception:
            default_project = None
        got = run_project_name(run_dir)
        if default_project and got and got != default_project:
            print(f"[predict.py] ERROR: latest run '{run_dir.name}' belongs to "
                  f"project '{got}' (expected '{default_project}' - e.g. a gpu "
                  f"smoke run would leak into results.csv). Re-run production or "
                  f"pass --run-id <id> to package a specific run.", file=sys.stderr)
            return 1

    if run_dir is None or not (run_dir / "candidate_metrics.csv").exists():
        print("[predict.py] no completed run found - executing full pipeline first")
        from trop2_design.cli import run as cli_run

        cli_run(project=ROOT / "configs" / "trop2_v1.yaml",
                tools=ROOT / "configs" / "tools.yaml",
                run_id=None, stages=None)
        run_dir = latest_run(outputs)
        if run_dir is None:
            print("[predict.py] ERROR: pipeline produced no results", file=sys.stderr)
            return 1

    import pandas as pd

    df = pd.read_csv(run_dir / "candidate_metrics.csv")
    manifest = {}
    mf = run_dir / "run_manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))

    rows = []
    for _, r in df.iterrows():
        # 结构文件: 正状态复合物 + 装配叠合 (以候选为前缀)
        struct_files = sorted(
            p.name for p in (run_dir / "complexes" / "positive").glob(f"{r.candidate_id}_*.cif"))
        struct_files += sorted(
            p.name for p in (run_dir / "assembly_overlays").glob(f"{r.candidate_id}_*.cif"))
        row = {
            "候选编号(candidate_id)": r.candidate_id,
            "设计名(design_name)": r.design_name,
            "所属赛道(track)": TRACK,
            "候选序列(sequence)": r.sequence,
            "序列长度(length)": len(str(r.sequence)),
            "Pareto层级(pareto_rank)": r.pareto_rank,
            "硬门槛状态(hard_filter_status)": r.hard_filter_status,
        }
        for col in METRIC_COLS:
            src_col = col
            if col == "intact_trop2_risk":
                src_col = "intact_risk"
            elif col == "epcam_risk":
                src_col = "epcam_risk"
            elif col == "cis_block_score":
                src_col = "cis_block"
            elif col == "trans_occlusion_score":
                src_col = "trans_occlusion"
            elif col == "t88_terminal_contact":
                src_col = "t88_contact"
            row[col] = r.get(src_col)
        row["对应模型与运行版本(model_and_version)"] = (
            f"{PLATFORM_VERSION}; run_id={run_dir.name}; "
            f"metric_source=proxy(geometric); predictor=heuristic-geometry")
        row["随机种子(seed)"] = manifest.get("seed", "")
        row["结构文件(structure_files)"] = ";".join(struct_files)
        row["备注(notes)"] = (
            r.rejection_reasons if isinstance(r.rejection_reasons, str)
            and r.rejection_reasons else "")
        rows.append(row)

    out = ROOT / "results" / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")

    # 结构文件副本随清单提交 (命名与清单一致)
    structs_dir = out.parent / "structures"
    structs_dir.mkdir(exist_ok=True)
    copied = 0
    for _, r in df.iterrows():
        for sub in ("complexes/positive", "assembly_overlays"):
            for src in (run_dir / sub).glob(f"{r.candidate_id}_*.cif"):
                dst = structs_dir / src.name
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                    copied += 1

    n_pass = sum(1 for r in rows if r["硬门槛状态(hard_filter_status)"] == "pass")
    print(f"[predict.py] results.csv: {len(rows)} 候选 ({n_pass} 通过硬门槛)")
    print(f"[predict.py] 结构文件副本: {copied} 个 -> {structs_dir}")
    print(f"[predict.py] 输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
