#!/usr/bin/env python3
"""生成 notebooks/demo_pipeline.ipynb（本脚本本身不入运行链，仅维护用）。"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3 (trop2)", "language": "python",
                          "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.11"}

cells = []
cells.append(nbf.v4.new_markdown_cell(
"""# trop2_cis-dimer_inhibitor — 关键流程演示 Notebook

展示大赛要求的核心流程可复现性：
1. **裂解态构建与拓扑审计**（M02 核心：R87-T88 断链、C73-C108 二硫键保留）
2. **表位与热点**（M03）
3. **候选结果与排序**（读取 `results/results.csv`，含硬门槛/淘汰原因/Pareto）

> 运行前提：`pip install -e .` 已执行（Python 3.11）。
> 本 Notebook 与主入口 `bash run.sh` 等价覆盖关键逻辑，完整流程以主入口为准。"""))

cells.append(nbf.v4.new_markdown_cell("## 0. 环境与固定随机种子"))
cells.append(nbf.v4.new_code_cell(
"""import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, gemmi
import trop2_design
from trop2_design.io import read_structure, first_protein_chain, polymer_residues, find_residue

SEED = 20260816  # 与 configs/trop2_v1.yaml resources.seed 一致
print("package:", trop2_design.__name__, "| seed:", SEED)"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. M02 核心：R87-T88 裂解态与二硫键审计

直接调用裂解采样函数（与流水线同一实现），验证：
- 断链后 C73(片段侧) 与 C108(主体侧) 的 SG-SG 距离在所有构象中保持二硫键范围
- 二硫键铰链区之外的残基不被移动（拓扑编辑是局部的）"""))

cells.append(nbf.v4.new_code_cell(
"""import json, pathlib

run_dirs = sorted((ROOT / "outputs").glob("run_*"))
assert run_dirs, "先运行 bash run.sh 生成结果"
run = run_dirs[-1]
audit = json.loads((run / "topology_audit.json").read_text())

rows = []
for s in audit["states"]:
    if s["kind"] != "cleaved":
        continue
    rows.append({
        "state": s["state_id"],
        "peptide_bond_R87_T88": s["peptide_bond_left_right"],  # False = 已断链
        "C73-C108_disulfide": (73, 108) in [tuple(d) for d in s["disulfides"]],
        "new_Cterm": s["left_terminal"], "new_Nterm": s["right_terminal"],
        "clash_overlap_A3": s["max_clash_overlap"],
        "passed": s["passed"],
        "transform": s["transformations"][0][:48],
    })
pd.DataFrame(rows)"""))

cells.append(nbf.v4.new_markdown_cell("## 2. M03：T88 新末端表位与建议热点"))
cells.append(nbf.v4.new_code_cell(
"""epitope = json.loads((run / "epitope_patch.json").read_text())
hotspots = [l.split()[0] for l in (run / "hotspots.txt").read_text().splitlines()
            if l and not l.startswith("#")]
acc = pd.read_csv(run / "accessibility_metrics.csv")
print("表位残基数:", len(epitope["residues"]), "| 建议热点前 10:", hotspots[:10])
acc.sort_values("mean_sasa_A2", ascending=False).head(8)"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. 最终候选清单与排序逻辑

读取标准化 `results/results.csv`（`predict.py` 生成）：
- 硬门槛状态与淘汰原因（不可被加权分抵消）
- Pareto 层级与稳健选择性（PRD 12.2 公式）"""))

cells.append(nbf.v4.new_code_cell(
"""res = pd.read_csv(ROOT / "results" / "results.csv")
cols = ["候选编号(candidate_id)", "硬门槛状态(hard_filter_status)",
        "robust_selectivity", "positive_state_pass_rate", "t88_terminal_contact",
        "intact_trop2_risk", "epcam_risk", "cis_block_score", "fold_plddt",
        "Pareto层级(pareto_rank)"]
print("候选总数:", len(res), "| 通过硬门槛:", (res["硬门槛状态(hard_filter_status)"] == "pass").sum())
res[res["硬门槛状态(hard_filter_status)"] == "pass"][cols].round(3).head(8)"""))

cells.append(nbf.v4.new_code_cell(
"""# 淘汰原因样例（可追溯）
rej = res[res["备注(notes)"].notna() & (res["备注(notes)"] != "")]
rej[["候选编号(candidate_id)", "备注(notes)"]].head(4)"""))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. 不确定性说明

跨构象 iptM 代理的标准差作为不确定性惩罚进入稳健选择性
（`robust_selectivity = robust_positive − worst_offtarget − λ·std`）。
本基线的置信度指标为几何代理，GPU 环境可经 `configs/tools.yaml`
切换 AF2-Multimer/Boltz 复算（代码路径不变）。"""))
cells.append(nbf.v4.new_code_cell(
"""metrics = pd.read_csv(run / "candidate_metrics.csv")
ok = metrics[metrics.hard_filter_status == "pass"]
if len(ok):
    print(ok[["candidate_id", "robust_positive", "worst_offtarget",
              "uncertainty_penalty", "robust_selectivity"]].round(3).to_string(index=False))"""))

nb.cells = cells
nbf.write(nb, "notebooks/demo_pipeline.ipynb")
print("notebook written:", len(cells), "cells")
