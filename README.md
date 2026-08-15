# trop2-binder-platform

**TROP2 裂解态（R87-T88）特异性小蛋白药物从头设计与多目标计算评估平台** —— PRD v1.0 的完整参考实现。

> 边界声明：本系统是研究级决策支持工具，不输出“已成药”“已安全”或“具有临床疗效”的结论；所有结合、选择性、机制、免疫原性和药代判断必须由实验验证。

## 1. 功能总览（PRD M00–M10）

| 模块 | 目录 | 功能 |
|---|---|---|
| M00 | `src/trop2_design/workflow/` | 配置校验（Pydantic 严格模式）、DAG 编排、内容哈希缓存、断点恢复、运行审计（`run_manifest.json` / `task_status.csv` / `resolved_config.yaml`） |
| M01 | `target_builder/ingest.py` | 结构/序列标准化、author↔label↔UniProt 编号映射、QC（缺失/突变/非标准/重复编号）、SHA-256 输入清单 |
| M02 | `target_builder/cleave.py` | R87-T88 断链（新 C 端 COO- / 新 N 端 NH3+，含 OXT）、C73-C108 二硫键保留与审计、≥5 个无冲突代表构象、拓扑审计 |
| M03 | `epitope/patch.py` | T88 邻域 SASA/极性/曲率/构象变异、热点排序、膜平面与糖链排斥掩膜 |
| M04 | `generation/` | RFdiffusion 适配器（GPU）+ 确定性螺旋束骨架发生器（CPU 回退）+ FASTA/PDB 导入与校验 |
| M05 | `sequence_design/` | ProteinMPNN 适配器 + 启发式可溶设计回退、单体折叠过滤（fold_plddt 门槛） |
| M06 | `scoring/binding.py` | 裂解态多构象结合评估：界面面积/形状互补/氢键/埋藏未满足极性/冲突 + **T88 游离 α-氨基直接接触**（硬门槛）+ 跨构象复现率与稳健分位聚合 |
| M07 | `scoring/specificity.py` | 完整 TROP2（cis/trans）负状态：表位对齐姿势迁移 + 几何风险；EpCAM patch 交叉对接代理；Foldseek/MMseqs2 脱靶筛查接口；最危险负状态聚合 |
| M08 | `scoring/mechanism.py` | 7E5N/7E5M 装配叠合、cis 阻断/trans 遮挡量化、膜/糖链碰撞报告 |
| M09 | `scoring/developability.py` | MW/pI/净电荷/GRAVY、CamSol 式溶解度、A3D 式聚集热点、序列 liability（脱酰胺/氧化/异构化/蛋白酶/NGS/未配对Cys）、NetMHCIIpan 适配器 + 确定性代理（缺失→review） |
| M10 | `ranking/` + `reporting/` | 硬门槛（终局性）、非支配排序（NSGA-II 式）、PRD 12.2 稳健选择性公式、序列家族聚类多样性上限、加权展示分（仅层内）、CSV + HTML 报告 |

## 2. 安装

```bash
# conda 环境（Python 3.11，PRD 5.2 工程约定）
conda create -y -n trop2-platform python=3.11 pip
conda activate trop2-platform
pip install -e ".[dev]"

# 外部算法（复用之前下载的 checkout；不入库，超 GitHub 100MB 限制）
scripts/setup_external_tools.sh /home/protein_design2026/external
cp configs/tools.example.yaml configs/tools.yaml   # 按需修改路径

# 数据（仓库已含 7E5N/7E5M/7PEE/4MZV + FASTA；可重新获取）
scripts/fetch_data.sh
```

## 3. 运行

```bash
trop2 run                      # 端到端 M01→M10（失败可重跑，自动命中缓存恢复）
trop2 run --stages prepare     # 仅 M01-M03
trop2 run --stages generate    # 仅 M04
trop2 run --stages evaluate    # M05-M09
trop2 run --stages rank        # M10
trop2 run --run-id run_xxx     # 复用已有 run 目录（缓存跳过已完成阶段）
trop2 report outputs/<run_id>  # 改权重/阈值后仅重渲染报告（PRD 场景E）
trop2 status outputs/<run_id>  # 查看各阶段状态
```

输出位于 `outputs/<run_id>/`：`report.html`（医学研究者可读）、`candidate_metrics.csv`、`pareto_front.csv`、`rejection_reasons.csv`、`top_candidates/`、各模块指标表与结构文件、`run_manifest.json`、`task_status.csv`。

## 4. 关键设计决策

- **编号约定**：7E5N/7E5M/7PEE 的 author 编号与 UniProt P09758 一致（已验证 R87/T88/C73/C108 均为直接序列位置）；M01 仍构建 author↔label↔UniProt 三方映射表并强制校验裂解残基。
- **裂解拓扑**：断链后输出为双链（NFR/BODY）结构并显式保留 C73-C108 二硫键 —— 绝不作为两条互不约束的链交给折叠模型（PRD M02 医学注意事项）。
- **构象集**：以二硫键为转轴的确定性刚体采样（种子控制），O(n) 审计冲突体积/最小距离/二硫键完整性；PyRosetta FastRelax / OpenMM MD 作为可插拔升级路径。
- **代理指标诚实性**：无 GPU 环境下 ipTM/pAE/fold_plDDT 等置信度类指标由确定性几何代理给出并标记 `metric_source=proxy`；T88 末端接触、界面几何、负状态风险、机制与可开发性指标全部为真实坐标计算。报告中显著声明需 GPU 复算后方可用于实验决策。
- **硬门槛终局性**：任何加权高分不能救回被负状态硬门槛淘汰的候选（AC-12 有对应测试）。
- **可复现/可恢复**：所有随机性由 `resources.seed` 派生；缓存键 = 代码版本+阶段+种子+输入内容哈希；task_status.csv 追加式记录（AC-14/AC-15 有对应测试）。

## 5. 测试

```bash
pytest              # 单元 + 集成（固定 fixture，确定性）
```

覆盖 PRD 验收标准的关键项：AC-01/02/03（编号映射、裂解拓扑、构象审计）、AC-04（导入校验）、AC-07/08（T88 接触与裂解/完整态区分）、AC-11（工具缺失→review）、AC-12（硬门槛不可逆）、AC-13（Pareto+家族多样性）、AC-14（可复现）、AC-15（缓存恢复）。

## 6. 数据与许可

- 结构：RCSB PDB 7E5N（cis）、7E5M（trans）、7PEE（ECD）、4MZV（EpCAM ECD）；序列：UniProt P09758 / P16422。
- RFdiffusion / ProteinMPNN / Boltz / AF2 权重均限非商业研究使用；运行清单记录许可，商业化前须重新审查。
- R87-T88 裂解证据：Trerotola et al., Neoplasia 2021。

## 7. 后续版本（PRD 第 10 节）

完整糖基化膜体系 MD、BindCraft 全流程、真实 KD 预测、全蛋白组共折叠、Fc/PEG 化格式优化等均不在 V1 范围。
