# trop2_cis-dimer_inhibitor

**TROP2 R87-T88 裂解态特异性小蛋白（cis 二聚体抑制剂）从头设计与多目标计算评估平台**
—— 第一届全球大学生生命科学挑战赛 · 代码提交版本 v1.1.0

> 边界声明：研究级决策支持工具，不输出"已成药/已安全/具有临床疗效"结论；一切结合、
> 选择性、机制、免疫原性判断必须由实验（BLI/SPR 等）验证。

---

## 1. 项目简介与模块总览（PRD M00-M10）

野生型人 TROP2（UniProt P09758）在 **R87-T88** 发生裂解：R87 成为新 C 端（COO-）、
T88 成为新 N 端（NH3+），N 端片段经 **C73-C108 二硫键** 与主体连接。本平台围绕该
裂解态新表位设计 60-120 aa 小蛋白抑制剂，核心思想：

1. **正设计**：对 ≥5 个裂解态代表构象做结合评估，**T88 游离 α-氨基直接识别为独立硬门槛**；
2. **负设计**：完整 TROP2（cis/trans）与 EpCAM 三重负状态，按"最危险负状态"聚合风险；
3. **机制几何**：7E5N/7E5M 装配叠合量化 cis 阻断 / trans 遮挡 / 膜-糖链碰撞；
4. **三层决策**：硬门槛（终局性）→ Pareto 非支配排序 + PRD 12.2 稳健选择性公式 →
   序列家族多样性聚类短名单。

| 模块 | 目录 | 功能 |
|---|---|---|
| M00 | `src/trop2_design/workflow/` | 配置校验（Pydantic 严格模式）、DAG 编排、内容哈希缓存、断点恢复、运行审计（`run_manifest.json` / `task_status.csv` / `resolved_config.yaml`） |
| M01 | `target_builder/ingest.py` | 结构/序列标准化、author↔label↔UniProt 编号映射、QC（缺失/突变/非标准/重复编号）、SHA-256 输入清单 |
| M02 | `target_builder/cleave.py` | R87-T88 断链（新 C 端 COO- 含 OXT / 新 N 端 NH3+）、C73-C108 二硫键保留与审计、≥5 个无冲突代表构象（局部铰链采样+极性感知冲突判定）、拓扑审计 |
| M03 | `epitope/patch.py` | T88 邻域 SASA/极性/曲率/构象变异、热点排序、膜平面与糖链排斥掩膜 |
| M04 | `generation/` | RFdiffusion 适配器（GPU）+ 确定性螺旋束骨架发生器（CPU 回退）+ FASTA/PDB 导入与校验 |
| M05 | `sequence_design/` | ProteinMPNN 适配器 + 径向分层两亲性设计回退、单体折叠过滤（fold_plddt 门槛） |
| M06 | `scoring/binding.py` | 裂解态多构象结合评估：界面面积/形状互补/氢键/埋藏未满足极性/冲突 + T88 游离 α-氨基直接接触（硬门槛）+ 跨构象复现率与稳健分位聚合 |
| M07 | `scoring/specificity.py` | 完整 TROP2（cis/trans）负状态：表位对齐姿势迁移 + 几何风险；EpCAM patch 交叉对接代理；Foldseek/MMseqs2 脱靶筛查接口；最危险负状态聚合 |
| M08 | `scoring/mechanism.py` | 7E5N/7E5M 装配叠合、cis 阻断/trans 遮挡量化、膜/糖链碰撞报告 |
| M09 | `scoring/developability.py` | MW/pI/净电荷/GRAVY、CamSol 式溶解度、A3D 式聚集热点、序列 liability（脱酰胺/氧化/异构化/蛋白酶/NGS/未配对Cys）、NetMHCIIpan 适配器 + 确定性代理（缺失→review） |
| M10 | `ranking/` + `reporting/` | 硬门槛（终局性）、非支配排序（NSGA-II 式）、PRD 12.2 稳健选择性公式、序列家族聚类多样性上限、加权展示分（仅层内）、CSV + HTML 报告 |

架构图与数据契约见 `docs/architecture.md`；验收标准对照见 `docs/acceptance_mapping.md`。

## 2. 目录结构

```
trop2_cis-dimer_inhibitor/
├── README.md               # 本文件：环境、命令、输入输出、结果与溯源说明
├── requirements.txt        # Python 依赖及精确版本（与提交结果所用环境一致）
├── environment.yml         # conda 环境一键创建
├── run.sh                  # ★ 一键端到端主运行入口（M01→M10 + results.csv）
├── design.py               # 一键设计入口（prepare+generate+evaluate）
├── predict.py              # ★ 一键生成最终结果文件 results/results.csv
├── data/                   # 输入数据 + 来源/版本/许可说明（data/README.md）
├── src/trop2_design/       # 核心源代码（M00-M10，核心逻辑有注释）
├── models/                 # Model Card + 第三方模型清单（本项目无自训权重）
├── notebooks/              # demo_pipeline.ipynb（关键流程演示，已执行含输出）
├── results/                # ★ 标准候选清单 results.csv + 结构文件 + 示例报告
├── logs/                   # 示例运行审计：种子、阶段状态、拓扑审计、生成日志
├── tests/                  # 51 单元 + 9 集成测试（pytest）
├── configs/                # trop2_v1.yaml（项目配置）/ tools.yaml（算法路径）
├── scripts/                # 数据获取 / 外部算法部署 / 推送脚本
├── docs/                   # 架构说明、PRD 验收对照
└── outputs/<run_id>/       # 每次运行的完整产物（不入库，可复现）
```

## 3. 环境配置

**解释器与操作系统**：Python 3.11；Linux（提交结果环境：Ubuntu 22.04 x86_64）。
**CPU 即可完整运行**——无 GPU/CUDA/驱动/系统库依赖（GPU 仅用于可选的真实预测器
复算，见 §7）。内存 ≤ 8 GB；磁盘 < 1 GB（不含可选外部算法）。

```bash
# 方式 A：pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 方式 B：conda（一键）
conda env create -f environment.yml && conda activate trop2-cis-dimer-inhibitor
```

**可选外部算法**（RFdiffusion / ProteinMPNN；仅 GPU 主机需要。CPU 基线自动回退到
确定性生成器并如实记录于 `generation_log.json`）：

```bash
scripts/setup_external_tools.sh <本地已下载目录>   # 复制到 external/（不入库）
# 或直接编辑 configs/tools.yaml 指向已有 checkout 路径
```

依赖库精确版本清单见 `requirements.txt`。

## 4. 运行命令

| 命令 | 作用 | 预期耗时（CPU, 24 候选） |
|---|---|---|
| `bash run.sh` | **一键端到端**：读取 `configs/trop2_v1.yaml` → M01-M10 → `outputs/<run_id>/` 全部产物 + `results/results.csv` | 6-8 分钟 |
| `python predict.py` | 从最新 run 打包标准化候选清单（无已完成 run 则自动先跑全流程） | < 10 秒（打包） |
| `python design.py` | 仅设计阶段（M01-M09，不含排序/报告） | 同上减 M10 |
| `trop2 run --stages prepare` | 分阶段运行（prepare/generate/evaluate/rank） | 按阶段 |
| `trop2 run --run-id <id>` | 断点续跑（已完成阶段命中内容哈希缓存） | 跳过已缓存 |
| `pytest tests/` | 全部 110 个测试（101 unit + 9 integration） | 约 6 分钟（含集成） |

**输入**：`configs/trop2_v1.yaml`（靶点/裂解位点/设计参数/硬门槛/种子——医学参数
只改此文件，未知字段报错）+ `data/raw/` 公开结构与序列（重新获取：
`scripts/fetch_data.sh`，来源与许可见 `data/README.md`）。

**输出**（`outputs/<run_id>/`）：`report.html`（医学可读报告）· `candidate_metrics.csv`
（逐候选原始指标+归一化+Pareto+家族）· `pareto_front.csv` · `rejection_reasons.csv`
（可追溯淘汰原因）· `top_candidates/`（短名单序列+结构）· `cleaved_states/` +
`intact_states/`（构象集+拓扑审计）· `epitope_patch.json` / `hotspots.txt` /
`exclusion_mask.json` · 正/负状态指标表 · 机制/可开发性指标表 ·
`run_manifest.json`（版本/输入 SHA-256/种子/工具许可审计）· `task_status.csv`
（阶段状态与耗时）。

## 5. 最终结果文件（候选清单）

`predict.py` 生成 **`results/results.csv`**（UTF-8），字段：
候选编号 · 所属赛道 · 候选序列 · 长度 · 14 项关键预测指标（正状态复现率、T88 末端
识别、intact/EpCAM 风险、cis 阻断、trans 遮挡、折叠置信、聚集、溶解、不确定性、
稳健选择性等，方向速查见 `results/README.md`）· Pareto 层级 · 硬门槛状态 ·
**对应模型与运行版本** · 随机种子 · **结构文件名**（副本在 `results/structures/`，
mmCIF，坐标单位 Å，链命名 NFR=裂解 N 端片段 / BODY=主体 / BND=binder / T=完整
TROP2）· 备注（可追溯淘汰原因）。

提交示例（`results/`）：当前 results.csv 为 **16 个设计、全部 `review`（0 通过硬门槛）**
——诚实评估链下回退骨架候选的真实状态，配套勘误见 `results/README.md`；
正式结果包待全流程（真实 RFdiffusion + AF2 梯度精修 + Boltz 实测）重跑后重新生成。

## 6. 可复现性与溯源

- **随机种子**：全部随机性由 `resources.seed = 20260816` 派生，写入每次运行的
  `resolved_config.yaml` 与 `run_manifest.json`；相同输入+种子+版本 ⇒ 一致候选集合
  与离散排序（自动化测试 `test_identical_seed_identical_output` 覆盖）。
- **输入哈希**：所有输入文件 SHA-256 写入 `run_manifest.json` / `target_registry.json`。
- **断点恢复**：阶段级内容哈希缓存，失败重跑不覆盖已成功产物
  （`test_failure_does_not_clobber_previous_success` 覆盖）。
- **数据溯源**：`data/README.md` 完整披露来源/版本/获取时间（2026-08-16）/许可/
  用途与预处理记录；本项目**无模型训练**，不存在训练集划分与数据泄漏问题
  （`models/README.md` 第一节声明）；未使用任何隐藏评测集。
- **训练入口**：不适用（未开展训练/微调，按大赛规范不强制提交训练脚本）。

## 7. 模型说明（摘要，详见 `models/README.md`）

- **技术路线**：确定性几何计算 + 规则化多目标决策为主；**无自训权重**。
- **第三方模型**：RFdiffusion、ProteinMPNN、AF2-Multimer/ColabFold、Boltz-2、
  Foldseek/MMseqs2、NetMHCIIpan——名称/版本/调用方式/关键参数/许可/**是否实际执行**
  逐项列于 `models/README.md` 第二节；CPU 基线未执行者均有适配器接口并如实标注。
- **置信度指标诚实性**：ipTM/pLDDT/pAE 在 CPU 基线为**几何代理**（metric_source=
  proxy，报告显著声明）；T88 末端接触、界面几何、负状态风险、机制与可开发性指标
  均为真实坐标计算。GPU 复算仅需配置 `configs/tools.yaml`，代码路径不变。
- **本项目创新贡献**（五点，含裂解态拓扑编辑、T88 末端硬门槛、三重负设计、稳健
  选择性公式、可复现工程）见 `models/README.md` 第四节。

## 7.5 GPU 接入（Boltz-2 实测复算）

本集群 GPU 节点（`gn1`：8×48 GB，CUDA 12.4）与家目录 NFS 共享，`configs/tools.yaml`
已配置 Boltz-2 实测链路：

```yaml
predictors:
  boltz:
    python: ~/miniconda3/envs/boltz/bin/python   # boltz env (2.0.3, torch 2.5.1+cu121)
    notes: ssh_host=gn1                          # CPU 管理节点发起时自动 SSH 派发
```

- **两阶段算力控制**（PRD 8.2）：几何代理全量筛选 → Boltz-2 仅对 `resources.
  boltz_recompute_top_k`（默认 8）个设计做**独立复算**（预测复合物与设计姿势无关，
  消除自证偏差），实测 ipTM/pLDDT/界面 PAE/T88 接触替换代理值，`metric_source=
  measured`、`predictor=boltz-2` 全程标注。
- **M05 单体折叠**同样走 Boltz 实测（pLDDT + 预测结构 vs 设计骨架的 bound/unbound
  RMSD，Kabsch 对齐）。
- 自动挑选空闲显存最大的 GPU（`nvidia-smi` 解析）；**指定固定卡**（如 6 号）：
  `configs/tools.yaml` → `predictors.boltz.device: 6`（持久）｜
  `TROP2_BOLTZ_DEVICE=6 bash run.sh`（临时，SSH 派发同样生效）｜手动单跑
  `ssh gn1` 后 `CUDA_VISIBLE_DEVICES=6 boltz predict ...`。解析优先级：
  device 字段 > notes `device=6` > 环境变量 > 自动选卡。离线运行（自序列
  MSA，无需 MSA 服务器）；固定种子。NFS 属性缓存延迟已用 readdir+重试处理。
- 实测速率参考：~70 aa 单体 ≈ 90 s；~300 残基三链复合物 ≈ 110 s（单卡）。
- **实测校准的重要结论**：Boltz 独立验证显示 fallback 启发式设计的真实结合很弱
  （ipTM 0.10–0.15 vs 几何代理 0.6–0.8）——这正是接入 GPU 复算的价值：进入实验
  决策的候选必须以 `metric_source=measured` 为准；正式生产应配合 RFdiffusion +
  ProteinMPNN 生成真实候选后再实测排序。
- 直接在 GPU 节点上运行（本地执行、免 SSH）：`ssh gn1` 后同样 `bash run.sh`
  （去掉 tools.yaml notes 中的 `ssh_host=` 即本地模式）。

## 7.6 裂解糖基化靶结构 target bundle（PRD v1.1）

```bash
trop2 prepare-target --project configs/trop2_v1_1.yaml   # 构建版本化 bundle（GPU, ~4min）
trop2 validate-target outputs/<run>/target_bundles       # 拓扑/文件审计（AC-30）
trop2 export-target-bundle outputs/<run>/target_bundles [--out-dir DIR]
```

- **混合策略**（实证驱动，替代 PRD 的 Chai-1/GlycanTreeModeler）：Boltz-2
  蛋白构象（六对天然二硫键 bond 约束，实测 1.4–1.9 Å 生效）× 确定性原子模板
  接枝（三糖型面板，全部 N-糖苷键精确 1.43 Å，软安装可审计）
- 产物：`target_bundles/<id>/`（manifest / glycosylated_states / protein_only_views /
  glycan_masks / topology / provenance），bundle_id 不可变（模板哈希+糖型+seed）
- 下游：M03 自动改用 bundle 的**真实糖链坐标球**（560 球替代 4 个启发式 12 Å 球）；
  M10 结果表携带 target_bundle_id（AC-26）
- 兼容：`configs/trop2_v1.yaml`（无 target_prediction 段）原样走 v1.0 legacy
  路径（AC-27 有回归测试）
- 边界：糖型为 assumed_sensitivity_panel（无位点特异性糖组学数据时），
  所有结构均为计算假设（computed_hypothesis），非实验结构

## 8. 测试

```bash
pytest tests/unit -q          # 51 个单元测试（< 10 秒）
pytest tests/ -q              # + 9 个端到端集成测试（约 6 分钟）
```

覆盖大赛复核关心的核心逻辑：裂解拓扑审计（AC-02/03）、导入校验（AC-04）、T88 末端
识别（AC-07/08）、负状态输出（AC-09）、硬门槛不可逆性（AC-12）、Pareto+多样性
（AC-13）、可复现（AC-14）、断点恢复（AC-15）——完整对照表
`docs/acceptance_mapping.md`。

## 9. 许可与合规

输入数据：RCSB PDB（CC0 1.0）与 UniProt（CC BY 4.0）。第三方算法许可逐项记录于
`models/README.md` 与每次运行的 `run_manifest.json`（RFdiffusion/ProteinMPNN 权重、
AF2 权重均为非商业研究许可，商业化前需重新审查）。

## 10. 故障排查

运行问题先看 `outputs/<run_id>/task_status.csv`（哪个阶段失败、失败原因）与
`run_manifest.json` 的 `failures` 字段；重跑 `bash run.sh <run_id>` 从断点续跑。
