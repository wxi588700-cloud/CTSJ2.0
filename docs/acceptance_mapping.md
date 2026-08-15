# PRD 验收标准对照表（v1.0 → 实现）

| AC | Given/When/Then | 实现位置 | 测试 | 状态 |
|---|---|---|---|---|
| AC-01 | 7E5N/7E5M/7PEE + FASTA → 统一编号、R87/T88/C73/C108 映射、SHA256 清单 | `target_builder/ingest.py` | `test_ac01_mapping_and_hashes` | ✅ |
| AC-02 | 完整 TROP2 → R87-T88 肽键不存在、新末端正确、C73-C108 二硫键存在 | `target_builder/cleave.py` | `test_ac02_ac03_topology_audit`、`test_topology.py` | ✅ |
| AC-03 | 构象集结构审计 ≥5 个通过 | `cleave.py`（SG 锚定采样+冲突过滤，不足5个则显式报错） | 同上 | ✅ |
| AC-04 | 合法/非法 binder FASTA 导入 | `generation/generate.py::import_fasta_candidates` | `test_schemas_import.py` | ✅ |
| AC-05 | RFdiffusion 适配器烟雾测试 | `generation/adapters.py`（不可用时确定性回退并记录 generation_log） | `test_ac05_candidates_generated` | ✅* |
| AC-06 | 单体预测输出结构/pLDDT/RMSD、失败有错误码 | `sequence_design/design.py::MonomerPredictor` + `monomer_metrics.csv.status` | `test_all_standard_outputs_exist` | ✅* |
| AC-07 | 阳性测试候选 M06 输出全部正状态指标并识别预置 T88 接触 | `scoring/binding.py::t88_terminal_evidence` | `test_scoring_logic.py` | ✅ |
| AC-08 | T88 恢复肽键后重新评分须区分裂解态/完整态 | `terminal_contact.json` 仅含 kind=cleaved 记录；intact 对照态单独保存在 intact_states/ | `test_topology.py`（intact 保留肽键审计） | ✅ |
| AC-09 | 完整 TROP2 + EpCAM 负设计分别输出风险并进入 candidate_metrics | `scoring/specificity.py` | `test_ac09_negative_states` | ✅ |
| AC-10 | cis/trans 叠合输出覆盖/遮挡/碰撞及结构文件 | `scoring/mechanism.py` + `assembly_overlays/` | `test_ac10_mechanism_outputs` | ✅ |
| AC-11 | 可开发性输出 MW/pI/聚集/免疫/序列风险；工具缺失→review | `scoring/developability.py` | `test_ac11_developability_outputs` | ✅ |
| AC-12 | 人为 EpCAM 高风险候选触发硬门槛且不可被加权分救回 | `ranking/pareto.py::apply_gates`（终局性） | `test_ac12_gate_irreversible` | ✅ |
| AC-13 | 20 候选 Pareto 排序输出层级/标准化值/权重版本/≥2 结构家族 | `ranking/rank.py` + `pareto_front.csv` | `test_ac13_pareto_and_diversity` | ✅ |
| AC-14 | 相同配置+种子重复运行结果一致 | 全链路 `default_rng(seed)` 派生 | `test_identical_seed_identical_output` | ✅ |
| AC-15 | 中途终止重启：缓存命中、不覆盖成功结果 | `workflow/engine.py` | `test_failure_does_not_clobber_previous_success` | ✅ |
| AC-16 | RTX 4090 基线 10 候选 24h 内完成 | 架构支持（分阶段+代理模式 CPU 即可完成；GPU 复算约数小时） | 需 GPU 环境 | ⏳ |
| AC-17 | 医学研究者可读 HTML 报告（前24/淘汰原因/阈值/对照建议） | `reporting/html.py` | `test_ac17_html_report_readable` | ✅ |
| AC-18 | 自动测试通过、核心覆盖率 ≥70% | `tests/` 51 单元 + 9 集成 | `pytest` | ✅ |

\* 无 GPU 环境下以确定性回退/代理模式验证流水线完整性；真实 RFdiffusion/AF2 运行需在 GPU 主机以同一适配器执行（tools.yaml 配置路径即可）。

## PRD 第 9 节 V1 必须完成功能对照

- 可安装 Python 项目 + CLI + 示例配置 + README：`pyproject.toml` / `trop2` CLI / `configs/`
- R87-T88 断链、新末端、C73-C108 二硫键拓扑审计：M02 ✅
- ≥5 裂解态构象 + 完整态对照：M02 ✅
- RFdiffusion 生成 + FASTA/PDB 导入双入口：M04 ✅（适配器+回退）
- ProteinMPNN 序列设计 + 单体折叠过滤：M05 ✅（适配器+回退）
- 裂解态结合 + T88 新 N 端直接接触：M06 ✅
- 完整 TROP2 与 EpCAM 负状态选择性：M07 ✅
- cis 阻断 / trans 遮挡 / 膜糖碰撞几何：M08 ✅
- 可溶性 / 聚集 / 免疫原性 / 序列降解风险：M09 ✅
- 硬门槛 / Pareto / 不确定性惩罚 / 多样性聚类：M10 ✅
- CSV / 结构 / 淘汰原因 / HTML 报告：M10 ✅
- 端到端可恢复 + 版本/配置/输入哈希/种子记录：M00 ✅
