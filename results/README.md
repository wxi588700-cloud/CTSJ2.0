# 运行结果与候选清单示例

本目录包含一次**完整端到端运行**（`outputs/run_20260816_043721`，种子 20260816，
CPU 确定性模式）产出的标准化候选清单，作为大赛复核示例。

## 文件说明

| 文件 | 说明 |
|---|---|
| `results.csv` | **标准化候选清单**（UTF-8，`predict.py` 生成）：候选编号 / 所属赛道 / 序列 / 14 项关键预测指标 / 模型与运行版本 / 随机种子 / 结构文件名 / 备注（含淘汰原因）。（历史声明 72 设计/6 通过——已失效；当前文件 16 设计全部 review、0 通过） |
| `structures/` | 清单对应的三维结构文件副本（正状态复合物 + cis/trans 装配叠合，mmCIF，链命名：NFR=裂解 N 端片段 / BODY=主体 / BND=binder / T=完整 TROP2；坐标单位 Å） |
| `example_candidate_metrics.csv` | M10 完整指标表（原始值 + 归一化值 + Pareto 层级 + 家族聚类） |
| `example_pareto_front.csv` | 硬门槛幸存者的非支配前沿 |
| `example_report.html` | 面向医学研究者的完整 HTML 报告（短名单、淘汰原因、阈值、建议实验对照） |

## 关键指标方向速查

| 指标 | 方向 | 含义 |
|---|---|---|
| robust_selectivity | 最大化 | 稳健选择性（保守分位正状态 − 最差负状态 − 不确定性惩罚，PRD 12.2） |
| positive_state_pass_rate | 最大化 | 裂解态构象结合复现率 |
| t88_terminal_contact | 硬门槛 | T88 游离 α-氨基直接识别 |
| intact_trop2_risk / epcam_risk | 最小化 | 完整 TROP2 / EpCAM 结合风险 |
| cis_block_score | 最大化 | cis 界面阻断几何 |
| trans_occlusion_score | 最小化 | trans 界面遮挡 |
| fold_plddt | 最大化 | 单体折叠置信度（本示例为几何代理） |
| aggregation_risk / solubility_score | 最小化 / 最大化 | 聚集 / 溶解 |
| uncertainty | 最小化 | 跨构象不一致性 |

> ⚠️ 本示例在无 GPU 服务器生成：置信度类指标为确定性几何代理
> （`model_and_version` 列已标注）。复核时在任意机器重跑 `bash run.sh`
> 可复现同一清单（相同种子）；在 GPU 机器配置 `configs/tools.yaml` 后
> 同一流水线将改用 AF2-Multimer/Boltz 实测置信度。

## 复现命令

```bash
bash run.sh                 # 全流程（约 6-8 分钟, CPU）
python predict.py           # 仅重新打包最新 run 为 results.csv
```

> **可复现性勘误（2026-08-20，audit-fix-v2）**：v2.0.2 之前的历史 run（含本目录
> example 文件与 results.csv）由进程随机化的 `hash()` 派生种子生成——**相同种子
> 重跑不可复现同一清单**。v2.0.2 起随机性全部经 SHA-256 稳定派生（io.stable_hash）
> 并在 run.sh 固定 PYTHONHASHSEED=0，声明方可成立。引用历史数据时请以
> run_manifest 的内容哈希为准，勿假设可重生成。

> **外审勘误（2026-08-21）**：历史"72 设计/6 通过"声明来自代理指标虚高时期，已被诚实评估链纠正。正式结果包待全流程重跑后重新生成。
