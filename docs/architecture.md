# 架构说明（PRD 第 4 节实现）

```
                 ┌────────────────────────────────────────────────┐
                 │  configs/trop2_v1.yaml (医学配置, 严格校验)      │
                 │  configs/tools.yaml  (算法路径/许可/版本)        │
                 └───────────────┬────────────────────────────────┘
                                 │
                        M00 编排/缓存/审计 (workflow/)
                                 │
        ┌───────────┬────────────┼──────────────┬─────────────┐
        ▼           ▼            ▼              ▼             ▼
      M01 ingest  M02 cleave   M03 patch   M04 generate   (M05 design)
      结构标准化   R87-T88断链  T88表位/热点  RFdiffusion/   ProteinMPNN/
      编号映射QC   +构象集+审计  膜/糖排斥     导入+校验      启发式回退+折叠过滤
        └───────────┴────────────┼──────────────┴─────────────┘
                                 ▼
                      M06 正状态结合评分
                      (界面几何全量指标 + T88末端接触[硬门槛] + 跨构象复现率)
                                 │ 仅 pass_rate>0 候选进入（PRD 8.2 算力控制）
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
        M07 负状态选择性       M08 cis/trans机制      M09 可开发性
        完整TROP2(cis/trans)   装配叠合/覆盖/遮挡     MW/pI/溶解/聚集/
        EpCAM patch交叉对接    膜糖碰撞报告          liability/MHC-II
            └────────────────────┼────────────────────┘
                                 ▼
                      M10 硬门槛 → Pareto(NSGA-II式) → 稳健选择性
                          → 多样性聚类 → CSV + HTML报告 → top_candidates/
```

## 数据契约

- 模块间只通过 `outputs/<run_id>/` 下的标准文件交互（PRD 第 7 节），每个阶段的标准输出见模块 docstring。
- `candidate_id` 全流程稳定（内容哈希），不随排序重编号。
- 所有阈值在 `ranking profile` 中版本化（`v1_strict`）；变更即产生新 profile id。

## 可复现性

- 随机性全部由 `resources.seed` 派生（numpy `default_rng`）。
- 缓存键 = SHA-256(代码版本 + 阶段名 + 种子 + 输入文件内容哈希 + 输出列表)。
- `run_manifest.json` 记录 git commit、python/包版本、工具版本与许可、输入哈希、种子、平台。
- `task_status.csv` 追加式记录每个阶段的 ok/cached/skipped/failed。
