# 运行日志与随机种子记录

## 随机种子

全部随机性由 `configs/trop2_v1.yaml` 的 `resources.seed` 派生（当前提交值
**20260816**）。相同输入 + 相同种子 + 相同代码版本 ⇒ 相同候选集合与一致的
离散排序结果（AC-14 有自动化测试覆盖）。变更种子即产生新的运行谱系，种子
写入每次运行的 `outputs/<run_id>/resolved_config.yaml` 与 `run_manifest.json`。

## 本目录内容

`logs/` 存放与提交结果配套的**示例运行审计文件**（复制自
`outputs/run_20260816_043721`，即 `results/results.csv` 的数据来源）：

| 文件 | 说明 |
|---|---|
| `example_run_manifest.json` | 运行清单：git commit、python/包版本、工具版本与许可、输入 SHA-256、种子、平台、各阶段状态 |
| `example_task_status.csv` | 10 个阶段（M01-M10）的状态/缓存键/起止时间/耗时 |
| `example_topology_audit.json` | 裂解态拓扑审计（每构象的断链/末端/二硫键/冲突记录） |
| `example_generation_log.json` | 候选生成日志（RFdiffusion 适配器可用性、回退生成记录） |

> 训练日志不适用：本项目无自训练模型（见 `models/README.md` 第一节声明）。

完整交互日志（每阶段的 append-only 记录）在任意运行后生成于
`outputs/<run_id>/task_status.csv` 与 `outputs/<run_id>/run_manifest.json`。
