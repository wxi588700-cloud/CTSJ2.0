# 模型说明（Model Card）与第三方模型清单

## 一、本项目模型性质声明（对应大赛第三节"特殊情形"）

**本项目未开展自训练/微调**，不存在自训模型权重。技术路线为
**确定性几何计算 + 规则化多目标决策**（非深度学习推理为主）：

- R87-T88 裂解态构建、表位分析、正/负状态界面评分、cis/trans 机制几何、
  可开发性指标 —— 全部为坐标级确定性计算（numpy/scipy/gemmi 实现，固定种子）。
- ipTM/pLDDT/pAE 类置信度指标在无 GPU 基线中由**确定性几何代理**给出，
  全部标注 `metric_source=proxy`（见报告与 `candidate_metrics.csv`）。
- 因此**无 train.py / 训练日志 / 数据划分**（大赛规范允许：未训练项目不强制提交）。

## 二、使用的开源/第三方模型（名称、版本、调用方式、许可）

| 模型/工具 | 版本 | 用途与调用方式 | 关键参数 | 许可 | 是否实际执行 |
|---|---|---|---|---|---|
| RFdiffusion | 本地 checkout（含 Complex_base 权重） | M04 binder 骨架生成（适配器 `src/trop2_design/generation/adapters.py`） | hotspot 残基=T88 邻域; 60-120 aa; seed 派生 | BSD-like 代码 / 权重非商业研究 | 本基线未执行（无 GPU），自动回退至确定性螺旋束生成器并记录于 generation_log.json |
| ProteinMPNN | 本地 checkout（vanilla 权重） | M05 序列设计（适配器 `sequence_design/design.py::ProteinMPNNAdapter`） | 每骨架 3 条序列; 禁止 Cys | MIT 代码 / 权重非商业研究 | 本基线未执行，回退至径向分层两亲性设计器并记录 |
| AlphaFold2-Multimer / ColabFold / Boltz-2 | AF2 权重未随附；Boltz-2 checkpoint 在服务器 `~/.boltz` | M06/M07 复合物复核预测（适配器接口 `configs/tools.yaml` predictors） | — | AF2 非商业；Boltz source-available | 未执行；代理指标显式标注，GPU 环境可插拔复算 |
| Foldseek / MMseqs2 | 未安装 | M07 脱靶筛查 | — | GPL-3 | 未执行，结果标注 review（不静默置零） |
| NetMHCIIpan 4.x | 未安装 | M09 MHC-II 呈递风险 | — | 学术许可 | 未执行，使用确定性倾向性代理并标注 |

外部算法本地部署方式：`scripts/setup_external_tools.sh <下载目录>`（复制到
`external/`，不入库；或直接编辑 `configs/tools.yaml` 指向已有路径）。

## 三、Model Card（本平台作为"模型"的说明）

- **适用范围**：人 TROP2（P09758）R87-T88 裂解态选择性小蛋白（60-120 aa）
  候选的端到端计算评估与排序；研究级决策支持。
- **输入**：`configs/trop2_v1.yaml`（靶点/裂解/设计/门槛配置）+ 公开结构/序列。
- **输出**：候选短名单（`results/results.csv`）、逐候选原始指标、淘汰原因、
  HTML 报告（`outputs/<run_id>/report.html`）。
- **已知局限**：
  1) 裂解态真实构象未知（构象集+最差状态评分仅是缓解）；
  2) 无 GPU 基线的置信度指标为几何代理，**进入实验决策前必须在 GPU 环境
     用 AF2-Multimer/Boltz 复算**；
  3) 计算分数不能替代 BLI/SPR 实测亲和力；不输出任何临床/成药结论；
  4) 免疫原性为风险排序，非"无免疫原性"证明。

## 四、本项目实际创新贡献（非"是否自训练"）

1. 化学拓扑级 R87-T88 裂解态构建与审计（断链/新末端/C73-C108 二硫键保留的
   可追溯拓扑编辑）；
2. 以 T88 游离 α-氨基直接识别为**独立硬门槛**的裂解态选择性设计框架；
3. 完整态 TROP2(cis/trans) + EpCAM 三重负状态负设计与"最危险负状态"聚合；
4. PRD 12.2 稳健选择性公式（保守分位正状态 - 最差负状态 - 不确定性惩罚）
   与硬门槛/Pareto/多样性聚类三层决策；
5. 全链路可复现工程（内容哈希缓存、断点恢复、逐指标溯源、固定种子）。
