# 数据说明与获取方式（大赛《代码提交要求》第三节）

本项目**不包含任何受限/私有数据**；全部输入为公开数据库资源，可一键重新获取
（`scripts/fetch_data.sh`）。

## 数据清单

| 文件 | 来源 | 获取时间 | 版本/编号 | 许可 | 用途 |
|---|---|---|---|---|---|
| `data/raw/pdb/7E5N.cif` | RCSB PDB | 2026-08-16 | 7E5N（cis 组装晶体结构） | CC0 1.0 | M01/M02/M08：cis 裂解态构建与装配叠合 |
| `data/raw/pdb/7E5M.cif` | RCSB PDB | 2026-08-16 | 7E5M（trans 组装晶体结构） | CC0 1.0 | M01/M08：trans 装配叠合（trans 遮挡评估） |
| `data/raw/pdb/7PEE.cif` | RCSB PDB | 2026-08-16 | 7PEE（TROP2 胞外结构域） | CC0 1.0 | M01：备用模板/对照 |
| `data/raw/pdb/4MZV.cif` | RCSB PDB | 2026-08-16 | 4MZV（人 EpCAM 胞外结构域） | CC0 1.0 | M07：EpCAM 负状态脱靶评估 |
| `data/raw/fasta/TROP2_human.fasta` | UniProt REST | 2026-08-16 | P09758 (TACSTD2, ISO=1) | CC BY 4.0 | M01：参考序列/编号映射校验 |
| `data/raw/fasta/EpCAM_human.fasta` | UniProt REST | 2026-08-16 | P16422 (EPCAM, ISO=1) | CC BY 4.0 | M07：EpCAM 序列对照 |

重新获取命令（网络可达时）：

```bash
scripts/fetch_data.sh
```

## 数据清洗与预处理记录

- M01 对每个结构执行：蛋白链抽取（去配体/水/糖链）、SEQRES↔UniProt 全局比对、
  缺失残基/突变/非标准残基/重复编号 QC（结果写入 `outputs/<run>/input_qc.json`）。
- 输入文件 SHA-256 哈希写入 `target_registry.json` 与 `run_manifest.json`，保证溯源。
- 未使用任何隐藏评测集；本项目亦不涉及模型训练，无训练/验证/测试划分与数据泄漏风险
  （详见 `models/README.md` 的「无训练数据」声明）。

## 参考文献

- 7E5N/7E5M：TROP2 cis/trans 组装结构（RCSB PDB 条目页引用文献）。
- R87-T88 裂解证据：Trerotola et al., *Neoplasia* 2021（PMC8042651）。
