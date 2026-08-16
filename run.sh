#!/usr/bin/env bash
# ============================================================================
# trop2_cis-dimer_inhibitor - 一键端到端主运行入口 (大赛提交主入口)
#
# 读取: configs/trop2_v1.yaml (+ configs/tools.yaml 算法路径, 可选)
# 产出: outputs/<run_id>/ 完整结果 (含 report.html, candidate_metrics.csv)
#       以及标准化候选清单 results/results.csv (由 predict.py 生成)
#
# 用法:
#   bash run.sh                 # 全新运行 (M01 -> M10)
#   bash run.sh <run_id>        # 复用已有 run 目录 (缓存续跑 / 断点恢复)
#
# 环境要求: Python 3.11 (requirements.txt / environment.yml)
# 预期耗时: CPU 模式 24 候选约 6-8 分钟 (内存 < 8 GB, 无 GPU 依赖)
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

RUN_ID="${1:-}"

# 定位安装了本项目的 Python 解释器
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python python3 \
      "${HOME}/miniconda3/envs/trop2-cis-dimer-inhibitor/bin/python" \
      "${HOME}/miniconda3/envs/trop2-platform/bin/python"; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c "import trop2_design" >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "[run.sh] ERROR: 未找到可导入 trop2_design 的 Python," \
       "请先: pip install -e . (见 README)" >&2
  exit 1
fi
echo "[run.sh] interpreter: ${PY}"

if [ -n "$RUN_ID" ]; then
  "${PY}" -m trop2_design.cli run --run-id "${RUN_ID}"
else
  "${PY}" -m trop2_design.cli run
fi

# 从最新 run 生成标准化候选清单 results/results.csv
"${PY}" predict.py

LATEST="$(ls -dt outputs/run_* 2>/dev/null | head -1)"
echo "[run.sh] 完成。"
echo "[run.sh] 候选清单: results/results.csv"
echo "[run.sh] 完整报告: ${LATEST}/report.html"
