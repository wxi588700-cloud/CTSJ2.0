"""HTML report rendering (M10, PRD AC-17).

Self-contained Jinja2 template: top candidates, per-candidate metric cards
with raw values / thresholds / directions / provenance, rejection reasons,
uncertainty and suggested experimental controls (WT, cleaved, R87A/T88A,
EpCAM - PRD milestone P5).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FunctionLoader, select_autoescape

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>TROP2 cleaved-state binder design report - {{ run_id }}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
         margin: 0; background: #f6f7f9; color: #1c2733; }
  header { background: #0d3b66; color: white; padding: 24px 32px; }
  header h1 { margin: 0 0 6px 0; font-size: 22px; }
  header .meta { opacity: .85; font-size: 13px; }
  main { padding: 24px 32px; max-width: 1200px; margin: auto; }
  section { background: white; border-radius: 10px; padding: 20px 24px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(16,36,64,.08); }
  h2 { margin-top: 0; font-size: 17px; color: #0d3b66; border-bottom: 2px solid #e8ecf1; padding-bottom: 8px;}
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th { background: #eef2f7; text-align: left; padding: 6px 8px; border-bottom: 2px solid #d5dce4; }
  td { padding: 6px 8px; border-bottom: 1px solid #edf0f3; }
  tr.pass td { background: #f0f9f1; }
  tr.reject td { background: #fdf0f0; }
  tr.review td { background: #fff8e6; }
  .badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .badge.pass { background:#d4f1d9; color:#1c6b2d; }
  .badge.reject { background:#fadbd8; color:#9c1f14; }
  .badge.review { background:#fdf3d0; color:#8a6d00; }
  .note { color:#5b6b7a; font-size:12.5px; }
  .warn { color:#9c4a00; font-size:12.5px; }
  code { background:#f0f3f6; padding:1px 5px; border-radius:4px; font-size:12px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(340px,1fr)); gap:14px; }
  .card { border:1px solid #e3e9ef; border-radius:8px; padding:12px 14px; }
  .card h3 { margin:0 0 8px 0; font-size:14px; color:#0d3b66; }
  .kv { font-size:12px; display:flex; justify-content:space-between; padding:2px 0; }
  .kv b { font-weight:600; }
  footer { padding: 18px 32px 40px; color:#7b8794; font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>TROP2 R87-T88 裂解态小蛋白 binder 设计报告</h1>
  <div class="meta">run {{ run_id }} · seed {{ seed }} · config {{ config_hash }}
  · generated {{ generated }}</div>
</header>
<main>

<section>
  <h2>1. 运行摘要</h2>
  <table>
    <tr><th>候选总数</th><td>{{ df|length }}</td></tr>
    <tr><th>硬门槛通过 (pass)</th><td>{{ n_pass }}</td></tr>
    <tr><th>硬门槛淘汰 (reject)</th><td>{{ n_reject }}</td></tr>
    <tr><th>需人工复核 (review)</th><td>{{ n_review }}</td></tr>
    <tr><th>结构家族数（序列一致性 {{ identity }}）</th><td>{{ n_families }}</td></tr>
    <tr><th>推荐实验短名单</th><td>{{ n_top }} 个（单家族上限 {{ max_family }}）</td></tr>
  </table>
  <p class="note">硬门槛来自 profile <code>{{ gate_profile }}</code>；加权分仅在同一 Pareto
  层内辅助排序，任何高亲和预测都不能抵消完整 TROP2 / EpCAM 交叉结合（PRD 12.3）。</p>
</section>

<section>
  <h2>2. 推荐实验候选短名单</h2>
  {% if not top %}
  <p class="warn">没有候选同时通过全部硬门槛。请查看第 4 节淘汰原因与第 5 节建议。</p>
  {% else %}
  <table>
    <tr><th>排名</th><th>Candidate</th><th>设计序列</th><th>长度</th>
        <th>Pareto层</th><th>robust_selectivity</th><th>T88接触占有率</th>
        <th>intact风险</th><th>EpCAM风险</th><th>family</th></tr>
    {% for r in top_rows %}
    <tr class="pass">
      <td>{{ loop.index }}</td>
      <td><code>{{ r.candidate_id }}/{{ r.design_name }}</code></td>
      <td style="font-family:monospace;font-size:11px;max-width:260px;word-break:break-all;">{{ r.sequence[:44] }}{% if r.sequence|length > 44 %}…{% endif %}</td>
      <td>{{ r.sequence|length }}</td>
      <td>{{ r.pareto_rank }}</td>
      <td>{{ '%.3f'|format(r.robust_selectivity if r.robust_selectivity is not none else 0) }}</td>
      <td>{{ '%.2f'|format(r.t88_contact_occupancy if r.t88_contact_occupancy is not none else 0) }}</td>
      <td>{{ '%.2f'|format(r.intact_risk if r.intact_risk is not none else 0) }}</td>
      <td>{{ '%.2f'|format(r.epcam_risk if r.epcam_risk is not none else 0) }}</td>
      <td>{{ r.family_cluster }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}
</section>

<section>
  <h2>3. 候选指标总表（原始值 + 状态）</h2>
  <table>
    <tr><th>Candidate</th><th>状态</th><th>pass率</th><th>T88接触</th><th>iptm(robust)</th>
        <th>intact</th><th>epcam</th><th>cis阻断</th><th>trans遮挡</th>
        <th>膜/糖碰撞</th><th>聚集</th><th>pLDDT</th><th>不确定性</th></tr>
    {% for r in all_rows %}
    <tr class="{{ r.hard_filter_status }}">
      <td><code>{{ r.candidate_id }}</code></td>
      <td><span class="badge {{ r.hard_filter_status }}">{{ r.hard_filter_status }}</span></td>
      <td>{{ '%.2f'|format(r.positive_state_pass_rate if r.positive_state_pass_rate is not none else -1) }}</td>
      <td>{{ '✓' if r.t88_contact else '✗' }}</td>
      <td>{{ '%.2f'|format(r.complex_iptm if r.complex_iptm is not none else -1) }}</td>
      <td>{{ '%.2f'|format(r.intact_risk if r.intact_risk is not none else -1) }}</td>
      <td>{{ '%.2f'|format(r.epcam_risk if r.epcam_risk is not none else -1) }}</td>
      <td>{{ '%.2f'|format(r.cis_block if r.cis_block is not none else -1) }}</td>
      <td>{{ '%.2f'|format(r.trans_occlusion if r.trans_occlusion is not none else -1) }}</td>
      <td>{{ r.glycan_membrane_clash if r.glycan_membrane_clash is not none else '-' }}</td>
      <td>{{ '%.2f'|format(r.aggregation_risk if r.aggregation_risk is not none else -1) }}</td>
      <td>{{ '%.0f'|format(r.fold_plddt if r.fold_plddt is not none else 0) }}</td>
      <td>{{ '%.3f'|format(r.uncertainty if r.uncertainty is not none else 0) }}</td>
    </tr>
    {% endfor %}
  </table>
  <p class="warn">指标来源标注：本基线运行在无 GPU 环境使用确定性几何代理（metric_source=proxy），
  结果仅用于流水线验证与相对排序；进入实验决策前必须在 GPU 环境用 AF2-Multimer/Boltz 复算。</p>
</section>

<section>
  <h2>4. 淘汰原因（可追溯）</h2>
  {% if not rejections %}<p class="note">无淘汰记录。</p>{% endif %}
  <div class="grid">
  {% for item in rejections %}
    <div class="card">
      <h3><code>{{ item.candidate_id }}/{{ item.design_name }}</code></h3>
      {% for reason in item.reasons %}<div class="kv"><span>{{ reason }}</span></div>{% endfor %}
    </div>
  {% endfor %}
  </div>
</section>

<section>
  <h2>5. 硬门槛与阈值（版本 {{ gate_profile }}）</h2>
  <table>
    <tr><th>指标</th><th>规则</th><th>阈值</th><th>说明</th></tr>
    {% for g in gates %}
    <tr><td><code>{{ g.metric }}</code></td><td>{{ g.op }}</td><td>{{ g.threshold }}</td>
        <td>{{ g.reject_message }}</td></tr>
    {% endfor %}
  </table>
</section>

<section>
  <h2>6. 建议实验验证方案（PRD P5 交接）</h2>
  <ul>
    <li>对短名单候选进行表达纯化，并行 BLI/SPR 测定对<b>裂解态 TROP2</b>的亲和力。</li>
    <li>必须包含的对照：<b>完整 TROP2（WT）</b>、<b>R87A/T88A 裂解受阻突变体</b>、<b>EpCAM ECD</b>，
        以验证裂解态选择性（负状态硬门槛对应实验）。</li>
    <li>细胞实验：裂解态高 vs 低的肿瘤细胞系染色，评估 cis 组装下降与 trans 组装兼容性。</li>
    <li>可开发性跟进：聚集（SEC-MALS）、热稳定性（DSF）、血清稳定性、MHC-II 验证性检测。</li>
  </ul>
</section>

<section>
  <h2>7. 局限与解释边界</h2>
  <ul>
    <li>裂解态真实构象未知：构象集采样 + 最差状态评分只是缓解（PRD 风险表）。</li>
    <li>计算分数不能替代真实 KD/kon/koff；结论必须经 BLI/SPR 验证。</li>
    <li>免疫原性为风险排序，不得解读为"无免疫原性"；不输出人体半衰期结论。</li>
    <li>本报告为研究级决策支持，不构成任何临床疗效或安全性结论。</li>
  </ul>
</section>

</main>
<footer>trop2_cis-dimer_inhibitor v1.0 · 生成于 {{ generated }} · 所有阈值可在 ranking profile 中版本化调整</footer>
</body>
</html>
"""


def render_report(out: Path, df: pd.DataFrame, top: pd.DataFrame,
                  gate_profile, metrics_profile, ctx) -> str:
    env = Environment(autoescape=select_autoescape(["html"]), loader=FunctionLoader(
        lambda _: TEMPLATE))
    tpl = env.from_string(TEMPLATE)

    def jsafe(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    all_rows = []
    for _, r in df.iterrows():
        d = {k: jsafe(v) for k, v in r.to_dict().items()}
        all_rows.append(d)
    top_rows = []
    for _, r in top.iterrows():
        top_rows.append({k: jsafe(v) for k, v in r.to_dict().items()})
    rejections = [
        {"candidate_id": r.candidate_id, "design_name": r.design_name,
         "reasons": [x for x in str(r.rejection_reasons).split(";") if x]}
        for _, r in df[df.hard_filter_status == "reject"].iterrows()
    ]
    import datetime

    manifest = {}
    mf = out / "run_manifest.json"
    if mf.exists():
        import json

        manifest = json.loads(mf.read_text())

    html = tpl.render(
        run_id=out.name,
        seed=ctx.seed,
        config_hash=manifest.get("config_hash", ""),
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        df=all_rows, all_rows=all_rows, top_rows=top_rows, top=top_rows,
        rejections=rejections,
        n_pass=int((df.hard_filter_status == "pass").sum()),
        n_reject=int((df.hard_filter_status == "reject").sum()),
        n_review=int((df.hard_filter_status == "review").sum()),
        n_families=int(df.family_cluster.nunique()) if "family_cluster" in df else 0,
        n_top=len(top_rows),
        identity=ctx.config.ranking.diversity_cluster_identity,
        max_family=ctx.config.ranking.max_per_family,
        gate_profile=gate_profile.profile_id,
        gates=[g.model_dump() for g in gate_profile.gates],
    )
    return html
