"""AF2 梯度 binder 设计 inner 脚本（cis 项目移植版）。

移植自 trop2-binder/design/_gradient_inner.py（生产验证：Boltz ipTM 0.716），
适配本项目裂解态双链靶标（BODY=NFR 两条链 + binder）：

  - prep_inputs(chain="A,B")：靶标双链（已实测 PLDDT 形状 317=181+56+80）
  - hotspot 使用链前缀格式 "A88,A96"（colabdesign pdb-indexed 形式）
  - 糖链避让按 binder 工程 v2.1 经验不做 JAX callback（tracing 限制），
    由外层阶段做后置距离过滤
  - 输出 schema 与 binder 版一致：design_results.json（traj/status/
    sequence/binder_plddt/i_ptm/i_pae/pdb/time_s）

运行环境：design conda env（jax 0.6.0 + colabdesign 1.1.3），GPU 上执行，
建议 env：CUDA_VISIBLE_DEVICES=<卡号> XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
         XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 PYTHONHASHSEED=0
"""
import argparse, json, time
from pathlib import Path

import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem


def design_binder(target_pdb, chain_str, hotspot_str, binder_len,
                  n_trajectories, seed, out_dir,
                  i_pae_weight=0.5, num_design_models=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 靶标残基数 = 指定链的并集（binder 追加在其后，plddt 切片用 -binder_len）
    seen = set()
    chain_ids = [c for c in chain_str.split(",") if c.strip()]
    for line in open(target_pdb):
        if line.startswith("ATOM") and line[21] in chain_ids:
            seen.add(int(line[22:26]))
    target_len = len(seen)
    print(f"  target_len={target_len} chains={chain_ids} "
          f"binder={binder_len} total={target_len + binder_len}")

    results = []
    for traj_i in range(n_trajectories):
        traj_seed = seed + traj_i * 1000
        print(f"\nTrajectory {traj_i + 1}/{n_trajectories} (seed={traj_seed})")
        clear_mem()
        try:
            af_model = mk_afdesign_model(
                protocol="binder", debug=False, data_dir=DATA_DIR,
                use_multimer=True, num_recycles=1, best_metric="loss",
            )
            af_model.prep_inputs(
                pdb_filename=str(target_pdb), chain=chain_str,
                binder_len=binder_len, hotspot=hotspot_str,
                seed=traj_seed, rm_aa="C",
            )
            af_model.opt["weights"].update({
                "plddt": 0.1, "i_pae": i_pae_weight,
                "con": 1.0, "i_con": 1.0,
            })

            t0 = time.time()
            nm = num_design_models
            af_model.design_logits(iters=50, e_soft=0.9, num_models=nm,
                                   sample_models=True, save_best=True)
            best_plddt = float(af_model._tmp["best"]["aux"]["plddt"][-binder_len:].mean())
            print(f"  [S1] logits(50,nm={nm}) pLDDT={best_plddt:.3f} ({time.time() - t0:.0f}s)")
            if best_plddt <= 0.40:
                results.append({"traj": traj_i, "status": "S1_fail", "plddt": best_plddt})
                continue
            af_model.design_logits(iters=25, e_soft=1.0, num_models=nm,
                                   sample_models=True, save_best=True)
            af_model.design_soft(iters=45, e_temp=1e-2, ramp_recycles=False)
            best_plddt = float(af_model._tmp["best"]["aux"]["plddt"][-binder_len:].mean())
            print(f"  [S2] soft(45) pLDDT={best_plddt:.3f} ({time.time() - t0:.0f}s)")
            if best_plddt <= 0.40:
                results.append({"traj": traj_i, "status": "S2_fail", "plddt": best_plddt})
                continue
            af_model.design_hard(iters=5, temp=1e-2, dropout=False, ramp_recycles=False)
            best_plddt = float(af_model._tmp["best"]["aux"]["plddt"][-binder_len:].mean())
            print(f"  [S3] hard(5) pLDDT={best_plddt:.3f} ({time.time() - t0:.0f}s)")
            tries = max(1, int(binder_len * 0.01))
            af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=15, tries=tries)
            best_plddt = float(af_model._tmp["best"]["aux"]["plddt"][-binder_len:].mean())
            print(f"  [S4] semigreedy(15) pLDDT={best_plddt:.3f} ({time.time() - t0:.0f}s)")

            seqs = af_model.get_seqs()
            best_seq = seqs[0] if seqs else ""
            _iptm = af_model.aux["log"].get("i_ptm")
            i_ptm = _iptm[-1] if isinstance(_iptm, list) and _iptm else _iptm
            _ipae = af_model.aux["log"].get("i_pae")
            i_pae = _ipae[-1] if isinstance(_ipae, list) and _ipae else _ipae
            pdb_path = out_dir / f"traj_{traj_i}_best.pdb"
            af_model.save_pdb(str(pdb_path))
            results.append({
                "traj": traj_i, "status": "designed", "seed": traj_seed,
                "sequence": best_seq, "length": len(best_seq),
                "binder_plddt": round(best_plddt, 3),
                "i_ptm": round(float(i_ptm), 3) if i_ptm is not None else None,
                "i_pae": round(float(i_pae), 3) if i_pae is not None else None,
                "pdb": str(pdb_path), "time_s": round(time.time() - t0, 1),
            })
            print(f"  OK pLDDT={best_plddt:.3f} i_ptm={i_ptm} i_pae={i_pae}")
        except Exception as e:
            import traceback
            print(f"  FAIL Trajectory {traj_i}: {e}")
            traceback.print_exc()
            results.append({"traj": traj_i, "status": "error", "error": str(e)[:200]})
        finally:
            clear_mem()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--chain", default="A,B", help="target chains in the pdb")
    ap.add_argument("--hotspot", required=True,
                    help="chain-prefixed hotspots, e.g. A88,A96")
    ap.add_argument("--binder-len", type=int, default=80)
    ap.add_argument("--n-trajectories", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-pae-weight", type=float, default=0.5)
    ap.add_argument("--num-design-models", type=int, default=2)
    ap.add_argument("--data-dir", default="/home/protein_design2026/external/af2_params")
    args = ap.parse_args()
    DATA_DIR = args.data_dir
    results = design_binder(
        target_pdb=args.target, chain_str=args.chain,
        hotspot_str=args.hotspot, binder_len=args.binder_len,
        n_trajectories=args.n_trajectories, seed=args.seed, out_dir=args.out,
        i_pae_weight=args.i_pae_weight, num_design_models=args.num_design_models,
    )
    (Path(args.out) / "design_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n完成: {sum(1 for r in results if r['status'] == 'designed')}/{len(results)}")
