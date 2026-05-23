#!/usr/bin/env python3
"""
exp021_seeds_supplement.py — Supplement 50 seeds (50-99) for 3 Table 1 configs.

Runs seeds 0-99 (re-running 0-49 for raw data) for:
  1. K=2, extreme
  2. K=2, moderate
  3. K=4, extreme
Outputs raw per-seed data and 100-seed aggregated results.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from exp001_hb_synthetic import (
    N, REGIMES, SUBGROUP_LABELS, DEFAULT_N_EXPERTS,
    generate_data, get_counts, diff_in_means,
    build_mixing_matrix, invert_mixing, invert_mixing_safe,
    estimate_cm_mle, estimate_cm_hb_eb, aggregate_results,
)

TARGET_CONFIGS = [
    (2, 'extreme'),
    (2, 'moderate'),
    (4, 'extreme'),
]

N_EXPERTS = DEFAULT_N_EXPERTS  # [50, 100, 250, 500, 1000, 2000]
SEED_RANGE = range(0, 100)
SUPPLEMENT_SEEDS = range(50, 100)

COND_THRESHOLD = 100


def run_one_mc_full(regime_name, regime_params, n_expert, k, seed):
    """Run one MC iteration returning rows for all methods (base + reg for K=4)."""
    data = generate_data(regime_params, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]
    tau_z0, tau_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)
    C_mle = estimate_cm_mle(counts, labels)
    C_hb, hb_diag = estimate_cm_hb_eb(counts, labels)

    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5

    M_gl = build_mixing_matrix(C_gl)
    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    base = {'regime': regime_name, 'n_expert': n_expert, 'k_subgroups': k, 'mc_run': seed}
    rows = []

    for s in labels:
        sm = S == s
        th0_or, se0_or = diff_in_means(Y, T, sm & (Z == 0))
        th1_or, se1_or = diff_in_means(Y, T, sm & (Z == 1))
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))

        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        tgc, sgc = invert_mixing(M_gl, tau_obs, se_obs)
        tml, sml = invert_mixing(M_mle[s], tau_obs, se_obs)
        thb, shb = invert_mixing(M_hb[s], tau_obs, se_obs)

        def mr(method, th0, th1, se0, se1):
            return {**base, 'subgroup': s, 'method': method,
                    'tau_hat_z0': th0, 'tau_hat_z1': th1,
                    'se_z0': se0, 'se_z1': se1,
                    'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1}

        rows.append(mr('oracle', th0_or, th1_or, se0_or, se1_or))
        rows.append(mr('naive', th0_n, th1_n, se0_n, se1_n))
        rows.append(mr('global_corrected', tgc[0], tgc[1], sgc[0], sgc[1]))
        rows.append(mr('mle_laplace', tml[0], tml[1], sml[0], sml[1]))
        rows.append(mr('hb_dirichlet', thb[0], thb[1], shb[0], shb[1]))

        if k == 4 and n_expert in [50, 100]:
            tml_r, sml_r = invert_mixing_safe(M_mle[s], tau_obs, se_obs, COND_THRESHOLD)
            thb_r, shb_r = invert_mixing_safe(M_hb[s], tau_obs, se_obs, COND_THRESHOLD)
            rows.append(mr('mle_laplace_reg', tml_r[0], tml_r[1], sml_r[0], sml_r[1]))
            rows.append(mr('hb_dirichlet_reg', thb_r[0], thb_r[1], shb_r[0], shb_r[1]))

    return rows


def main():
    os.makedirs('artifacts/seeds_100', exist_ok=True)

    total = len(TARGET_CONFIGS) * len(N_EXPERTS) * len(SEED_RANGE)
    print(f"exp021: {len(TARGET_CONFIGS)} configs × {len(N_EXPERTS)} n_experts × "
          f"{len(SEED_RANGE)} seeds = {total} runs (EB)")

    # Time a single run
    k0, r0 = TARGET_CONFIGS[0]
    t0 = time.time()
    _ = run_one_mc_full(r0, REGIMES[k0][r0], 500, k0, seed=9999)
    t_one = time.time() - t0
    print(f"Single run: {t_one:.2f}s → estimated total: {t_one * total / 60:.0f} min\n")

    all_rows = []
    count = 0
    t0 = time.time()

    for k, regime in TARGET_CONFIGS:
        rp = REGIMES[k][regime]
        for ne in N_EXPERTS:
            for seed in SEED_RANGE:
                rows = run_one_mc_full(regime, rp, ne, k, seed)
                all_rows.extend(rows)
                count += 1
                if count % max(1, total // 20) == 0:
                    el = time.time() - t0
                    eta = el / count * (total - count)
                    print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s | "
                          f"K={k} {regime} ne={ne} seed={seed}")

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.1f}s ({count} runs)\n")

    df_raw = pd.DataFrame(all_rows)
    raw_cols = ['regime', 'n_expert', 'k_subgroups', 'mc_run', 'subgroup', 'method',
                'tau_hat_z0', 'tau_hat_z1', 'se_z0', 'se_z1',
                'tau_true_z0', 'tau_true_z1']

    # Save supplement (seeds 50-99 only)
    df_supp = df_raw[df_raw['mc_run'].isin(SUPPLEMENT_SEEDS)].copy()
    supp_path = 'artifacts/seeds_100/supplement_seeds50_99.csv'
    df_supp[raw_cols].to_csv(supp_path, index=False)
    print(f"Saved supplement: {supp_path} ({len(df_supp)} rows)")

    # Save full raw data
    full_raw_path = 'artifacts/seeds_100/full_raw_100seeds.csv'
    df_raw[raw_cols].to_csv(full_raw_path, index=False)
    print(f"Saved full raw: {full_raw_path} ({len(df_raw)} rows)")

    # Aggregate all 100 seeds
    df_agg_100 = aggregate_results(df_raw)
    agg_path = 'artifacts/seeds_100/combined_100_seeds.csv'
    df_agg_100.to_csv(agg_path, index=False)
    print(f"Saved combined aggregated: {agg_path} ({len(df_agg_100)} rows)")

    # Also aggregate seeds 0-49 and 50-99 separately for comparison
    df_first50 = df_raw[df_raw['mc_run'] < 50]
    df_last50 = df_raw[df_raw['mc_run'] >= 50]
    df_agg_50a = aggregate_results(df_first50)
    df_agg_50b = aggregate_results(df_last50)

    # Generate Table 1 markdown
    generate_table1(df_agg_100, df_agg_50a, df_agg_50b)


def generate_table1(df_100, df_50a, df_50b):
    """Generate Table 1 markdown with 100-seed RMSE ± SE, plus comparison."""
    lines = []
    lines.append("# Table 1: RMSE ± SE (100 seeds, EB method)\n")

    methods_k2 = ['oracle', 'naive', 'global_corrected', 'mle_laplace', 'hb_dirichlet']
    methods_k4 = methods_k2 + ['mle_laplace_reg', 'hb_dirichlet_reg']
    display = {
        'oracle': 'Oracle', 'naive': 'Naive', 'global_corrected': 'Global',
        'mle_laplace': 'MLE', 'hb_dirichlet': 'HB',
        'mle_laplace_reg': 'MLE-reg', 'hb_dirichlet_reg': 'HB-reg',
    }

    for k, regime in TARGET_CONFIGS:
        rp = REGIMES[k][regime]
        methods = methods_k4 if k == 4 else methods_k2
        lines.append(f"\n## K={k}, {regime}")
        lines.append(f"pi_s = {rp['misclass']}, tau(Z=0)={rp['tau_z0']}, tau(Z=1)={rp['tau_z1']}\n")

        header = "| n_expert | Subgroup | " + " | ".join(display[m] for m in methods) + " |"
        sep = "|" + "|".join(["---"] * (2 + len(methods))) + "|"
        lines.append(header)
        lines.append(sep)

        labels = SUBGROUP_LABELS[k]
        for ne in DEFAULT_N_EXPERTS:
            for s in labels:
                cells = [f"{ne}", s]
                for m in methods:
                    row = df_100[(df_100['regime'] == regime) &
                                (df_100['k_subgroups'] == k) &
                                (df_100['n_expert'] == ne) &
                                (df_100['subgroup'] == s) &
                                (df_100['method'] == m)]
                    if row.empty:
                        cells.append("--")
                        continue
                    r = row.iloc[0]
                    rmse_avg = (r['rmse_z0'] + r['rmse_z1']) / 2
                    se_avg = (r['rmse_z0_se'] + r['rmse_z1_se']) / 2
                    cells.append(f"{rmse_avg:.4f} +/- {se_avg:.4f}")
                lines.append("| " + " | ".join(cells) + " |")

    # Compact comparison: method-averaged RMSE per (config, n_expert)
    lines.append("\n\n# 50-seed vs 100-seed Stability Comparison\n")
    lines.append("Subgroup-averaged RMSE (mean of z0 and z1, averaged across subgroups).\n")
    lines.append("| Config | n_expert | Method | RMSE (50 seeds) | RMSE (100 seeds) | Delta% |")
    lines.append("|---|---|---|---|---|---|")

    for k, regime in TARGET_CONFIGS:
        methods = methods_k4 if k == 4 else methods_k2
        for m in methods:
            for ne in DEFAULT_N_EXPERTS:
                r50 = df_50a[(df_50a['regime'] == regime) &
                             (df_50a['k_subgroups'] == k) &
                             (df_50a['n_expert'] == ne) &
                             (df_50a['method'] == m)]
                r100 = df_100[(df_100['regime'] == regime) &
                              (df_100['k_subgroups'] == k) &
                              (df_100['n_expert'] == ne) &
                              (df_100['method'] == m)]
                if r50.empty or r100.empty:
                    continue

                rmse50_avg = r50[['rmse_z0', 'rmse_z1']].mean().mean()
                rmse100_avg = r100[['rmse_z0', 'rmse_z1']].mean().mean()
                pct = (rmse100_avg - rmse50_avg) / rmse50_avg * 100 if rmse50_avg > 1e-10 else 0

                lines.append(f"| K={k}/{regime} | {ne} | {display[m]} | "
                             f"{rmse50_avg:.4f} | {rmse100_avg:.4f} | {pct:+.1f}% |")

    # Summary statistics
    lines.append("\n\n# Summary: Stability Metrics\n")
    all_deltas = []
    for k, regime in TARGET_CONFIGS:
        methods = methods_k4 if k == 4 else methods_k2
        for m in methods:
            for ne in DEFAULT_N_EXPERTS:
                r50 = df_50a[(df_50a['regime'] == regime) &
                             (df_50a['k_subgroups'] == k) &
                             (df_50a['n_expert'] == ne) &
                             (df_50a['method'] == m)]
                r100 = df_100[(df_100['regime'] == regime) &
                              (df_100['k_subgroups'] == k) &
                              (df_100['n_expert'] == ne) &
                              (df_100['method'] == m)]
                if r50.empty or r100.empty:
                    continue
                rmse50 = r50[['rmse_z0', 'rmse_z1']].mean().mean()
                rmse100 = r100[['rmse_z0', 'rmse_z1']].mean().mean()
                pct = abs(rmse100 - rmse50) / rmse50 * 100 if rmse50 > 1e-10 else 0
                all_deltas.append(pct)

    lines.append(f"- Total cells compared: {len(all_deltas)}")
    lines.append(f"- Median |Delta|: {np.median(all_deltas):.1f}%")
    lines.append(f"- Max |Delta|: {np.max(all_deltas):.1f}%")
    lines.append(f"- Cells with |Delta| < 5%: {sum(1 for d in all_deltas if d < 5)}/{len(all_deltas)} "
                 f"({sum(1 for d in all_deltas if d < 5)/len(all_deltas)*100:.0f}%)")
    lines.append(f"- Cells with |Delta| < 10%: {sum(1 for d in all_deltas if d < 10)}/{len(all_deltas)} "
                 f"({sum(1 for d in all_deltas if d < 10)/len(all_deltas)*100:.0f}%)")

    table_path = 'artifacts/seeds_100/table1_100seeds.md'
    with open(table_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved table: {table_path}")


if __name__ == '__main__':
    main()
