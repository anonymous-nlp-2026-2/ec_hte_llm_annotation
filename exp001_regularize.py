#!/usr/bin/env python3
"""exp001_regularize.py - Rerun affected cells with condition-number-guarded inversion."""

import argparse
import time
import numpy as np
import pandas as pd

from exp001_hb_synthetic import (
    REGIMES, SUBGROUP_LABELS,
    generate_data, get_counts, build_mixing_matrix, diff_in_means,
    estimate_cm_mle, estimate_cm_hb_eb, estimate_cm_hb_gibbs,
    invert_mixing_safe, aggregate_results,
)

AFFECTED_CELLS = [
    (4, 'extreme', 50),
    (4, 'extreme', 100),
    (4, 'moderate', 50),
    (4, 'moderate', 100),
    (4, 'homogeneous', 50),
    (4, 'homogeneous', 100),
]


def run_one_mc_reg(regime_name, regime_params, n_expert, k, seed,
                   cond_threshold=100, n_chains=4, n_iter=4000, n_warmup=2000):
    data = generate_data(regime_params, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]
    tau_z0, tau_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)
    C_mle = estimate_cm_mle(counts, labels)
    C_hb, hb_diag = estimate_cm_hb_gibbs(counts, labels, seed=seed,
                                          n_chains=n_chains, n_iter=n_iter,
                                          n_warmup=n_warmup)

    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    base = {'regime': regime_name, 'n_expert': n_expert,
            'k_subgroups': k, 'mc_run': seed}
    rows = []
    guard_hb = 0
    guard_mle = 0

    for s in labels:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))

        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        cond_mle = np.linalg.cond(M_mle[s])
        cond_hb = np.linalg.cond(M_hb[s])
        if cond_mle > cond_threshold:
            guard_mle += 1
        if cond_hb > cond_threshold:
            guard_hb += 1

        tml_r, sml_r = invert_mixing_safe(M_mle[s], tau_obs, se_obs, cond_threshold)
        thb_r, shb_r = invert_mixing_safe(M_hb[s], tau_obs, se_obs, cond_threshold)

        def mr(method, th0, th1, se0, se1):
            return {**base, 'subgroup': s, 'method': method,
                    'tau_hat_z0': th0, 'tau_hat_z1': th1,
                    'se_z0': se0, 'se_z1': se1,
                    'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1}

        rows.append(mr('mle_laplace_reg', tml_r[0], tml_r[1], sml_r[0], sml_r[1]))
        rows.append(mr('hb_dirichlet_reg', thb_r[0], thb_r[1], shb_r[0], shb_r[1]))

    return rows, hb_diag, guard_hb, guard_mle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=100)
    parser.add_argument('--cond-threshold', type=float, default=100)
    args = parser.parse_args()

    print(f"exp-001 Regularization fix v2 | cond_threshold={args.cond_threshold}")
    print(f"Affected cells: {AFFECTED_CELLS}")
    print(f"n_mc={args.n_mc}")
    print(f"MCMC config: n_chains=4, n_iter=4000, n_warmup=2000\n")

    # ── MCMC diagnostic comparison (before/after) ────────────────────────────
    print("Running MCMC diagnostic comparison (seed=0, per config)...")
    diag_rows = []
    for k, regime, n_expert in AFFECTED_CELLS:
        rp = REGIMES[k][regime]
        d = generate_data(rp, n_expert, k, seed=0)
        em = d['expert_mask']
        labels = SUBGROUP_LABELS[k]
        counts = get_counts(d['Z'][em], d['Z_hat'][em], d['S'][em], labels)

        _, diag_before = estimate_cm_hb_gibbs(counts, labels, seed=0,
                                               n_chains=2, n_iter=2000, n_warmup=1000)
        _, diag_after = estimate_cm_hb_gibbs(counts, labels, seed=0,
                                              n_chains=4, n_iter=4000, n_warmup=2000)
        diag_rows.append({
            'config': f"K={k}/{regime}/n={n_expert}",
            'rhat_before': diag_before['rhat_max'],
            'rhat_after': diag_after['rhat_max'],
            'ess_before': diag_before['ess_min'],
            'ess_after': diag_after['ess_min'],
        })

    print("\n" + "=" * 100)
    print("MCMC Diagnostic Comparison (seed=0)")
    print(f"{'Config':>30s} | {'Rhat_max (2ch)':>14s} | {'Rhat_max (4ch)':>14s} | "
          f"{'ESS_min (2ch)':>14s} | {'ESS_min (4ch)':>14s}")
    print("-" * 100)
    for r in diag_rows:
        print(f"{r['config']:>30s} | {r['rhat_before']:14.4f} | {r['rhat_after']:14.4f} | "
              f"{r['ess_before']:14.1f} | {r['ess_after']:14.1f}")
    print("=" * 100 + "\n")

    # ── Main MC loop ─────────────────────────────────────────────────────────
    all_rows = []
    guard_stats = {}
    diag_agg = {}
    total = len(AFFECTED_CELLS) * args.n_mc
    count = 0
    t0 = time.time()

    for k, regime, n_expert in AFFECTED_CELLS:
        key = (k, regime, n_expert)
        guard_stats[key] = {'total': 0, 'guard_hb': 0, 'guard_mle': 0}
        diag_agg[key] = {'rhat_max': [], 'ess_min': []}
        rp = REGIMES[k][regime]
        for mc in range(args.n_mc):
            rows, hb_diag, g_hb, g_mle = run_one_mc_reg(
                regime, rp, n_expert, k, seed=mc,
                cond_threshold=args.cond_threshold,
                n_chains=4, n_iter=4000, n_warmup=2000)
            all_rows.extend(rows)
            guard_stats[key]['total'] += 1
            guard_stats[key]['guard_hb'] += g_hb
            guard_stats[key]['guard_mle'] += g_mle
            diag_agg[key]['rhat_max'].append(hb_diag['rhat_max'])
            diag_agg[key]['ess_min'].append(hb_diag['ess_min'])
            count += 1
            if count % 50 == 0:
                el = time.time() - t0
                print(f"  [{count}/{total}] {el:.0f}s elapsed")

    print(f"  Done: {time.time() - t0:.1f}s ({count} runs)\n")

    # ── Singularity guard trigger stats ──────────────────────────────────────
    print("=" * 90)
    print("Singularity Guard Trigger Statistics")
    print(f"{'Config':>30s} | {'Total MC':>8s} | {'Guard HB':>10s} | {'Guard MLE':>10s}")
    print("-" * 90)
    for key in guard_stats:
        k, regime, n_expert = key
        gs = guard_stats[key]
        config = f"K={k}/{regime}/n={n_expert}"
        print(f"{config:>30s} | {gs['total']:8d} | {gs['guard_hb']:10d} | {gs['guard_mle']:10d}")
    print("=" * 90 + "\n")

    # ── Aggregated MCMC diagnostics across MC runs ───────────────────────────
    print("=" * 100)
    print("Aggregated MCMC Diagnostics (4 chains, 4000 iter)")
    print(f"{'Config':>30s} | {'Rhat_max mean':>14s} | {'Rhat_max max':>14s} | "
          f"{'ESS_min mean':>14s} | {'ESS_min min':>14s}")
    print("-" * 100)
    for key in diag_agg:
        k, regime, n_expert = key
        config = f"K={k}/{regime}/n={n_expert}"
        rhats = diag_agg[key]['rhat_max']
        esss = diag_agg[key]['ess_min']
        print(f"{config:>30s} | {np.mean(rhats):14.4f} | {np.max(rhats):14.4f} | "
              f"{np.mean(esss):14.1f} | {np.min(esss):14.1f}")
    print("=" * 100 + "\n")

    # ── Aggregate and write results ──────────────────────────────────────────
    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate_results(df_raw)

    csv_orig = 'results/exp001_hb_results.csv'
    csv_v2 = 'results/exp001_hb_results_v2.csv'
    df_existing = pd.read_csv(csv_orig)

    # Remove previously appended reg rows (keep original 540)
    reg_methods = ['hb_dirichlet_reg', 'mle_laplace_reg']
    df_base = df_existing[~df_existing['method'].isin(reg_methods)].copy()

    agg_cols = list(df_base.columns)
    df_combined = pd.concat([df_base, df_agg[agg_cols]], ignore_index=True)
    df_combined.to_csv(csv_v2, index=False)

    # ── RMSE comparison table ────────────────────────────────────────────────
    print("=" * 120)
    print("RMSE Comparison (original vs regularized)")
    print(f"{'Config':>40s} | {'Method':>15s} | {'RMSE_z0 (orig)':>15s} | "
          f"{'RMSE_z0 (reg)':>14s} | {'RMSE_z1 (orig)':>15s} | {'RMSE_z1 (reg)':>14s}")
    print("-" * 120)

    for k, regime, n_expert in AFFECTED_CELLS:
        for subgroup in SUBGROUP_LABELS[k]:
            mask_base = ((df_base['k_subgroups'] == k) &
                         (df_base['regime'] == regime) &
                         (df_base['n_expert'] == n_expert) &
                         (df_base['subgroup'] == subgroup))

            for orig, reg in [('hb_dirichlet', 'hb_dirichlet_reg'),
                              ('mle_laplace', 'mle_laplace_reg')]:
                before = df_base[mask_base & (df_base['method'] == orig)]
                after = df_agg[(df_agg['k_subgroups'] == k) &
                               (df_agg['regime'] == regime) &
                               (df_agg['n_expert'] == n_expert) &
                               (df_agg['subgroup'] == subgroup) &
                               (df_agg['method'] == reg)]

                if before.empty or after.empty:
                    continue

                b_z0 = before['rmse_z0'].values[0]
                a_z0 = after['rmse_z0'].values[0]
                b_z1 = before['rmse_z1'].values[0]
                a_z1 = after['rmse_z1'].values[0]

                config = f"K={k}/{regime}/n={n_expert}/{subgroup}"
                print(f"{config:>40s} | {orig:>15s} | {b_z0:15.4f} | "
                      f"{a_z0:14.4f} | {b_z1:15.4f} | {a_z1:14.4f}")

    # Global method comparison
    print("-" * 120)
    for method in ['diff_in_means', 'mle_laplace', 'hb_dirichlet']:
        mask = df_base['method'] == method
        if mask.any():
            r0 = df_base.loc[mask, 'rmse_z0'].mean()
            r1 = df_base.loc[mask, 'rmse_z1'].mean()
            print(f"{'Global avg':>40s} | {method:>15s} | {r0:15.4f} | {'---':>14s} | {r1:15.4f} | {'---':>14s}")
    for method in ['mle_laplace_reg', 'hb_dirichlet_reg']:
        mask = df_agg['method'] == method
        if mask.any():
            r0 = df_agg.loc[mask, 'rmse_z0'].mean()
            r1 = df_agg.loc[mask, 'rmse_z1'].mean()
            print(f"{'Global avg (reg cells)':>40s} | {method:>15s} | {'---':>15s} | {r0:14.4f} | {'---':>15s} | {r1:14.4f}")

    print("=" * 120)
    print(f"\nWrote {csv_v2}: {len(df_base)} original + {len(df_agg)} new = {len(df_combined)} rows")
    print(f"Original file {csv_orig} unchanged.")
