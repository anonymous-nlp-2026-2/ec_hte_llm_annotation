#!/usr/bin/env python3
"""
exp013_boundary_condition.py - Boundary condition test: n_expert=50, K=4, extreme regime.

With only ~12.5 expert labels per subgroup, per-subgroup CM estimation is extremely noisy.
Tests whether stratified methods degrade vs global correction under these conditions.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from exp001_hb_synthetic import (
    generate_data, get_counts, estimate_cm_mle, estimate_cm_hb_eb,
    estimate_cm_hb_gibbs, build_mixing_matrix, invert_mixing_safe,
    diff_in_means, REGIMES, SUBGROUP_LABELS, N,
)

K = 4
REGIME_NAME = 'extreme'
REGIME_PARAMS = REGIMES[K][REGIME_NAME]
N_EXPERT = 50
LABELS = SUBGROUP_LABELS[K]


def run_one(seed, hb_method='eb'):
    data = generate_data(REGIME_PARAMS, N_EXPERT, K, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    tau_z0, tau_z1 = REGIME_PARAMS['tau_z0'], REGIME_PARAMS['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], LABELS)

    C_mle = estimate_cm_mle(counts, LABELS)
    if hb_method == 'gibbs':
        C_hb, _ = estimate_cm_hb_gibbs(counts, LABELS, seed=seed)
    else:
        C_hb, _ = estimate_cm_hb_eb(counts, LABELS)

    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5

    M_gl = build_mixing_matrix(C_gl)
    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in LABELS}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in LABELS}

    rows = []
    for s in LABELS:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        tgc, sgc = invert_mixing_safe(M_gl, tau_obs, se_obs)
        tml, sml = invert_mixing_safe(M_mle[s], tau_obs, se_obs)
        thb, shb = invert_mixing_safe(M_hb[s], tau_obs, se_obs)

        for method, t0, t1, s0, s1 in [
            ('naive',            th0_n, th1_n, se0_n, se1_n),
            ('global_corrected', tgc[0], tgc[1], sgc[0], sgc[1]),
            ('mle_laplace',      tml[0], tml[1], sml[0], sml[1]),
            ('hb_eb',            thb[0], thb[1], shb[0], shb[1]),
        ]:
            for z_label, tau_c, se_c, tau_true in [
                ('z0', t0, s0, tau_z0), ('z1', t1, s1, tau_z1)
            ]:
                lo = tau_c - 1.96 * se_c
                hi = tau_c + 1.96 * se_c
                rows.append({
                    'mc_run': seed, 'subgroup': s, 'method': method,
                    'z_level': z_label,
                    'tau_corrected': tau_c, 'tau_true': tau_true,
                    'se_corrected': se_c,
                    'coverage': int(lo <= tau_true <= hi),
                })
    return rows


def summarize(df):
    rows = []
    for (sg, method, z), grp in df.groupby(['subgroup', 'method', 'z_level']):
        err = grp['tau_corrected'].values - grp['tau_true'].values
        n_mc = len(grp)
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        cov = grp['coverage'].mean()
        rows.append({
            'subgroup': sg, 'method': method, 'z_level': z,
            'bias': bias, 'abs_bias': abs(bias), 'rmse': rmse, 'coverage': cov,
            'bias_se': err.std() / np.sqrt(n_mc),
            'rmse_se': (err ** 2).std() / np.sqrt(n_mc) / (2 * rmse) if rmse > 1e-10 else np.nan,
            'coverage_se': np.sqrt(cov * (1 - cov) / n_mc),
            'n_mc': n_mc,
        })
    return pd.DataFrame(rows)


def write_analysis(df_summary):
    methods = ['naive', 'global_corrected', 'mle_laplace', 'hb_eb']
    subgroups = list(LABELS)
    misclass = dict(zip(LABELS, REGIME_PARAMS['misclass']))

    lines = [
        '# exp-013: Boundary Condition Analysis',
        f'',
        f'**Config**: K={K}, regime={REGIME_NAME}, n_expert={N_EXPERT}, '
        f'N={N}, misclass={REGIME_PARAMS["misclass"]}',
        f'',
        f'Each subgroup gets ~{N_EXPERT / K:.0f} expert labels.',
        '',
        '## Per-subgroup Results',
        '',
    ]

    header = '| Subgroup (pi) | Method | |Bias|_z0 | RMSE_z0 | Cov_z0 | |Bias|_z1 | RMSE_z1 | Cov_z1 |'
    sep =    '|---|---|---|---|---|---|---|---|'
    lines += [header, sep]

    for s in subgroups:
        for m in methods:
            r0 = df_summary[(df_summary['subgroup'] == s) & (df_summary['method'] == m) & (df_summary['z_level'] == 'z0')]
            r1 = df_summary[(df_summary['subgroup'] == s) & (df_summary['method'] == m) & (df_summary['z_level'] == 'z1')]
            if r0.empty or r1.empty:
                continue
            r0, r1 = r0.iloc[0], r1.iloc[0]
            lines.append(
                f'| {s} ({misclass[s]:.2f}) | {m} | '
                f'{r0["abs_bias"]:.4f} | {r0["rmse"]:.4f} | {r0["coverage"]:.2f} | '
                f'{r1["abs_bias"]:.4f} | {r1["rmse"]:.4f} | {r1["coverage"]:.2f} |'
            )

    lines += ['', '## Average Across Subgroups', '']
    lines += [
        '| Method | |Bias|_z0 | RMSE_z0 | Cov_z0 | |Bias|_z1 | RMSE_z1 | Cov_z1 |',
        '|---|---|---|---|---|---|---|',
    ]
    for m in methods:
        msub = df_summary[df_summary['method'] == m]
        r0 = msub[msub['z_level'] == 'z0']
        r1 = msub[msub['z_level'] == 'z1']
        if r0.empty or r1.empty:
            continue
        lines.append(
            f'| {m} | '
            f'{r0["abs_bias"].mean():.4f} | {r0["rmse"].mean():.4f} | {r0["coverage"].mean():.2f} | '
            f'{r1["abs_bias"].mean():.4f} | {r1["rmse"].mean():.4f} | {r1["coverage"].mean():.2f} |'
        )

    lines += ['', '## RMSE Amplification (stratified vs global)', '']
    gl_rmse = {}
    for z in ['z0', 'z1']:
        r = df_summary[(df_summary['method'] == 'global_corrected') & (df_summary['z_level'] == z)]
        gl_rmse[z] = r['rmse'].mean() if not r.empty else np.nan

    lines += ['| Subgroup | Method | RMSE_z0 / Global | RMSE_z1 / Global |', '|---|---|---|---|']
    for s in subgroups:
        for m in ['mle_laplace', 'hb_eb']:
            r0 = df_summary[(df_summary['subgroup'] == s) & (df_summary['method'] == m) & (df_summary['z_level'] == 'z0')]
            r1 = df_summary[(df_summary['subgroup'] == s) & (df_summary['method'] == m) & (df_summary['z_level'] == 'z1')]
            if r0.empty or r1.empty:
                continue
            amp0 = r0.iloc[0]['rmse'] / gl_rmse['z0'] if gl_rmse['z0'] > 1e-10 else np.nan
            amp1 = r1.iloc[0]['rmse'] / gl_rmse['z1'] if gl_rmse['z1'] > 1e-10 else np.nan
            lines.append(f'| {s} ({misclass[s]:.2f}) | {m} | {amp0:.2f}x | {amp1:.2f}x |')

    # Analysis
    worst_sg = None
    worst_rmse = 0
    for s in subgroups:
        for m in ['mle_laplace', 'hb_eb']:
            for z in ['z0', 'z1']:
                r = df_summary[(df_summary['subgroup'] == s) & (df_summary['method'] == m) & (df_summary['z_level'] == z)]
                if not r.empty and r.iloc[0]['rmse'] > worst_rmse:
                    worst_rmse = r.iloc[0]['rmse']
                    worst_sg = (s, m, z)

    lines += ['', '## Key Findings', '']
    if worst_sg:
        lines.append(f'1. **Worst degradation**: subgroup {worst_sg[0]} ({worst_sg[1]}, {worst_sg[2]}) '
                      f'with RMSE={worst_rmse:.4f}')

    avg_strat_rmse_z0 = df_summary[(df_summary['method'].isin(['mle_laplace', 'hb_eb'])) & (df_summary['z_level'] == 'z0')]['rmse'].mean()
    avg_strat_rmse_z1 = df_summary[(df_summary['method'].isin(['mle_laplace', 'hb_eb'])) & (df_summary['z_level'] == 'z1')]['rmse'].mean()
    lines.append(f'2. **Average stratified RMSE**: z0={avg_strat_rmse_z0:.4f}, z1={avg_strat_rmse_z1:.4f} '
                 f'vs global z0={gl_rmse["z0"]:.4f}, z1={gl_rmse["z1"]:.4f}')
    amp_z0 = avg_strat_rmse_z0 / gl_rmse['z0'] if gl_rmse['z0'] > 1e-10 else np.nan
    amp_z1 = avg_strat_rmse_z1 / gl_rmse['z1'] if gl_rmse['z1'] > 1e-10 else np.nan
    lines.append(f'3. **Amplification factor**: {amp_z0:.2f}x (z0), {amp_z1:.2f}x (z1)')

    threshold_met = N_EXPERT / K >= 25
    lines.append(f'4. **n_expert/K = {N_EXPERT / K:.1f}** {">=25 threshold met" if threshold_met else "< 25 threshold NOT met"}. '
                 f'{"Stratified methods should be usable." if threshold_met else "Below recommended minimum — expect high variance in per-subgroup estimates."}')

    # EB vs MLE comparison
    eb_rmse = df_summary[df_summary['method'] == 'hb_eb']['rmse'].mean()
    mle_rmse = df_summary[df_summary['method'] == 'mle_laplace']['rmse'].mean()
    if mle_rmse > 1e-10:
        pct = (mle_rmse - eb_rmse) / mle_rmse * 100
        lines.append(f'5. **EB vs MLE**: EB RMSE {pct:+.1f}% relative to MLE '
                     f'({"EB helps with shrinkage" if pct > 0 else "MLE competitive"})')

    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--hb-method', choices=['eb', 'gibbs'], default='eb')
    args = parser.parse_args()

    n_mc = 5 if args.dry_run else args.n_mc
    tag = ' (DRY RUN)' if args.dry_run else ''
    print(f'exp-013 Boundary Condition{tag} | K={K} {REGIME_NAME} n_expert={N_EXPERT} n_mc={n_mc} hb={args.hb_method}')

    all_rows = []
    t0 = time.time()
    for mc in range(n_mc):
        rows = run_one(mc, hb_method=args.hb_method)
        all_rows.extend(rows)
        if (mc + 1) % max(1, n_mc // 10) == 0:
            el = time.time() - t0
            print(f'  [{mc + 1}/{n_mc}] {el:.1f}s elapsed')

    print(f'  Done: {time.time() - t0:.1f}s ({len(all_rows)} rows)')

    df = pd.DataFrame(all_rows)
    df_summary = summarize(df)

    os.makedirs('results', exist_ok=True)
    df.to_csv('results/exp013_boundary_condition.csv', index=False)
    df_summary.to_csv('results/exp013_boundary_summary.csv', index=False)
    print(f'Saved: results/exp013_boundary_condition.csv ({len(df)} rows)')
    print(f'Saved: results/exp013_boundary_summary.csv ({len(df_summary)} rows)')

    analysis = write_analysis(df_summary)
    with open('results/exp013_boundary_analysis.md', 'w') as f:
        f.write(analysis)
    print(f'Saved: results/exp013_boundary_analysis.md')

    print('\n' + analysis)


if __name__ == '__main__':
    main()
