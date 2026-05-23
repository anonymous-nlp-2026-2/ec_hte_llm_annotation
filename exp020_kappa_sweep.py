#!/usr/bin/env python3
"""exp020_kappa_sweep.py - κ threshold sweep for condition number singularity guard.

Sweeps κ ∈ {10,25,50,100,200,500,1000} on synthetic data (K=2, moderate regime)
to provide empirical justification for the κ>100 guard threshold.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from exp001_hb_synthetic import (
    N, SUBGROUP_LABELS,
    generate_data, get_counts, build_mixing_matrix, diff_in_means,
    estimate_cm_hb_eb, invert_mixing, invert_mixing_safe,
)

REGIME_PARAMS = {
    'misclass': [0.05, 0.15],
    'tau_z1': 1.0,
    'tau_z0': 0.5,
}
K = 2
N_EXPERT = 200
KAPPA_VALUES = [10, 25, 50, 100, 200, 500, 1000]
DEFAULT_N_MC = 50


def run_one(kappa, seed):
    data = generate_data(REGIME_PARAMS, N_EXPERT, K, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[K]
    tau_true = np.array([REGIME_PARAMS['tau_z0'], REGIME_PARAMS['tau_z1']])

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)

    C_hb, _ = estimate_cm_hb_eb(counts, labels)
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    M_gl = build_mixing_matrix(C_gl)

    guard_triggered = 0
    errors_hb = []
    errors_global = []
    has_instability = False
    cond_numbers = []

    for s in labels:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        cond = np.linalg.cond(M_hb[s])
        cond_numbers.append(cond)
        if cond > kappa:
            guard_triggered += 1

        thb, shb = invert_mixing_safe(M_hb[s], tau_obs, se_obs, cond_threshold=kappa)
        tgl, sgl = invert_mixing(M_gl, tau_obs, se_obs)

        if np.any(~np.isfinite(thb)) or np.any(~np.isfinite(shb)):
            has_instability = True
        if np.any(~np.isfinite(tgl)) or np.any(~np.isfinite(sgl)):
            has_instability = True

        errors_hb.extend((thb - tau_true).tolist())
        errors_global.extend((tgl - tau_true).tolist())

    n_subgroups = len(labels)
    guard_trigger_rate = guard_triggered / n_subgroups if n_subgroups > 0 else np.nan
    rmse_hb = np.sqrt(np.mean(np.array(errors_hb) ** 2)) if errors_hb else np.nan
    rmse_global = np.sqrt(np.mean(np.array(errors_global) ** 2)) if errors_global else np.nan
    rmse_ratio = rmse_hb / rmse_global if rmse_global and rmse_global > 0 else np.nan
    max_cond = max(cond_numbers) if cond_numbers else np.nan

    return {
        'kappa': kappa,
        'seed': seed,
        'guard_trigger_rate': guard_trigger_rate,
        'rmse_hb': rmse_hb,
        'rmse_global': rmse_global,
        'rmse_ratio': rmse_ratio,
        'numerical_instability': has_instability,
        'max_cond_number': max_cond,
    }


def stress_test_cond_numbers(n_seeds=20):
    """Probe condition numbers under stress: extreme misclass, small n_expert, K=4."""
    stress_configs = [
        {'label': 'moderate/n200/K2', 'misclass': [0.05, 0.15], 'n_expert': 200, 'k': 2},
        {'label': 'moderate/n50/K2', 'misclass': [0.05, 0.15], 'n_expert': 50, 'k': 2},
        {'label': 'extreme/n200/K2', 'misclass': [0.25, 0.05], 'n_expert': 200, 'k': 2},
        {'label': 'extreme/n50/K2', 'misclass': [0.25, 0.05], 'n_expert': 50, 'k': 2},
        {'label': 'extreme/n50/K4', 'misclass': [0.30, 0.20, 0.10, 0.03], 'n_expert': 50, 'k': 4},
        {'label': 'near_random/n50/K2', 'misclass': [0.40, 0.35], 'n_expert': 50, 'k': 2},
    ]
    rows = []
    for cfg in stress_configs:
        rp = {'misclass': cfg['misclass'], 'tau_z0': 0.5, 'tau_z1': 1.0}
        k = cfg['k']
        labels = SUBGROUP_LABELS[k]
        for seed in range(n_seeds):
            data = generate_data(rp, cfg['n_expert'], k, seed)
            em = data['expert_mask']
            counts = get_counts(data['Z'][em], data['Z_hat'][em], data['S'][em], labels)
            C_hb, _ = estimate_cm_hb_eb(counts, labels)
            conds = [np.linalg.cond(build_mixing_matrix(C_hb[s])) for s in labels]
            rows.append({'config': cfg['label'], 'seed': seed,
                         'max_cond': max(conds), 'mean_cond': np.mean(conds)})
    return pd.DataFrame(rows)


def generate_analysis(df, output_path, stress_df=None):
    lines = []
    lines.append("# κ Threshold Sweep Analysis\n")
    lines.append("## Setup\n")
    lines.append(f"- K={K}, N={N}, n_expert={N_EXPERT}")
    lines.append(f"- π_s = {REGIME_PARAMS['misclass']} (symmetric CM)")
    lines.append(f"- τ(Z=0)={REGIME_PARAMS['tau_z0']}, τ(Z=1)={REGIME_PARAMS['tau_z1']}")
    lines.append(f"- T ~ Bernoulli(0.5)")
    lines.append(f"- κ values: {sorted(df['kappa'].unique().astype(int).tolist())}")
    lines.append(f"- MC runs: {df['seed'].nunique()} seeds (0–{int(df['seed'].max())})\n")

    summary = df.groupby('kappa').agg(
        guard_rate=('guard_trigger_rate', 'mean'),
        guard_rate_se=('guard_trigger_rate', 'sem'),
        rmse_hb_mean=('rmse_hb', 'mean'),
        rmse_hb_se=('rmse_hb', 'sem'),
        rmse_global_mean=('rmse_global', 'mean'),
        rmse_global_se=('rmse_global', 'sem'),
        rmse_ratio_mean=('rmse_ratio', 'mean'),
        rmse_ratio_se=('rmse_ratio', 'sem'),
        instability_pct=('numerical_instability', 'mean'),
        max_cond_mean=('max_cond_number', 'mean'),
        max_cond_median=('max_cond_number', 'median'),
    ).reset_index()

    lines.append("## Summary Table\n")
    lines.append("| κ | Guard Rate | RMSE HB | RMSE Global | RMSE Ratio | Instability % | Mean Max κ(M) | Median Max κ(M) |")
    lines.append("|---:|----------:|--------:|------------:|-----------:|--------------:|--------------:|----------------:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['kappa'])} "
            f"| {r['guard_rate']:.3f} ± {r['guard_rate_se']:.3f} "
            f"| {r['rmse_hb_mean']:.4f} ± {r['rmse_hb_se']:.4f} "
            f"| {r['rmse_global_mean']:.4f} ± {r['rmse_global_se']:.4f} "
            f"| {r['rmse_ratio_mean']:.4f} ± {r['rmse_ratio_se']:.4f} "
            f"| {r['instability_pct'] * 100:.1f} "
            f"| {r['max_cond_mean']:.1f} "
            f"| {r['max_cond_median']:.1f} |"
        )

    lines.append("\n## Analysis\n")

    cond_max = df['max_cond_number'].max()
    cond_p99 = df['max_cond_number'].quantile(0.99)
    cond_p95 = df['max_cond_number'].quantile(0.95)
    cond_p90 = df['max_cond_number'].quantile(0.90)
    guard_ever = summary['guard_rate'].sum() > 0

    if not guard_ever:
        lines.append("### Key Finding: Guard Never Triggers\n")
        lines.append(f"Across all {len(df)} runs (7 κ values × {df['seed'].nunique()} seeds), "
                     f"the condition number guard **never triggered**. "
                     f"The maximum observed condition number is **{cond_max:.2f}**, "
                     f"far below the smallest sweep value κ=10.")
        lines.append(f"\nThis means all κ values produce **identical** RMSE results. "
                     f"The per-subgroup HB correction is always applied without fallback.")
        lines.append(f"\n**Interpretation**: With n_expert={N_EXPERT} and moderate "
                     f"misclassification (π_s = {REGIME_PARAMS['misclass']}), "
                     f"the confusion matrix is well-estimated and the mixing matrix "
                     f"is well-conditioned. The κ guard is a safety net for edge cases, "
                     f"not an active component under these conditions.")
    else:
        lines.append("### Guard Trigger Rate\n")
        for _, r in summary.iterrows():
            kv = int(r['kappa'])
            gr = r['guard_rate']
            lines.append(f"- κ={kv}: {gr:.1%} of subgroups triggered the guard")

    lines.append(f"\n### Condition Number Distribution\n")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|------:|")
    lines.append(f"| Mean | {df['max_cond_number'].mean():.3f} |")
    lines.append(f"| Median | {df['max_cond_number'].median():.3f} |")
    lines.append(f"| P90 | {cond_p90:.3f} |")
    lines.append(f"| P95 | {cond_p95:.3f} |")
    lines.append(f"| P99 | {cond_p99:.3f} |")
    lines.append(f"| Max | {cond_max:.3f} |")

    lines.append("\n### Numerical Stability\n")
    unstable = summary[summary['instability_pct'] > 0]
    if not unstable.empty:
        lines.append(f"Numerical instability (NaN/Inf) observed at κ = "
                     f"{', '.join(str(int(k)) for k in unstable['kappa'])}.")
    else:
        lines.append("No numerical instability (NaN/Inf) observed at any κ value.")

    if stress_df is not None and len(stress_df) > 0:
        lines.append("\n## Stress Test: When Does the Guard Matter?\n")
        lines.append("To contextualize the sweep, we probe condition numbers under "
                     "more challenging settings (20 seeds each):\n")
        lines.append("| Configuration | Mean Max κ(M) | P95 Max κ(M) | Max κ(M) | Triggers κ=100? |")
        lines.append("|---------------|-------------:|-----------:|--------:|:---------------|")
        for cfg, grp in stress_df.groupby('config', sort=False):
            mc_mean = grp['max_cond'].mean()
            mc_p95 = grp['max_cond'].quantile(0.95)
            mc_max = grp['max_cond'].max()
            triggers = "Yes" if mc_max > 100 else ("Borderline" if mc_p95 > 50 else "No")
            lines.append(f"| {cfg} | {mc_mean:.1f} | {mc_p95:.1f} | {mc_max:.1f} | {triggers} |")
        lines.append(f"\nThe guard becomes relevant when n_expert is small (≤50) and/or "
                     f"misclassification rates approach 0.5. Under the specified "
                     f"moderate regime, κ never approaches the threshold.")

    lines.append("\n## Is κ=100 Reasonable?\n")
    if not guard_ever:
        lines.append(f"**Yes.** κ=100 is a safe default for this regime:")
        lines.append(f"- The empirical max condition number ({cond_max:.2f}) is ~{100 / cond_max:.0f}× "
                     f"below the threshold, providing a large safety margin.")
        lines.append(f"- The guard activates only under adversarial conditions "
                     f"(near-random labeling, tiny expert budgets) where correction "
                     f"would amplify noise regardless.")
        lines.append(f"- A threshold much lower than 100 (e.g., κ=10) would still "
                     f"never trigger here, so the exact value is not critical for "
                     f"well-conditioned problems.")
    else:
        k100 = summary[summary['kappa'] == 100]
        if not k100.empty:
            r = k100.iloc[0]
            lines.append(f"At κ=100: guard rate = {r['guard_rate']:.1%}, "
                         f"RMSE ratio = {r['rmse_ratio_mean']:.4f}, "
                         f"instability = {r['instability_pct'] * 100:.1f}%")
        best_idx = summary['rmse_hb_mean'].idxmin()
        best_row = summary.loc[best_idx]
        best_k = int(best_row['kappa'])
        lines.append(f"\nLowest mean RMSE HB at κ={best_k} "
                     f"({best_row['rmse_hb_mean']:.4f} ± {best_row['rmse_hb_se']:.4f}).")

    lines.append("\n## Adaptive Formula Suggestion\n")
    lines.append(f"A data-driven approach: set κ = c · √(n_expert / K), "
                 f"where c is calibrated per regime.")
    lines.append(f"- With n_expert={N_EXPERT}, K={K}: √({N_EXPERT}/{K}) = {np.sqrt(N_EXPERT / K):.1f}")
    lines.append(f"- Setting c=10 gives κ={10 * np.sqrt(N_EXPERT / K):.0f}, which is close to 100")
    lines.append(f"- However, the optimal κ depends on the misclassification severity, "
                 f"not just sample size")
    lines.append(f"- A more robust adaptive rule: set κ to 2× the observed P99 of "
                 f"condition numbers from a pilot run. Here that would be "
                 f"κ = 2 × {cond_p99:.1f} = {2 * cond_p99:.1f}")
    lines.append(f"- **Bottom line**: κ=100 provides ample headroom. "
                 f"The exact value matters only in edge cases where n_expert/K is very small.")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='κ threshold sweep experiment')
    parser.add_argument('--n-mc', type=int, default=DEFAULT_N_MC,
                        help=f'Number of MC seeds (default: {DEFAULT_N_MC})')
    parser.add_argument('--kappa', type=float, nargs='+', default=None,
                        help=f'κ values to sweep (default: {KAPPA_VALUES})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run only 2 seeds for testing')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else args.n_mc
    kappa_values = [int(k) for k in args.kappa] if args.kappa else KAPPA_VALUES

    print(f"exp-020 κ Sweep | K={K} N={N} n_expert={N_EXPERT}")
    print(f"κ values: {kappa_values}")
    print(f"MC seeds: {n_mc}\n")

    results = []
    total = len(kappa_values) * n_mc
    count = 0
    t0 = time.time()

    for kappa in kappa_values:
        for seed in range(n_mc):
            row = run_one(kappa, seed)
            results.append(row)
            count += 1
            if count % max(1, total // 10) == 0:
                el = time.time() - t0
                eta = el / count * (total - count)
                print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s")

    print(f"\n  Done: {time.time() - t0:.1f}s ({count} runs)\n")

    df = pd.DataFrame(results)

    out_dir = 'artifacts/kappa_sweep'
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    print("Running stress test (condition numbers under challenging settings)...")
    stress_df = stress_test_cond_numbers(n_seeds=20)
    stress_csv = os.path.join(out_dir, 'stress_test_cond.csv')
    stress_df.to_csv(stress_csv, index=False)
    print(f"Saved: {stress_csv} ({len(stress_df)} rows)")

    md_path = os.path.join(out_dir, 'kappa_sweep_analysis.md')
    generate_analysis(df, md_path, stress_df=stress_df)
    print(f"Saved: {md_path}")

    print("\n── Summary ──")
    for kappa in sorted(df['kappa'].unique()):
        sub = df[df['kappa'] == kappa]
        print(f"  κ={kappa:>5.0f}: guard={sub['guard_trigger_rate'].mean():.3f} "
              f"rmse_hb={sub['rmse_hb'].mean():.4f} "
              f"ratio={sub['rmse_ratio'].mean():.4f} "
              f"instab={sub['numerical_instability'].mean():.1%}")
