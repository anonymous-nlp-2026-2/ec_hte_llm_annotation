#!/usr/bin/env python3
"""
exp-001 hyperprior sensitivity analysis.
Tests robustness of HB Dirichlet confusion matrix estimator
to hyperprior specification.

Sweeps over hyperprior scale factors on two configurations:
  - K=2/moderate/n=250 (safe baseline)
  - K=4/moderate/n=100 (harder: more subgroups, fewer experts)
Each config × 3 scales × 100 MC = 600 runs total.

Output: results/exp001_hyperprior_sensitivity.csv
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

from exp001_hb_synthetic import (
    REGIMES, SUBGROUP_LABELS, generate_data, get_counts,
    estimate_cm_mle, estimate_cm_hb_gibbs,
    build_mixing_matrix, invert_mixing, invert_mixing_safe, diff_in_means,
)

# Default Gamma hyperprior on alpha_0: Gamma(a0=3, b0=1), mean=3, var=3.
# Scaling: Gamma(a0/s, b0/s) keeps mean fixed, scales variance by s.
#   s=0.5 → Gamma(6, 2), var=1.5  (tighter, more shrinkage)
#   s=1.0 → Gamma(3, 1), var=3    (baseline)
#   s=2.0 → Gamma(1.5, 0.5), var=6 (wider, closer to MLE)
DEFAULT_HP_A0 = 3.0
DEFAULT_HP_B0 = 1.0
HYPERPRIOR_SCALES = [0.5, 1.0, 2.0]

CONFIGS = [
    {'k': 2, 'regime': 'moderate', 'n_expert': 250, 'label': 'K2_moderate_n250'},
    {'k': 4, 'regime': 'moderate', 'n_expert': 100, 'label': 'K4_moderate_n100'},
]


def run_one_sensitivity(regime_params, n_expert, k, seed, hp_a0, hp_b0,
                        use_safe_invert=False):
    data = generate_data(regime_params, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]
    tau_z0, tau_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)

    C_mle = estimate_cm_mle(counts, labels)
    C_hb, _ = estimate_cm_hb_gibbs(counts, labels, seed=seed, hp_a0=hp_a0, hp_b0=hp_b0)

    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    rows = []
    for s in labels:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        _invert = invert_mixing_safe if use_safe_invert else invert_mixing
        tml, sml = _invert(M_mle[s], tau_obs, se_obs)
        thb, shb = _invert(M_hb[s], tau_obs, se_obs)

        base = {'subgroup': s, 'mc_run': seed, 'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1}
        rows.append({**base, 'method': 'mle_laplace',
                     'tau_hat_z0': tml[0], 'tau_hat_z1': tml[1],
                     'se_z0': sml[0], 'se_z1': sml[1]})
        rows.append({**base, 'method': 'hb_dirichlet',
                     'tau_hat_z0': thb[0], 'tau_hat_z1': thb[1],
                     'se_z0': shb[0], 'se_z1': shb[1]})
    return rows


def aggregate(df):
    rows = []
    for (config, scale, method, subgroup), grp in df.groupby(
            ['config', 'hyperprior_scale', 'method', 'subgroup']):
        row = {'config': config, 'hyperprior_scale': scale,
               'method': method, 'subgroup': subgroup}
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'].values - grp[f'tau_true_{z}'].values
            bias = err.mean()
            rmse = np.sqrt((err ** 2).mean())
            ci_lo = grp[f'tau_hat_{z}'].values - 1.96 * grp[f'se_{z}'].values
            ci_hi = grp[f'tau_hat_{z}'].values + 1.96 * grp[f'se_{z}'].values
            cov = ((ci_lo <= grp[f'tau_true_{z}'].values) &
                   (grp[f'tau_true_{z}'].values <= ci_hi)).mean()
            row[f'bias_{z}'] = bias
            row[f'rmse_{z}'] = rmse
            row[f'coverage_{z}'] = cov
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='exp-001 hyperprior sensitivity')
    parser.add_argument('--n-mc', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else args.n_mc

    print(f"Hyperprior sensitivity: {len(CONFIGS)} configs × "
          f"{len(HYPERPRIOR_SCALES)} scales × {n_mc} MC = "
          f"{len(CONFIGS) * len(HYPERPRIOR_SCALES) * n_mc} runs")
    print(f"Scales: {HYPERPRIOR_SCALES}")
    print(f"Default hyperprior: Gamma({DEFAULT_HP_A0}, {DEFAULT_HP_B0})\n")

    all_rows = []
    total = len(CONFIGS) * len(HYPERPRIOR_SCALES) * n_mc
    count = 0
    t0 = time.time()

    for cfg in CONFIGS:
        k, regime_name = cfg['k'], cfg['regime']
        n_expert, label = cfg['n_expert'], cfg['label']
        regime_params = REGIMES[k][regime_name]
        use_safe = (k >= 4)

        print(f"\n── Config: {label} (K={k}, {regime_name}, n_expert={n_expert})"
              f"{' [safe invert]' if use_safe else ''} ──")

        for scale in HYPERPRIOR_SCALES:
            hp_a0 = DEFAULT_HP_A0 / scale
            hp_b0 = DEFAULT_HP_B0 / scale
            print(f"  scale={scale:.1f} → Gamma({hp_a0:.1f}, {hp_b0:.1f})")

            for mc in range(n_mc):
                rows = run_one_sensitivity(regime_params, n_expert, k,
                                           seed=mc, hp_a0=hp_a0, hp_b0=hp_b0,
                                           use_safe_invert=use_safe)
                for r in rows:
                    r['hyperprior_scale'] = scale
                    r['config'] = label
                all_rows.extend(rows)
                count += 1
                if count % max(1, total // 10) == 0:
                    el = time.time() - t0
                    eta = el / count * (total - count)
                    print(f"    [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate(df_raw)

    os.makedirs('results', exist_ok=True)
    out_cols = ['config', 'hyperprior_scale', 'method', 'subgroup',
                'bias_z0', 'rmse_z0', 'coverage_z0',
                'bias_z1', 'rmse_z1', 'coverage_z1']
    df_agg[out_cols].to_csv('results/exp001_hyperprior_sensitivity.csv', index=False)

    print(f"\nDone: {time.time() - t0:.1f}s ({count} runs)\n")

    for cfg in CONFIGS:
        label = cfg['label']
        sub_agg = df_agg[df_agg['config'] == label]
        print(f"\n{'═' * 80}")
        print(f"  {label}")
        print(f"{'═' * 80}")
        print("hyperprior_scale | method       | subgroup | RMSE_z0 | RMSE_z1 | Cov_z0 | Cov_z1")
        print("-" * 85)
        for _, row in sub_agg.sort_values(['hyperprior_scale', 'method', 'subgroup']).iterrows():
            print(f"  {row['hyperprior_scale']:4.1f}  {row['subgroup']:<8s} | "
                  f"{row['method']:<12s} | "
                  f"{row['rmse_z0']:.4f}  | {row['rmse_z1']:.4f}  | "
                  f"{row['coverage_z0']:.3f}  | {row['coverage_z1']:.3f}")

        print(f"\n  ── {label}: averaged over subgroups ──")
        print(f"  {'scale':>5s} | {'HB RMSE_z0':>10s} | {'HB RMSE_z1':>10s} | "
              f"{'HB Cov_z0':>9s} | {'HB Cov_z1':>9s} | {'MLE RMSE_z0 (ref)':>18s}")
        print("  " + "-" * 78)
        for scale in HYPERPRIOR_SCALES:
            sub = sub_agg[sub_agg['hyperprior_scale'] == scale]
            hb = sub[sub['method'] == 'hb_dirichlet']
            mle = sub[sub['method'] == 'mle_laplace']
            print(f"  {scale:5.1f} | {hb['rmse_z0'].mean():10.4f} | {hb['rmse_z1'].mean():10.4f} | "
                  f"{hb['coverage_z0'].mean():9.3f} | {hb['coverage_z1'].mean():9.3f} | "
                  f"{mle['rmse_z0'].mean():18.4f}")

    print(f"\nSaved: results/exp001_hyperprior_sensitivity.csv ({len(df_agg)} rows)")


if __name__ == '__main__':
    main()
