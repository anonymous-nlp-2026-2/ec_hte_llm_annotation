#!/usr/bin/env python3
"""exp017_rlearner.py - R-learner EC-HTE compatibility (exp-017)

Validates EC-HTE correction with R-learner (Nie & Wager 2021) as CATE estimator.
Four methods: oracle (Z_true), naive (Z_hat), global (global M^-1),
ec_hte (per-subgroup M_s^-1).

Config: K=2, extreme (pi=[0.05, 0.20]), n_expert=500, 50 MC seeds.
DGP matches exp-011 for direct comparability.
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

N = 5000
K = 2
REGIME = 'extreme'
N_EXPERT = 500
TAU_TRUE = {0: 0.5, 1: 1.5}
MISCLASS_RATES = [0.05, 0.20]


def generate_data(seed):
    rng = np.random.RandomState(seed)
    S = rng.randint(0, K, N)
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, MISCLASS_RATES[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    tau = np.where(Z_true == 1, TAU_TRUE[1], TAU_TRUE[0])
    baseline = 1.0 + 0.5 * Z_true
    Y = baseline + tau * T + rng.normal(0, 1, N)

    expert_idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


def compute_confusion_matrix(z_true, z_hat):
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mz = z_true == z
        C[z, 0] = (z_hat[mz] == 0).sum() + 1
        C[z, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        if cs > 0:
            M[zh, 0] = C[0, zh] / cs
            M[zh, 1] = C[1, zh] / cs
        else:
            M[zh, :] = 0.5
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    cond = np.linalg.cond(M)
    if cond > cond_threshold:
        return tau_obs.copy(), se_obs.copy()
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


def rlearner_cate(Y, T, Z_feat, S):
    """R-learner CATE estimation with cross-fitted nuisance.

    Steps (Nie & Wager 2021):
    1. Cross-fit m(X) = E[Y|X] via RandomForest
    2. e(X) = 0.5 (known RCT)
    3. Pseudo-outcome: tilde_Y = (Y - m_hat) / (T - 0.5)
    4. Group-average CATE per (z, s)

    Returns {(z, s): (cate_estimate, standard_error)}.
    """
    n = len(Y)
    X = np.column_stack([Z_feat, S]).astype(float)

    m_hat = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, random_state=42)
        rf.fit(X[train_idx], Y[train_idx])
        m_hat[test_idx] = rf.predict(X[test_idx])

    T_res = T - 0.5
    Y_res = Y - m_hat
    pseudo = Y_res / T_res

    results = {}
    for s in range(K):
        for z in [0, 1]:
            mask = (Z_feat == z) & (S == s)
            n_group = mask.sum()
            if n_group > 1:
                po = pseudo[mask]
                results[(z, s)] = (po.mean(), po.std(ddof=1) / np.sqrt(n_group))
            else:
                results[(z, s)] = (np.nan, np.nan)

    return results


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    cate_hat = rlearner_cate(Y, T, Z_hat, S)
    cate_oracle = rlearner_cate(Y, T, Z_true, S)

    C_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])
    M_global = build_mixing_matrix(C_global)

    M_sub = {}
    for s in range(K):
        mask_s = expert_mask & (S == s)
        if mask_s.sum() >= 4:
            C_s = compute_confusion_matrix(Z_true[mask_s], Z_hat[mask_s])
        else:
            C_s = np.array([[0.5, 0.5], [0.5, 0.5]])
        M_sub[s] = build_mixing_matrix(C_s)

    rows = []
    for s in range(K):
        tau_obs = np.array([cate_hat[(0, s)][0], cate_hat[(1, s)][0]])
        se_obs = np.array([cate_hat[(0, s)][1], cate_hat[(1, s)][1]])

        tau_oracle = np.array([cate_oracle[(0, s)][0], cate_oracle[(1, s)][0]])
        se_oracle = np.array([cate_oracle[(0, s)][1], cate_oracle[(1, s)][1]])

        tau_naive, se_naive = tau_obs.copy(), se_obs.copy()
        tau_global, se_global = invert_mixing_safe(M_global, tau_obs, se_obs)
        tau_ec, se_ec = invert_mixing_safe(M_sub[s], tau_obs, se_obs)

        for z in [0, 1]:
            tau_true = TAU_TRUE[z]
            for method, th, se in [
                ('oracle_rlearner', tau_oracle[z], se_oracle[z]),
                ('naive_rlearner', tau_naive[z], se_naive[z]),
                ('global_rlearner', tau_global[z], se_global[z]),
                ('ec_hte_rlearner', tau_ec[z], se_ec[z]),
            ]:
                if np.isnan(se) or se <= 0:
                    coverage = np.nan
                else:
                    ci_lo = th - 1.96 * se
                    ci_hi = th + 1.96 * se
                    coverage = int(ci_lo <= tau_true <= ci_hi)
                rows.append({
                    'seed': seed, 'subgroup': s, 'z': z, 'method': method,
                    'bias': th - tau_true,
                    'rmse': abs(th - tau_true),
                    'coverage': coverage,
                    'cate_est': th, 'cate_true': tau_true,
                })

    return rows


def write_analysis(df):
    """Generate analysis markdown comparing R-learner to exp-011 CausalForest."""
    lines = [
        '# exp-017: R-learner EC-HTE Compatibility',
        '',
        '## Setup',
        f'- N={N}, K={K}, regime={REGIME}, n_expert={N_EXPERT}',
        f'- Misclassification rates: S0 pi={MISCLASS_RATES[0]}, S1 pi={MISCLASS_RATES[1]}',
        f'- True CATE: tau(Z=0)={TAU_TRUE[0]}, tau(Z=1)={TAU_TRUE[1]}',
        f'- MC seeds: {df["seed"].nunique()}',
        f'- CATE estimator: R-learner (Nie & Wager 2021)',
        '',
        '## Per-method summary',
        '',
        '| Method | Avg |Bias| | Avg RMSE | Avg Coverage |',
        '|--------|-----------|----------|--------------|',
    ]

    methods = ['oracle_rlearner', 'naive_rlearner', 'global_rlearner', 'ec_hte_rlearner']
    summary = {}
    for method in methods:
        m = df[df['method'] == method]
        grp = m.groupby(['subgroup', 'z'])['bias']
        avg_abs_bias = grp.mean().abs().mean()
        avg_rmse = np.sqrt(grp.apply(lambda x: (x ** 2).mean())).mean()
        avg_cov = m['coverage'].mean()
        summary[method] = {'abs_bias': avg_abs_bias, 'rmse': avg_rmse, 'coverage': avg_cov}
        lines.append(f'| {method} | {avg_abs_bias:.4f} | {avg_rmse:.4f} | {avg_cov:.2f} |')

    lines += [
        '',
        '## Per-subgroup detail',
        '',
        '| Method | Sub | Z | Bias | RMSE | Coverage |',
        '|--------|-----|---|------|------|----------|',
    ]

    for method in methods:
        for s in range(K):
            for z in [0, 1]:
                sub = df[(df['method'] == method) & (df['subgroup'] == s) & (df['z'] == z)]
                bias = sub['bias'].mean()
                rmse = np.sqrt((sub['bias'] ** 2).mean())
                cov = sub['coverage'].mean()
                lines.append(f'| {method} | {s} | {z} | {bias:+.4f} | {rmse:.4f} | {cov:.2f} |')

    # Compare with exp-011
    exp011_path = 'results/exp011_causal_forest.csv'
    if os.path.exists(exp011_path):
        df011 = pd.read_csv(exp011_path)
        df011 = df011[(df011['K'] == 2) & (df011['regime'] == 'extreme') & (df011['n_expert'] == 500)]
        lines += [
            '',
            '## Comparison with exp-011 (CausalForestDML)',
            '',
            '| Estimator | Method | Avg |Bias| | Avg RMSE | Avg Coverage |',
            '|-----------|--------|-----------|----------|--------------|',
        ]
        method_map = {
            'naive_forest': 'naive',
            'global_forest': 'global',
            'ec_hte_forest': 'ec_hte',
        }
        for m011 in ['naive_forest', 'global_forest', 'ec_hte_forest']:
            sub011 = df011[df011['method'] == m011]
            b011 = sub011['bias'].abs().mean()
            r011 = sub011['rmse'].mean()
            c011 = sub011['coverage'].mean()
            lines.append(f'| CausalForest | {m011} | {b011:.4f} | {r011:.4f} | {c011:.2f} |')

        rlearner_map = {
            'naive': 'naive_rlearner',
            'global': 'global_rlearner',
            'ec_hte': 'ec_hte_rlearner',
        }
        for tag, rm in rlearner_map.items():
            s = summary[rm]
            lines.append(f'| R-learner | {rm} | {s["abs_bias"]:.4f} | {s["rmse"]:.4f} | {s["coverage"]:.2f} |')

    lines += [
        '',
        '## Conclusion',
        '',
    ]

    ec_bias = summary['ec_hte_rlearner']['abs_bias']
    na_bias = summary['naive_rlearner']['abs_bias']
    gl_bias = summary['global_rlearner']['abs_bias']
    ec_best = ec_bias < na_bias and ec_bias < gl_bias

    if ec_best:
        lines.append(
            'EC-HTE correction is compatible with R-learner. '
            f'ec_hte_rlearner achieves the lowest avg |bias| ({ec_bias:.4f}) '
            f'vs naive ({na_bias:.4f}) and global ({gl_bias:.4f}), '
            'consistent with CausalForest results from exp-011. '
            'The correction generalizes across CATE estimators.'
        )
    else:
        lines.append(
            f'ec_hte_rlearner avg |bias|={ec_bias:.4f}, '
            f'naive={na_bias:.4f}, global={gl_bias:.4f}. '
            'Pattern differs from exp-011 CausalForest; further investigation needed.'
        )

    path = 'results/exp017_rlearner_analysis.md'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exp-017 R-learner EC-HTE')
    parser.add_argument('--n-mc', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else args.n_mc

    print(f'exp-017 R-learner EC-HTE | N={N} K={K} {REGIME} '
          f'n_expert={N_EXPERT} n_mc={n_mc}')
    print(f'True CATE: tau(Z=0)={TAU_TRUE[0]}, tau(Z=1)={TAU_TRUE[1]}')
    print(f'Misclass rates: {MISCLASS_RATES}')

    all_rows = []
    t0 = time.time()

    for mc in range(n_mc):
        try:
            rows = run_one_mc(seed=mc)
            all_rows.extend(rows)
        except Exception as e:
            print(f'  WARN: mc={mc} failed: {e}')
        if (mc + 1) % max(1, n_mc // 10) == 0:
            el = time.time() - t0
            eta = el / (mc + 1) * (n_mc - mc - 1)
            print(f'  [{mc+1}/{n_mc}] {el:.0f}s elapsed, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'\nDone: {elapsed:.1f}s ({n_mc} runs, {elapsed/n_mc:.1f}s/run)')

    df = pd.DataFrame(all_rows)

    os.makedirs('results', exist_ok=True)
    df.to_csv('results/exp017_rlearner_results.csv', index=False)
    print(f'Saved: results/exp017_rlearner_results.csv ({len(df)} rows)')

    # Print summary
    print('\n' + '=' * 80)
    print('exp-017 R-learner Results Summary')
    print('=' * 80)

    for method in ['oracle_rlearner', 'naive_rlearner',
                    'global_rlearner', 'ec_hte_rlearner']:
        m = df[df['method'] == method]
        grp = m.groupby(['subgroup', 'z'])['bias']
        avg_abs_bias = grp.mean().abs().mean()
        avg_rmse = np.sqrt(grp.apply(lambda x: (x ** 2).mean())).mean()
        avg_cov = m['coverage'].mean()
        print(f'  {method:>20s}: |bias|={avg_abs_bias:.4f}  '
              f'RMSE={avg_rmse:.4f}  cov={avg_cov:.2f}')

    for s in range(K):
        print(f'\n  Subgroup {s} (pi={MISCLASS_RATES[s]:.2f}):')
        for method in ['oracle_rlearner', 'naive_rlearner',
                        'global_rlearner', 'ec_hte_rlearner']:
            for z in [0, 1]:
                sub = df[(df['method'] == method)
                         & (df['subgroup'] == s) & (df['z'] == z)]
                bias = sub['bias'].mean()
                rmse = np.sqrt((sub['bias'] ** 2).mean())
                cov = sub['coverage'].mean()
                print(f'    {method:>20s} z={z}: '
                      f'bias={bias:+.4f}  rmse={rmse:.4f}  cov={cov:.2f}')

    write_analysis(df)
