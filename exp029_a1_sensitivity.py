#!/usr/bin/env python3
"""exp029_a1_sensitivity.py — A1 violation sensitivity analysis (exp-029)

Quantifies EC-HTE degradation when Assumption A1 (outcome-independent
misclassification) is violated. Sensitivity parameter gamma controls
the strength of Y-dependent misclassification.

DGP: K=2, pi=[0.05,0.25], tau(0)=0, tau(1)=0.3, N=5000, n_expert=250
Violation: P(Z_hat != Z | Z,S,Y) = pi_s + gamma * (Y - mu_Y) / sigma_Y
gamma in {0, 0.05, 0.10, 0.15, 0.20}

Output: results/exp_a1_sensitivity.csv, results/exp_a1_sensitivity_summary.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N = 5000
K = 2
N_EXPERT = 250
PI = [0.05, 0.25]
TAU_TRUE = {0: 0.0, 1: 0.3}
GAMMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_MC = 100


def generate_data(seed, gamma):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=[0.5, 0.5])
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    tau = np.array([TAU_TRUE[z] for z in Z_true])
    baseline = 1.0 + 0.5 * Z_true
    Y = baseline + tau * T + rng.normal(0, 1, N)

    mu_Y = Y.mean()
    sigma_Y = Y.std()

    Z_hat = Z_true.copy()
    clip_count = 0
    for s in range(K):
        mask_s = S == s
        n_s = mask_s.sum()
        p_misclass = PI[s] + gamma * (Y[mask_s] - mu_Y) / sigma_Y
        raw_count = n_s
        p_misclass = np.clip(p_misclass, 0.001, 0.499)
        clip_count += (n_s - ((p_misclass > 0.001) & (p_misclass < 0.499)).sum())
        flip = rng.binomial(1, p_misclass)
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    expert_idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


def compute_confusion_matrix(z_true, z_hat):
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mz = z_true == z
        C[z, 0] = (z_hat[mz] == 0).sum() + 1  # Laplace smoothing
        C[z, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


def dim_estimate(Y, T, Z_labels, S, s):
    """DIM estimator for subgroup s, stratified by Z_labels."""
    mask_s = S == s
    results = {}
    for z in [0, 1]:
        mask_sz = mask_s & (Z_labels == z)
        mask_sz_t1 = mask_sz & (T == 1)
        mask_sz_t0 = mask_sz & (T == 0)
        n1 = mask_sz_t1.sum()
        n0 = mask_sz_t0.sum()
        if n1 < 2 or n0 < 2:
            results[z] = (np.nan, np.nan)
            continue
        y1 = Y[mask_sz_t1].mean()
        y0 = Y[mask_sz_t0].mean()
        tau_hat = y1 - y0
        se = np.sqrt(Y[mask_sz_t1].var(ddof=1) / n1 + Y[mask_sz_t0].var(ddof=1) / n0)
        results[z] = (tau_hat, se)
    return results


def invert_mixing(C, tau_obs_vec, se_obs_vec, cond_threshold=100):
    """Build mixing matrix from confusion matrix and invert."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        col_sum = C[0, zh] + C[1, zh]
        if col_sum > 0:
            M[zh, 0] = C[0, zh] / col_sum
            M[zh, 1] = C[1, zh] / col_sum
        else:
            M[zh, :] = 0.5

    cond = np.linalg.cond(M)
    if cond > cond_threshold or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs_vec.copy(), se_obs_vec.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs_vec
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs_vec ** 2) @ Mi.T), 0))
    return tau_c, se_c


def hb_confusion_matrix(z_true_list, z_hat_list, alpha_prior=3.0, beta_prior=1.0):
    """Hierarchical Bayes CM estimation via empirical Bayes shrinkage."""
    K_sub = len(z_true_list)
    raw_cms = []
    ns = []
    for i in range(K_sub):
        C = compute_confusion_matrix(z_true_list[i], z_hat_list[i])
        raw_cms.append(C)
        ns.append(len(z_true_list[i]))

    global_z_true = np.concatenate(z_true_list)
    global_z_hat = np.concatenate(z_hat_list)
    C_global = compute_confusion_matrix(global_z_true, global_z_hat)

    shrunk_cms = []
    for i in range(K_sub):
        n_i = ns[i]
        w = n_i / (n_i + alpha_prior)
        C_shrunk = w * raw_cms[i] + (1 - w) * C_global
        C_shrunk = C_shrunk / C_shrunk.sum(axis=1, keepdims=True)
        shrunk_cms.append(C_shrunk)

    return shrunk_cms


def run_one_mc(seed, gamma):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed, gamma)

    # Expert data for CM estimation
    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    # Per-subgroup MLE CMs
    cm_mle = [compute_confusion_matrix(z_true_expert[s], z_hat_expert[s]) for s in range(K)]

    # HB CMs
    cm_hb = hb_confusion_matrix(z_true_expert, z_hat_expert)

    # Global CM
    cm_global = compute_confusion_matrix(
        Z_true[expert_mask], Z_hat[expert_mask]
    )

    rows = []
    for s in range(K):
        # DIM with Z_hat (naive estimates)
        dim_zhat = dim_estimate(Y, T, Z_hat, S, s)

        tau_obs = np.array([dim_zhat[0][0], dim_zhat[1][0]])
        se_obs = np.array([dim_zhat[0][1], dim_zhat[1][1]])

        if np.any(np.isnan(tau_obs)) or np.any(np.isnan(se_obs)):
            continue

        # Methods
        tau_mle, se_mle = invert_mixing(cm_mle[s], tau_obs, se_obs)
        tau_hb, se_hb = invert_mixing(cm_hb[s], tau_obs, se_obs)
        tau_gl, se_gl = invert_mixing(cm_global, tau_obs, se_obs)

        methods = [
            ('Naive', tau_obs, se_obs),
            ('Global', tau_gl, se_gl),
            ('Strat MLE', tau_mle, se_mle),
            ('HB EC-HTE', tau_hb, se_hb),
        ]

        for method_name, tau_hat_vec, se_hat_vec in methods:
            for zi, z in enumerate([0, 1]):
                th = tau_hat_vec[zi]
                se = se_hat_vec[zi]
                ci_lo = th - 1.96 * se
                ci_hi = th + 1.96 * se
                covers = int(ci_lo <= TAU_TRUE[z] <= ci_hi)
                rows.append({
                    'gamma': gamma,
                    'mc_seed': seed,
                    'subgroup': s,
                    'z_level': z,
                    'method': method_name,
                    'tau_hat': th,
                    'tau_true': TAU_TRUE[z],
                    'bias': th - TAU_TRUE[z],
                    'se': se,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'covers': covers,
                })

    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    all_rows = []
    for gamma in GAMMAS:
        print(f"gamma={gamma:.2f} ...", end=' ', flush=True)
        gt = time.time()
        for mc in range(N_MC):
            seed = 42 + mc
            rows = run_one_mc(seed, gamma)
            all_rows.extend(rows)
        print(f"done ({time.time()-gt:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_a1_sensitivity.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_a1_sensitivity.csv")

    # Summary
    summary = df.groupby(['gamma', 'method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
        n=('covers', 'count'),
    ).reset_index()

    summary.to_csv('results/exp_a1_sensitivity_summary.csv', index=False)
    print(f"Wrote summary to results/exp_a1_sensitivity_summary.csv")

    # Print table
    print("\n" + "="*80)
    print("A1 SENSITIVITY ANALYSIS — AGGREGATE RESULTS")
    print("="*80)
    print(f"{'gamma':>6} {'Method':>12} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6} {'Mean SE':>8}")
    print("-"*56)
    for _, row in summary.iterrows():
        print(f"{row['gamma']:6.2f} {row['method']:>12} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f} {row['mean_se']:8.4f}")

    # Per-subgroup summary
    summary_sub = df.groupby(['gamma', 'method', 'subgroup']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
    ).reset_index()
    summary_sub.to_csv('results/exp_a1_sensitivity_subgroup.csv', index=False)

    # Degradation relative to gamma=0
    print("\n" + "="*80)
    print("DEGRADATION RELATIVE TO gamma=0")
    print("="*80)
    baseline = summary[summary['gamma'] == 0.0].set_index('method')
    for gamma in GAMMAS[1:]:
        print(f"\ngamma={gamma:.2f}:")
        current = summary[summary['gamma'] == gamma].set_index('method')
        for method in ['Naive', 'Global', 'Strat MLE', 'HB EC-HTE']:
            if method in current.index and method in baseline.index:
                b0 = baseline.loc[method, 'mean_abs_bias']
                b1 = current.loc[method, 'mean_abs_bias']
                c0 = baseline.loc[method, 'coverage']
                c1 = current.loc[method, 'coverage']
                print(f"  {method:>12}: |Bias| {b0:.4f}→{b1:.4f} "
                      f"({(b1-b0)/max(b0,1e-6)*100:+.1f}%), "
                      f"Cov {c0:.3f}→{c1:.3f} ({(c1-c0)*100:+.1f}pp)")

    # Coverage threshold
    print("\n" + "="*80)
    print("COVERAGE < 90% THRESHOLD")
    print("="*80)
    for method in ['Naive', 'Global', 'Strat MLE', 'HB EC-HTE']:
        method_data = summary[summary['method'] == method]
        below_90 = method_data[method_data['coverage'] < 0.90]
        if len(below_90) > 0:
            first_gamma = below_90['gamma'].min()
            print(f"  {method:>12}: first < 90% at gamma={first_gamma:.2f}")
        else:
            print(f"  {method:>12}: always >= 90%")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
