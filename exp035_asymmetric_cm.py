#!/usr/bin/env python3
"""exp035_asymmetric_cm.py — Asymmetric CM end-to-end validation

Tests EC-HTE under realistic asymmetric confusion matrices (FPR != FNR)
with ratios up to 7:1, simulating GPT-4.1 on Civil Comments.

Setup:
  - K=2, N=5000, n_expert=250
  - s0: FPR=0.05, FNR=0.02 (low misclassification, roughly symmetric)
  - s1: FPR=0.38, FNR=0.054 (high misclassification, 7:1 FPR/FNR ratio)
  - Remark 2 general asymmetric formula (full 2x2 CM inversion, no FPR=FNR assumption)

Methods:
  1. Oracle: true Z labels
  2. Naive: Z_hat uncorrected
  3. Global (symmetric): uses symmetric pi = (FPR+FNR)/2 for CM construction
  4. Global (asymmetric): uses full 2x2 CM with separate FPR and FNR
  5. EC-HTE symmetric: subgroup-stratified with symmetric pi assumption
  6. EC-HTE asymmetric: subgroup-stratified with full 2x2 CM (correct approach)

Output:
  results/exp_asymmetric_cm.csv
  results/exp_asymmetric_cm_summary.csv
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
TAU_Z = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 100

# Asymmetric CMs per subgroup: [FPR, FNR]
# s0: low error, nearly symmetric
# s1: high FPR, low FNR (simulates GPT-4.1 on Civil Comments)
CM_PARAMS = {
    0: {'FPR': 0.05, 'FNR': 0.02},
    1: {'FPR': 0.38, 'FNR': 0.054},
}


def generate_data(seed):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng.binomial(1, 0.5, N)
    T = rng.binomial(1, 0.5, N)

    # Asymmetric misclassification: FPR and FNR applied separately
    Z_hat = Z_true.copy()
    for s in range(K):
        fpr = CM_PARAMS[s]['FPR']
        fnr = CM_PARAMS[s]['FNR']
        mask_s = S == s
        for i in np.where(mask_s)[0]:
            if Z_true[i] == 0:
                # True negative: flip with probability FPR
                if rng.random() < fpr:
                    Z_hat[i] = 1
            else:
                # True positive: flip with probability FNR
                if rng.random() < fnr:
                    Z_hat[i] = 0

    tau = np.array([TAU_Z[z] for z in Z_true])
    Y = 1.0 + 0.5 * Z_true + tau * T + rng.normal(0, 1, N)

    expert_idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


def estimate_asymmetric_cm(z_true, z_hat):
    """Estimate full 2x2 CM with Laplace smoothing: C[z_true, z_hat]."""
    C = np.zeros((2, 2))
    for zt in [0, 1]:
        mz = z_true == zt
        C[zt, 0] = (z_hat[mz] == 0).sum() + 1
        C[zt, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


def estimate_symmetric_cm(z_true, z_hat):
    """Estimate CM assuming FPR=FNR (symmetric pi)."""
    pi = (z_true != z_hat).mean()
    pi = max(min(pi, 0.499), 0.001)  # clip
    C = np.array([[1 - pi, pi], [pi, 1 - pi]])
    return C


def hb_shrink_cms(raw_cms, ns, C_global, alpha_prior=3.0):
    shrunk = []
    for i in range(len(raw_cms)):
        w = ns[i] / (ns[i] + alpha_prior)
        C_s = w * raw_cms[i] + (1 - w) * C_global
        C_s = C_s / C_s.sum(axis=1, keepdims=True)
        shrunk.append(C_s)
    return shrunk


def invert_cm(C, tau_obs_vec, se_obs_vec, cond_threshold=100):
    """Invert confusion matrix (column-stochastic mixing matrix) to correct estimates."""
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


def dim_estimate(Y, T, Z_labels, S, s):
    mask_s = S == s
    results = {}
    for z in [0, 1]:
        mask_sz = mask_s & (Z_labels == z)
        idx_t1 = np.where(mask_sz & (T == 1))[0]
        idx_t0 = np.where(mask_sz & (T == 0))[0]
        n1, n0 = len(idx_t1), len(idx_t0)
        if n1 < 2 or n0 < 2:
            results[z] = (np.nan, np.nan)
            continue
        mu1, mu0 = Y[idx_t1].mean(), Y[idx_t0].mean()
        tau_hat = mu1 - mu0
        se = np.sqrt(Y[idx_t1].var(ddof=1) / n1 + Y[idx_t0].var(ddof=1) / n0)
        results[z] = (tau_hat, se)
    return results


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    # Asymmetric CMs (correct)
    cm_asym_per_sg = [estimate_asymmetric_cm(z_true_expert[s], z_hat_expert[s]) for s in range(K)]
    cm_asym_global = estimate_asymmetric_cm(Z_true[expert_mask], Z_hat[expert_mask])

    # Symmetric CMs (assumption FPR=FNR)
    cm_sym_per_sg = [estimate_symmetric_cm(z_true_expert[s], z_hat_expert[s]) for s in range(K)]
    cm_sym_global = estimate_symmetric_cm(Z_true[expert_mask], Z_hat[expert_mask])

    # HB shrinkage — asymmetric
    ns = [len(z_true_expert[s]) for s in range(K)]
    cm_hb_asym = hb_shrink_cms(cm_asym_per_sg, ns, cm_asym_global)
    cm_hb_sym = hb_shrink_cms(cm_sym_per_sg, ns, cm_sym_global)

    rows = []
    for s in range(K):
        oracle = dim_estimate(Y, T, Z_true, S, s)
        dim_zhat = dim_estimate(Y, T, Z_hat, S, s)

        tau_obs = np.array([dim_zhat[0][0], dim_zhat[1][0]])
        se_obs = np.array([dim_zhat[0][1], dim_zhat[1][1]])

        if np.any(np.isnan(tau_obs)) or np.any(np.isnan(se_obs)):
            continue

        o0, o1 = oracle[0], oracle[1]
        if np.isnan(o0[0]) or np.isnan(o1[0]):
            continue
        tau_oracle = np.array([o0[0], o1[0]])
        se_oracle = np.array([o0[1], o1[1]])

        # All correction methods
        tau_gl_asym, se_gl_asym = invert_cm(cm_asym_global, tau_obs, se_obs)
        tau_gl_sym, se_gl_sym = invert_cm(cm_sym_global, tau_obs, se_obs)
        tau_hb_asym, se_hb_asym = invert_cm(cm_hb_asym[s], tau_obs, se_obs)
        tau_hb_sym, se_hb_sym = invert_cm(cm_hb_sym[s], tau_obs, se_obs)

        methods = [
            ('Oracle', tau_oracle, se_oracle),
            ('Naive', tau_obs, se_obs),
            ('Global (symmetric)', tau_gl_sym, se_gl_sym),
            ('Global (asymmetric)', tau_gl_asym, se_gl_asym),
            ('EC-HTE (symmetric)', tau_hb_sym, se_hb_sym),
            ('EC-HTE (asymmetric)', tau_hb_asym, se_hb_asym),
        ]

        for method_name, tau_hat_vec, se_hat_vec in methods:
            for zi, z in enumerate([0, 1]):
                th = tau_hat_vec[zi]
                se = se_hat_vec[zi]
                ci_lo = th - 1.96 * se
                ci_hi = th + 1.96 * se
                true_tau = TAU_Z[z]
                covers = int(ci_lo <= true_tau <= ci_hi)

                rows.append({
                    'mc_seed': seed,
                    'subgroup': s,
                    'z_level': z,
                    'method': method_name,
                    'tau_hat': th,
                    'tau_true': true_tau,
                    'bias': th - true_tau,
                    'se': se,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'covers': covers,
                    'fpr_s': CM_PARAMS[s]['FPR'],
                    'fnr_s': CM_PARAMS[s]['FNR'],
                    'fpr_fnr_ratio': CM_PARAMS[s]['FPR'] / max(CM_PARAMS[s]['FNR'], 1e-10),
                })
    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"Asymmetric CM Validation: K={K}, N={N}, n_expert={N_EXPERT}")
    for s in range(K):
        p = CM_PARAMS[s]
        print(f"  s{s}: FPR={p['FPR']}, FNR={p['FNR']}, ratio={p['FPR']/p['FNR']:.1f}:1")
    print(f"N_MC={N_MC}")

    all_rows = []
    for mc in range(N_MC):
        if mc % 20 == 0:
            print(f"  MC {mc}/{N_MC} ...", flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_asymmetric_cm.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_asymmetric_cm.csv")

    # Aggregate summary
    summary = df.groupby(['method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()
    summary.to_csv('results/exp_asymmetric_cm_summary.csv', index=False)

    # Per-subgroup summary
    sg_summary = df.groupby(['method', 'subgroup']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
    ).reset_index()
    sg_summary.to_csv('results/exp_asymmetric_cm_subgroup.csv', index=False)

    print("\n" + "=" * 80)
    print("ASYMMETRIC CM — AGGREGATE")
    print("=" * 80)
    print(f"{'Method':>25} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 55)
    for _, row in summary.sort_values('mean_abs_bias').iterrows():
        print(f"{row['method']:>25} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f}")

    print("\n" + "=" * 80)
    print("ASYMMETRIC CM — PER-SUBGROUP")
    print("=" * 80)
    print(f"{'Method':>25} {'SG':>3} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 55)
    for _, row in sg_summary.sort_values(['subgroup', 'method']).iterrows():
        print(f"{row['method']:>25} {row['subgroup']:3.0f} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f}")

    # Symmetric vs Asymmetric comparison
    print("\n" + "=" * 80)
    print("SYMMETRIC vs ASYMMETRIC FORMULA COMPARISON")
    print("=" * 80)
    for level in ['Global', 'EC-HTE']:
        sym = summary[summary['method'] == f'{level} (symmetric)'].iloc[0]
        asym = summary[summary['method'] == f'{level} (asymmetric)'].iloc[0]
        print(f"\n{level}:")
        print(f"  Symmetric:  |Bias|={sym['mean_abs_bias']:.4f}, RMSE={sym['rmse']:.4f}, "
              f"Cov={sym['coverage']:.3f}")
        print(f"  Asymmetric: |Bias|={asym['mean_abs_bias']:.4f}, RMSE={asym['rmse']:.4f}, "
              f"Cov={asym['coverage']:.3f}")
        delta_cov = asym['coverage'] - sym['coverage']
        delta_rmse = asym['rmse'] - sym['rmse']
        print(f"  Delta: RMSE {delta_rmse:+.4f}, Coverage {delta_cov:+.3f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
