#!/usr/bin/env python3
"""exp030_k8_subgroups.py — K=8 subgroup experiment (W11)

Validates EC-HTE HB partial pooling at K=8 (per-subgroup ~31 expert labels).
Pi uniformly spread in [0.05, 0.35] across 8 subgroups.
CATE heterogeneity is Z-dependent (tau(Z=0)=0, tau(Z=1)=0.3), same for all subgroups.

Output: results/exp_k8_subgroups.csv, results/exp_k8_subgroups_summary.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

N = 5000
K = 8
N_EXPERT = 250
PI = [0.05 + 0.30 * i / (K - 1) for i in range(K)]  # [0.05, ..., 0.35]
TAU_Z = {0: 0.0, 1: 0.3}  # CATE depends on Z (binary), not subgroup
SUB_WEIGHTS = [1.0 / K] * K
N_MC = 100


def generate_data(seed):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, PI[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    tau = np.array([TAU_Z[z] for z in Z_true])
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


def dim_estimate(Y, T, Z_labels, S, s):
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


def invert_cm(C, tau_obs_vec, se_obs_vec, cond_threshold=100):
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


def hb_confusion_matrices(z_true_list, z_hat_list, alpha_prior=3.0):
    raw_cms = []
    ns = []
    for i in range(len(z_true_list)):
        C = compute_confusion_matrix(z_true_list[i], z_hat_list[i])
        raw_cms.append(C)
        ns.append(len(z_true_list[i]))

    global_z_true = np.concatenate(z_true_list)
    global_z_hat = np.concatenate(z_hat_list)
    C_global = compute_confusion_matrix(global_z_true, global_z_hat)

    shrunk = []
    for i in range(len(z_true_list)):
        w = ns[i] / (ns[i] + alpha_prior)
        C_s = w * raw_cms[i] + (1 - w) * C_global
        C_s = C_s / C_s.sum(axis=1, keepdims=True)
        shrunk.append(C_s)
    return shrunk


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    # Oracle DIM with true Z
    oracle_results = {}
    for s in range(K):
        oracle_results[s] = dim_estimate(Y, T, Z_true, S, s)

    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    cm_mle = [compute_confusion_matrix(z_true_expert[s], z_hat_expert[s]) for s in range(K)]
    cm_hb = hb_confusion_matrices(z_true_expert, z_hat_expert)
    cm_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])

    rows = []
    for s in range(K):
        dim_zhat = dim_estimate(Y, T, Z_hat, S, s)
        tau_obs = np.array([dim_zhat[0][0], dim_zhat[1][0]])
        se_obs = np.array([dim_zhat[0][1], dim_zhat[1][1]])

        if np.any(np.isnan(tau_obs)) or np.any(np.isnan(se_obs)):
            continue

        # Oracle
        o0, o1 = oracle_results[s][0], oracle_results[s][1]
        if np.isnan(o0[0]) or np.isnan(o1[0]):
            continue
        tau_oracle = np.array([o0[0], o1[0]])
        se_oracle = np.array([o0[1], o1[1]])

        tau_mle, se_mle = invert_cm(cm_mle[s], tau_obs, se_obs)
        tau_hb, se_hb = invert_cm(cm_hb[s], tau_obs, se_obs)
        tau_gl, se_gl = invert_cm(cm_global, tau_obs, se_obs)

        methods = [
            ('Oracle', tau_oracle, se_oracle),
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
                    'pi_s': PI[s],
                })
    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"K={K}, N={N}, n_expert={N_EXPERT}")
    print(f"PI = {[f'{p:.3f}' for p in PI]}")
    print(f"TAU(Z=0)={TAU_Z[0]}, TAU(Z=1)={TAU_Z[1]}")
    print(f"Per-subgroup expert labels ≈ {N_EXPERT / K:.0f}")

    all_rows = []
    for mc in range(N_MC):
        if mc % 20 == 0:
            print(f"  MC {mc}/{N_MC} ...", flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_k8_subgroups.csv', index=False)
    print(f"\nWrote {len(df)} rows")

    summary = df.groupby(['method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()

    summary.to_csv('results/exp_k8_subgroups_summary.csv', index=False)

    print("\n" + "=" * 60)
    print(f"K=8 RESULTS (aggregate over subgroups and z-levels)")
    print("=" * 60)
    print(f"{'Method':>12} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 40)
    for _, row in summary.iterrows():
        print(f"{row['method']:>12} {row['mean_abs_bias']:8.4f} {row['rmse']:8.4f} {row['coverage']:6.3f}")

    # Per-subgroup coverage
    sub_summary = df.groupby(['method', 'subgroup']).agg(
        coverage=('covers', 'mean'),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        pi_s=('pi_s', 'first'),
    ).reset_index()

    print("\n" + "=" * 60)
    print("PER-SUBGROUP COVERAGE (s0=low-pi ... s7=high-pi)")
    print("=" * 60)
    for method in ['Oracle', 'Naive', 'Global', 'Strat MLE', 'HB EC-HTE']:
        m_data = sub_summary[sub_summary['method'] == method].sort_values('subgroup')
        covs = [f"{r['coverage']:.2f}" for _, r in m_data.iterrows()]
        print(f"  {method:>12}: {' '.join(covs)}")

    print("\n" + "=" * 60)
    print("PER-SUBGROUP RMSE")
    print("=" * 60)
    for method in ['Oracle', 'Naive', 'Global', 'Strat MLE', 'HB EC-HTE']:
        m_data = sub_summary[sub_summary['method'] == method].sort_values('subgroup')
        rmses = [f"{r['rmse']:.3f}" for _, r in m_data.iterrows()]
        print(f"  {method:>12}: {' '.join(rmses)}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
