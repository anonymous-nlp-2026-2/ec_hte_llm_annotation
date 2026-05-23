#!/usr/bin/env python3
"""exp033_k8_adaptive.py — K=8 adaptive shrinkage variants

Tests 3 method variants to reduce HB RMSE at K=8:
  V1: Adaptive alpha (alpha=10 for high-pi subgroups, alpha=3 for low-pi)
  V2: kappa-threshold fallback (fallback to Global CM when kappa>10)
  V3: V1 + V2 combined

Same DGP as exp-030: K=8, N=5000, n_expert=250, pi uniformly in [0.05, 0.35].

Output: results/exp_k8_adaptive.csv, results/exp_k8_adaptive_summary.csv
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
PI = [0.05 + 0.30 * i / (K - 1) for i in range(K)]
TAU_Z = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [1.0 / K] * K
N_MC = 50
PI_THRESHOLD = 0.20
ALPHA_LOW = 3.0
ALPHA_HIGH = 10.0
KAPPA_STRICT = 10.0


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


def hb_adaptive_alpha(z_true_list, z_hat_list):
    """V1: per-subgroup alpha based on pi threshold."""
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
        alpha = ALPHA_HIGH if PI[i] >= PI_THRESHOLD else ALPHA_LOW
        w = ns[i] / (ns[i] + alpha)
        C_s = w * raw_cms[i] + (1 - w) * C_global
        C_s = C_s / C_s.sum(axis=1, keepdims=True)
        shrunk.append(C_s)
    return shrunk


def apply_kappa_fallback(cm_list, cm_global, kappa_threshold=KAPPA_STRICT):
    """V2: replace subgroup CM with global when kappa exceeds threshold."""
    result = []
    for C_s in cm_list:
        M = np.zeros((2, 2))
        for zh in [0, 1]:
            col_sum = C_s[0, zh] + C_s[1, zh]
            if col_sum > 0:
                M[zh, 0] = C_s[0, zh] / col_sum
                M[zh, 1] = C_s[1, zh] / col_sum
            else:
                M[zh, :] = 0.5
        kappa = np.linalg.cond(M)
        if kappa > kappa_threshold:
            result.append(cm_global.copy())
        else:
            result.append(C_s)
    return result


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    cm_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])

    # Baseline methods
    cm_hb_base = hb_confusion_matrices(z_true_expert, z_hat_expert, alpha_prior=3.0)

    # V1: adaptive alpha
    cm_v1 = hb_adaptive_alpha(z_true_expert, z_hat_expert)

    # V2: kappa fallback on baseline HB
    cm_v2 = apply_kappa_fallback(cm_hb_base, cm_global)

    # V3: adaptive alpha + kappa fallback
    cm_v3 = apply_kappa_fallback(cm_v1, cm_global)

    rows = []
    for s in range(K):
        # Oracle
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

        tau_gl, se_gl = invert_cm(cm_global, tau_obs, se_obs)
        tau_hb, se_hb = invert_cm(cm_hb_base[s], tau_obs, se_obs)
        tau_v1, se_v1 = invert_cm(cm_v1[s], tau_obs, se_obs)
        tau_v2, se_v2 = invert_cm(cm_v2[s], tau_obs, se_obs)
        tau_v3, se_v3 = invert_cm(cm_v3[s], tau_obs, se_obs)

        methods = [
            ('Oracle', tau_oracle, se_oracle),
            ('Naive', tau_obs, se_obs),
            ('Global', tau_gl, se_gl),
            ('HB (baseline)', tau_hb, se_hb),
            ('V1: Adaptive α', tau_v1, se_v1),
            ('V2: κ-fallback', tau_v2, se_v2),
            ('V3: V1+V2', tau_v3, se_v3),
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

    print(f"K=8 Adaptive Shrinkage: N={N}, n_expert={N_EXPERT}, N_MC={N_MC}")
    print(f"PI = {[f'{p:.3f}' for p in PI]}")
    print(f"PI_THRESHOLD={PI_THRESHOLD}, ALPHA_LOW={ALPHA_LOW}, ALPHA_HIGH={ALPHA_HIGH}")
    print(f"KAPPA_STRICT={KAPPA_STRICT}")

    all_rows = []
    for mc in range(N_MC):
        if mc % 10 == 0:
            print(f"  MC {mc}/{N_MC} ...", flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_k8_adaptive.csv', index=False)
    print(f"\nWrote {len(df)} rows")

    summary = df.groupby(['method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()

    summary.to_csv('results/exp_k8_adaptive_summary.csv', index=False)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'Method':>18} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 44)
    for _, row in summary.iterrows():
        print(f"{row['method']:>18} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f}")

    # Per-subgroup RMSE
    sub_summary = df.groupby(['method', 'subgroup']).agg(
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        pi_s=('pi_s', 'first'),
    ).reset_index()

    print("\n" + "=" * 70)
    print("PER-SUBGROUP RMSE")
    print("=" * 70)
    for method in ['Oracle', 'Global', 'HB (baseline)', 'V1: Adaptive α',
                    'V2: κ-fallback', 'V3: V1+V2']:
        m_data = sub_summary[sub_summary['method'] == method].sort_values('subgroup')
        rmses = [f"{r['rmse']:.3f}" for _, r in m_data.iterrows()]
        print(f"  {method:>18}: {' '.join(rmses)}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
