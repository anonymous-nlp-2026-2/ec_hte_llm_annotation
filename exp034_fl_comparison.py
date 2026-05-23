#!/usr/bin/env python3
"""exp034_fl_comparison.py — F-L vs EC-HTE direct comparison

Demonstrates that Theorem 1's delta_s * Delta_tau product structure is NOT a
simple extension of Frazis-Loewenstein (2003). F-L's scalar correction
tau_FL = tau_obs / (1-2*pi_bar) predicts bias direction incorrectly under
subgroup heterogeneity with sign-reversal of delta_s.

Setup:
  - K=2, N=5000, n_expert=250, pi=[0.05, 0.25] (standard extreme)
  - tau(0)=0, tau(1)=0.3 -> Delta_tau=0.3
  - pi_bar = 0.15, so delta_s0 = 0.05-0.15 = -0.10, delta_s1 = 0.25-0.15 = +0.10
  - s0 has low misclassification (pi=0.05), s1 has high misclassification (pi=0.25)

Methods compared:
  1. F-L scalar: tau_FL = tau_obs / (1-2*pi_bar) — single global scaling
  2. F-L per-subgroup: tau_FL_s = tau_obs_s / (1-2*pi_s) — subgroup-level scaling
  3. EC-HTE (HB): full hierarchical Bayes pipeline with CM inversion
  4. Naive: no correction
  5. Oracle: true Z labels

Output:
  results/exp_fl_comparison.csv         (raw per-MC rows)
  results/exp_fl_comparison_summary.csv (grouped by method)
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
TAU_Z = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 100
PI_BAR = np.mean(PI)  # 0.15


def generate_data(seed):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng.binomial(1, 0.5, N)
    T = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, PI[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    tau = np.array([TAU_Z[z] for z in Z_true])
    Y = 1.0 + 0.5 * Z_true + tau * T + rng.normal(0, 1, N)

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


def dim_estimate(Y, T, Z_labels, S, s):
    """DIM estimator for subgroup s, returns dict z -> (tau_hat, se)."""
    mask_s = S == s
    results = {}
    for z in [0, 1]:
        mask_sz = mask_s & (Z_labels == z)
        idx_t1 = np.where(mask_sz & (T == 1))[0]
        idx_t0 = np.where(mask_sz & (T == 0))[0]
        n1 = len(idx_t1)
        n0 = len(idx_t0)
        if n1 < 2 or n0 < 2:
            results[z] = (np.nan, np.nan)
            continue
        mu1 = Y[idx_t1].mean()
        mu0 = Y[idx_t0].mean()
        tau_hat = mu1 - mu0
        se = np.sqrt(Y[idx_t1].var(ddof=1) / n1 + Y[idx_t0].var(ddof=1) / n0)
        results[z] = (tau_hat, se)
    return results


def fl_scalar_correction(tau_obs, se_obs, pi_bar):
    """F-L scalar: tau_FL = tau_obs / (1-2*pi_bar)."""
    denom = 1 - 2 * pi_bar
    if abs(denom) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    tau_fl = tau_obs / denom
    se_fl = se_obs / abs(denom)
    return tau_fl, se_fl


def fl_subgroup_correction(tau_obs, se_obs, pi_s):
    """F-L per-subgroup: tau_FL_s = tau_obs_s / (1-2*pi_s)."""
    denom = 1 - 2 * pi_s
    if abs(denom) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    tau_fl = tau_obs / denom
    se_fl = se_obs / abs(denom)
    return tau_fl, se_fl


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    cm_hb = hb_confusion_matrices(z_true_expert, z_hat_expert)
    cm_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])

    # Estimate pi_bar from expert data (for F-L scalar)
    pi_bar_hat = (Z_true[expert_mask] != Z_hat[expert_mask]).mean()
    # Estimate pi_s from expert data (for F-L per-subgroup)
    pi_s_hat = []
    for s in range(K):
        mask_es = expert_mask & (S == s)
        if mask_es.sum() > 0:
            pi_s_hat.append((Z_true[mask_es] != Z_hat[mask_es]).mean())
        else:
            pi_s_hat.append(PI_BAR)

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

        # F-L scalar correction (same scaling for all subgroups)
        tau_fl_scalar, se_fl_scalar = fl_scalar_correction(tau_obs, se_obs, pi_bar_hat)

        # F-L per-subgroup correction
        tau_fl_sub, se_fl_sub = fl_subgroup_correction(tau_obs, se_obs, pi_s_hat[s])

        # EC-HTE correction (CM inversion)
        tau_hb, se_hb = invert_cm(cm_hb[s], tau_obs, se_obs)

        # Global correction (CM inversion with global CM)
        tau_gl, se_gl = invert_cm(cm_global, tau_obs, se_obs)

        methods = [
            ('Oracle', tau_oracle, se_oracle),
            ('Naive', tau_obs, se_obs),
            ('F-L Scalar', tau_fl_scalar, se_fl_scalar),
            ('F-L Per-Subgroup', tau_fl_sub, se_fl_sub),
            ('Global (CM inv)', tau_gl, se_gl),
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

                # Bias direction analysis
                bias = th - true_tau
                bias_sign = 'positive' if bias > 0 else ('negative' if bias < 0 else 'zero')
                delta_s = PI[s] - PI_BAR  # true delta_s

                rows.append({
                    'mc_seed': seed,
                    'subgroup': s,
                    'z_level': z,
                    'method': method_name,
                    'tau_hat': th,
                    'tau_true': true_tau,
                    'bias': bias,
                    'bias_sign': bias_sign,
                    'se': se,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'covers': covers,
                    'pi_s': PI[s],
                    'delta_s': delta_s,
                    'pi_bar_hat': pi_bar_hat,
                    'pi_s_hat': pi_s_hat[s],
                })
    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"F-L vs EC-HTE Comparison: K={K}, N={N}, n_expert={N_EXPERT}")
    print(f"PI={PI}, PI_BAR={PI_BAR}, TAU_Z={TAU_Z}")
    print(f"delta_s0={PI[0]-PI_BAR:.2f}, delta_s1={PI[1]-PI_BAR:.2f}")
    print(f"N_MC={N_MC}")

    all_rows = []
    for mc in range(N_MC):
        if mc % 20 == 0:
            print(f"  MC {mc}/{N_MC} ...", flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_fl_comparison.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_fl_comparison.csv")

    # Summary
    summary = df.groupby(['method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()
    summary.to_csv('results/exp_fl_comparison_summary.csv', index=False)

    # Per-subgroup summary (key for sign-reversal analysis)
    sg_summary = df.groupby(['method', 'subgroup']).agg(
        mean_bias=('bias', 'mean'),
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        bias_positive_pct=('bias', lambda x: (x > 0).mean()),
    ).reset_index()
    sg_summary.to_csv('results/exp_fl_comparison_subgroup.csv', index=False)

    # Print results
    print("\n" + "=" * 80)
    print("F-L vs EC-HTE — AGGREGATE")
    print("=" * 80)
    print(f"{'Method':>20} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 50)
    for _, row in summary.sort_values('mean_abs_bias').iterrows():
        print(f"{row['method']:>20} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f}")

    print("\n" + "=" * 80)
    print("F-L vs EC-HTE — PER-SUBGROUP (key: bias direction)")
    print("=" * 80)
    print(f"{'Method':>20} {'SG':>3} {'Mean Bias':>10} {'|Bias|':>8} "
          f"{'RMSE':>8} {'Cov':>6} {'Bias>0%':>8}")
    print("-" * 70)
    for _, row in sg_summary.sort_values(['subgroup', 'method']).iterrows():
        print(f"{row['method']:>20} {row['subgroup']:3.0f} {row['mean_bias']:10.4f} "
              f"{row['mean_abs_bias']:8.4f} {row['rmse']:8.4f} "
              f"{row['coverage']:6.3f} {row['bias_positive_pct']:8.1%}")

    # Sign-reversal analysis
    print("\n" + "=" * 80)
    print("SIGN-REVERSAL ANALYSIS")
    print("=" * 80)
    print("Expected: delta_s0=-0.10 (s0 over-corrected by global/F-L scalar)")
    print("          delta_s1=+0.10 (s1 under-corrected by global/F-L scalar)")
    for method in ['F-L Scalar', 'F-L Per-Subgroup', 'Global (CM inv)', 'HB EC-HTE']:
        sub = sg_summary[sg_summary['method'] == method]
        s0_bias = sub[sub['subgroup'] == 0]['mean_bias'].values[0]
        s1_bias = sub[sub['subgroup'] == 1]['mean_bias'].values[0]
        sign_consistent = (s0_bias * s1_bias < 0)
        print(f"  {method:>20}: s0 bias={s0_bias:+.4f}, s1 bias={s1_bias:+.4f}, "
              f"sign-reversal={'YES' if sign_consistent else 'NO'}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
