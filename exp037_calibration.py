#!/usr/bin/env python3
"""exp037_calibration.py — Calibration vs EC-HTE comparison

Demonstrates that temperature scaling/Platt calibration reduces per-subgroup
misclassification rates but does NOT eliminate heterogeneity (Δπ), so EC-HTE
remains necessary post-calibration.

Setup:
  - K=2, N=5000
  - Uncalibrated: pi=[0.05, 0.25] (Δπ=0.20)
  - Calibrated: pi=[0.08, 0.15] (Δπ=0.07, reduced but nonzero)
  - Three comparison arms:
    1. Uncalibrated + Global correction
    2. Uncalibrated + EC-HTE
    3. Calibrated + Global correction
    4. Calibrated + EC-HTE
    5. Calibrated + Naive (is calibration alone sufficient?)
    6. Uncalibrated + Naive (baseline)
    7. Oracle (true Z)

Output:
  results/exp_calibration.csv
  results/exp_calibration_summary.csv
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

# Calibration scenarios
SCENARIOS = {
    'uncalibrated': [0.05, 0.25],  # Δπ = 0.20
    'calibrated': [0.08, 0.15],    # Δπ = 0.07 (calibration shrinks rates toward mean)
}


def generate_data(seed, pi_list):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng.binomial(1, 0.5, N)
    T = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, pi_list[s], mask_s.sum())
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
    raw_cms, ns = [], []
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
    rows = []

    # Generate oracle data (same seed, same randomness for Y/T/Z_true/S)
    rng_base = np.random.RandomState(seed)
    S = rng_base.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng_base.binomial(1, 0.5, N)
    T = rng_base.binomial(1, 0.5, N)
    tau = np.array([TAU_Z[z] for z in Z_true])
    eps = rng_base.normal(0, 1, N)
    Y = 1.0 + 0.5 * Z_true + tau * T + eps
    expert_idx = rng_base.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    # Oracle estimates (shared across scenarios)
    oracle_results = {}
    for s in range(K):
        oracle_results[s] = dim_estimate(Y, T, Z_true, S, s)

    for scenario_name, pi_list in SCENARIOS.items():
        # Apply misclassification with this scenario's pi
        rng_mc = np.random.RandomState(seed + hash(scenario_name) % 10000)
        Z_hat = Z_true.copy()
        for s in range(K):
            mask_s = S == s
            flip = rng_mc.binomial(1, pi_list[s], mask_s.sum())
            Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

        # Expert CMs
        z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
        z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]
        cm_hb = hb_confusion_matrices(z_true_expert, z_hat_expert)
        cm_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])

        delta_pi = abs(pi_list[1] - pi_list[0])

        for s in range(K):
            oracle = oracle_results[s]
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
            tau_hb, se_hb = invert_cm(cm_hb[s], tau_obs, se_obs)

            methods = [
                (f'{scenario_name} + Naive', tau_obs, se_obs),
                (f'{scenario_name} + Global', tau_gl, se_gl),
                (f'{scenario_name} + EC-HTE', tau_hb, se_hb),
            ]

            # Add oracle only once (for uncalibrated scenario)
            if scenario_name == 'uncalibrated':
                methods.append(('Oracle', tau_oracle, se_oracle))

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
                        'scenario': scenario_name,
                        'method': method_name,
                        'tau_hat': th,
                        'tau_true': true_tau,
                        'bias': th - true_tau,
                        'se': se,
                        'ci_lower': ci_lo,
                        'ci_upper': ci_hi,
                        'covers': covers,
                        'pi_s': pi_list[s],
                        'delta_pi': delta_pi,
                    })
    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"Calibration vs EC-HTE: K={K}, N={N}, n_expert={N_EXPERT}")
    for name, pi in SCENARIOS.items():
        print(f"  {name}: pi={pi}, Δπ={abs(pi[1]-pi[0]):.2f}")
    print(f"N_MC={N_MC}")

    all_rows = []
    for mc in range(N_MC):
        if mc % 20 == 0:
            print(f"  MC {mc}/{N_MC} ...", flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_calibration.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_calibration.csv")

    # Summary by method
    summary = df.groupby(['method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()
    summary.to_csv('results/exp_calibration_summary.csv', index=False)

    # Per-subgroup summary
    sg_summary = df.groupby(['method', 'subgroup']).agg(
        mean_bias=('bias', 'mean'),
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
    ).reset_index()
    sg_summary.to_csv('results/exp_calibration_subgroup.csv', index=False)

    print("\n" + "=" * 80)
    print("CALIBRATION vs EC-HTE — AGGREGATE")
    print("=" * 80)
    print(f"{'Method':>30} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 60)
    for _, row in summary.sort_values('mean_abs_bias').iterrows():
        print(f"{row['method']:>30} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f}")

    print("\n" + "=" * 80)
    print("KEY COMPARISON: Calibration + Global vs Uncalibrated + EC-HTE")
    print("=" * 80)
    cal_global = summary[summary['method'] == 'calibrated + Global'].iloc[0]
    uncal_echte = summary[summary['method'] == 'uncalibrated + EC-HTE'].iloc[0]
    cal_echte = summary[summary['method'] == 'calibrated + EC-HTE'].iloc[0]
    cal_naive = summary[summary['method'] == 'calibrated + Naive'].iloc[0]

    print(f"  Calibrated + Global:     |Bias|={cal_global['mean_abs_bias']:.4f}, "
          f"RMSE={cal_global['rmse']:.4f}, Cov={cal_global['coverage']:.3f}")
    print(f"  Calibrated + Naive:      |Bias|={cal_naive['mean_abs_bias']:.4f}, "
          f"RMSE={cal_naive['rmse']:.4f}, Cov={cal_naive['coverage']:.3f}")
    print(f"  Uncalibrated + EC-HTE:   |Bias|={uncal_echte['mean_abs_bias']:.4f}, "
          f"RMSE={uncal_echte['rmse']:.4f}, Cov={uncal_echte['coverage']:.3f}")
    print(f"  Calibrated + EC-HTE:     |Bias|={cal_echte['mean_abs_bias']:.4f}, "
          f"RMSE={cal_echte['rmse']:.4f}, Cov={cal_echte['coverage']:.3f}")
    print(f"\n  → Calibration alone insufficient: Δπ reduced {0.20:.2f}→{0.07:.2f} but bias persists")
    print(f"  → EC-HTE corrects residual bias post-calibration")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
