#!/usr/bin/env python3
"""exp036_bootstrap_se_v2.py — Bootstrap SE with correct n_estimators=200

Fixes hyperparameter inconsistency: original exp036 used n_estimators=48,
but standard experiments use 200. This version uses n_estimators=200 with
reduced MC=20 and B=100 to control computation time.

Setup:
  - K=2, N=5000, n_expert=250, pi=[0.05, 0.25]
  - CausalForest via econml CausalForestDML (discrete_treatment=True)
  - n_estimators=200, min_samples_leaf=20
  - Bootstrap B=100, re-estimating CM + CF per bootstrap sample
  - 20 MC runs

Output:
  results/exp_bootstrap_se_v2.csv
  results/exp_bootstrap_se_v2_summary.csv
  results/exp_bootstrap_se_v2_subgroup.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False

N = 5000
K = 2
N_EXPERT = 250
PI = [0.05, 0.25]
TAU_Z = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 20
N_BOOTSTRAP = 100
N_ESTIMATORS = 200


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


def fit_cf(Y, T, X_feat, seed):
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=20,
        random_state=seed,
    )
    cf.fit(Y, T, X=X_feat, W=None)
    return cf


def get_cf_subgroup_estimates(cf, X_feat, Z_labels, S):
    results = {}
    for s in range(K):
        for z in [0, 1]:
            mask_sz = (S == s) & (Z_labels == z)
            if mask_sz.sum() < 5:
                results[(s, z)] = (np.nan, np.nan)
                continue
            te = cf.effect(X_feat[mask_sz])
            tau_hat = te.mean()
            try:
                inf = cf.effect_inference(X_feat[mask_sz])
                se_raw = inf.stderr
                if hasattr(se_raw, 'values'):
                    se_raw = se_raw.values
                se_hat = np.mean(se_raw.flatten())
            except Exception:
                se_hat = te.std() / np.sqrt(len(te))
            results[(s, z)] = (tau_hat, se_hat)
    return results


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)
    X_feat = np.column_stack([Z_hat, S])

    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]
    cm_hb = hb_confusion_matrices(z_true_expert, z_hat_expert)

    # 1. Plug-in estimates
    cf = fit_cf(Y, T, X_feat, seed)
    plugin_raw = get_cf_subgroup_estimates(cf, X_feat, Z_hat, S)

    plugin_corrected = {}
    for s in range(K):
        if (s, 0) not in plugin_raw or (s, 1) not in plugin_raw:
            continue
        t0, t1 = plugin_raw[(s, 0)], plugin_raw[(s, 1)]
        if np.isnan(t0[0]) or np.isnan(t1[0]):
            continue
        tau_obs = np.array([t0[0], t1[0]])
        se_obs = np.array([t0[1], t1[1]])
        tau_c, se_c = invert_cm(cm_hb[s], tau_obs, se_obs)
        plugin_corrected[s] = {0: (tau_c[0], se_c[0]), 1: (tau_c[1], se_c[1])}

    # 2. Bootstrap SE
    rng = np.random.RandomState(seed + 10000)
    boot_estimates = {(s, z): [] for s in range(K) for z in [0, 1]}

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(N, N, replace=True)
        Y_b, T_b, Z_hat_b, S_b = Y[idx], T[idx], Z_hat[idx], S[idx]
        Z_true_b = Z_true[idx]
        expert_mask_b = expert_mask[idx]

        z_true_expert_b = [Z_true_b[expert_mask_b & (S_b == s)] for s in range(K)]
        z_hat_expert_b = [Z_hat_b[expert_mask_b & (S_b == s)] for s in range(K)]

        if any(len(z_true_expert_b[s]) < 3 for s in range(K)):
            continue

        cm_hb_b = hb_confusion_matrices(z_true_expert_b, z_hat_expert_b)
        X_feat_b = np.column_stack([Z_hat_b, S_b])

        try:
            cf_b = fit_cf(Y_b, T_b, X_feat_b, seed + b)
        except Exception:
            continue

        for s in range(K):
            tau_obs_list = []
            for z in [0, 1]:
                mask_sz = (S_b == s) & (Z_hat_b == z)
                if mask_sz.sum() < 5:
                    tau_obs_list.append(np.nan)
                    continue
                te = cf_b.effect(X_feat_b[mask_sz])
                tau_obs_list.append(te.mean())

            if any(np.isnan(t) for t in tau_obs_list):
                continue

            tau_obs_b = np.array(tau_obs_list)
            se_dummy = np.array([0.1, 0.1])
            tau_c, _ = invert_cm(cm_hb_b[s], tau_obs_b, se_dummy)
            for zi, z in enumerate([0, 1]):
                boot_estimates[(s, z)].append(tau_c[zi])

    boot_se = {}
    for key, vals in boot_estimates.items():
        boot_se[key] = np.std(vals, ddof=1) if len(vals) >= 10 else np.nan

    # 3. Oracle CF
    X_feat_oracle = np.column_stack([Z_true, S])
    cf_oracle = fit_cf(Y, T, X_feat_oracle, seed)

    rows = []
    for s in range(K):
        if s not in plugin_corrected:
            continue
        for z in [0, 1]:
            true_tau = TAU_Z[z]

            tau_p = plugin_corrected[s][z][0]
            se_p = plugin_corrected[s][z][1]
            ci_lo_p = tau_p - 1.96 * se_p
            ci_hi_p = tau_p + 1.96 * se_p
            covers_p = int(ci_lo_p <= true_tau <= ci_hi_p)

            se_b = boot_se.get((s, z), np.nan)
            if not np.isnan(se_b):
                ci_lo_b = tau_p - 1.96 * se_b
                ci_hi_b = tau_p + 1.96 * se_b
                covers_b = int(ci_lo_b <= true_tau <= ci_hi_b)
            else:
                covers_b = np.nan

            mask_sz_o = (S == s) & (Z_true == z)
            if mask_sz_o.sum() >= 5:
                te_o = cf_oracle.effect(X_feat_oracle[mask_sz_o])
                tau_o = te_o.mean()
                try:
                    inf_o = cf_oracle.effect_inference(X_feat_oracle[mask_sz_o])
                    se_o_raw = inf_o.stderr
                    if hasattr(se_o_raw, 'values'):
                        se_o_raw = se_o_raw.values
                    se_o = np.mean(se_o_raw.flatten())
                except Exception:
                    se_o = te_o.std() / np.sqrt(len(te_o))
                ci_lo_o = tau_o - 1.96 * se_o
                ci_hi_o = tau_o + 1.96 * se_o
                covers_o = int(ci_lo_o <= true_tau <= ci_hi_o)
            else:
                tau_o, se_o, covers_o = np.nan, np.nan, np.nan

            rows.append({
                'mc_seed': seed,
                'subgroup': s,
                'z_level': z,
                'tau_true': true_tau,
                'tau_echte': tau_p,
                'se_plugin': se_p,
                'covers_plugin': covers_p,
                'se_bootstrap': se_b,
                'covers_bootstrap': covers_b,
                'tau_oracle': tau_o,
                'se_oracle': se_o,
                'covers_oracle': covers_o,
                'se_ratio': se_b / se_p if se_p > 0 and not np.isnan(se_b) else np.nan,
            })
    return rows


def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"Bootstrap SE v2 (n_estimators={N_ESTIMATORS}): K={K}, N={N}, n_expert={N_EXPERT}")
    print(f"PI={PI}, TAU_Z={TAU_Z}")
    print(f"N_MC={N_MC}, N_BOOTSTRAP={N_BOOTSTRAP}")
    print(f"econml available: {HAS_ECONML}")

    if not HAS_ECONML:
        print("ERROR: econml required.")
        sys.exit(1)

    all_rows = []
    for mc in range(N_MC):
        mc_t0 = time.time()
        print(f"  MC {mc+1}/{N_MC} ...", end='', flush=True)
        rows = run_one_mc(42 + mc)
        all_rows.extend(rows)
        elapsed = time.time() - mc_t0
        print(f" {elapsed:.1f}s ({len(rows)} rows)")

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_bootstrap_se_v2.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_bootstrap_se_v2.csv")

    summary = pd.DataFrame({
        'se_type': ['plugin', 'bootstrap', 'oracle'],
        'mean_coverage': [
            df['covers_plugin'].mean(),
            df['covers_bootstrap'].dropna().mean(),
            df['covers_oracle'].dropna().mean(),
        ],
        'mean_se': [
            df['se_plugin'].mean(),
            df['se_bootstrap'].dropna().mean(),
            df['se_oracle'].mean(),
        ],
        'se_ratio_boot_plugin': [
            np.nan,
            df['se_ratio'].dropna().mean(),
            np.nan,
        ],
    })
    summary.to_csv('results/exp_bootstrap_se_v2_summary.csv', index=False)

    sg_summary = df.groupby('subgroup').agg(
        plugin_coverage=('covers_plugin', 'mean'),
        bootstrap_coverage=('covers_bootstrap', lambda x: x.dropna().mean()),
        oracle_coverage=('covers_oracle', lambda x: x.dropna().mean()),
        plugin_se=('se_plugin', 'mean'),
        bootstrap_se=('se_bootstrap', lambda x: x.dropna().mean()),
        se_ratio=('se_ratio', lambda x: x.dropna().mean()),
    ).reset_index()
    sg_summary.to_csv('results/exp_bootstrap_se_v2_subgroup.csv', index=False)

    print("\n" + "=" * 80)
    print("BOOTSTRAP SE v2 — COVERAGE COMPARISON")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("BOOTSTRAP SE v2 — PER-SUBGROUP")
    print("=" * 80)
    print(sg_summary.to_string(index=False))

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
