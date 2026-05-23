#!/usr/bin/env python3
"""exp027b_cf_oracle_k2.py — CausalForest Oracle Coverage, K=2 extreme (exp-027b)

K=2 variant of exp-027 to match the paper's DGP for the 83-85% coverage gap analysis.

Six methods compared:
  1. Oracle (true Z): CausalForestDML with Z_true, no correction
  2. EC-HTE true CM: Z_hat forest + per-subgroup correction with true C_s
  3. EC-HTE estimated CM: Z_hat forest + per-subgroup correction with estimated C_s (HB)
  4. Global true CM: Z_hat forest + global correction with true C_bar
  5. Global estimated CM: Z_hat forest + global correction with estimated C_bar
  6. Naive: Z_hat forest, no correction

DGP: K=2, pi=[0.05,0.25], weights=[0.5,0.5], tau(0)=0, tau(1)=0.3, N=5000, n_expert=250
Output: results/exp_cf_oracle_coverage_k2.csv
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

N = 5000
K = 2
N_EXPERT = 250
PI = [0.05, 0.25]
TAU_TRUE = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 100
BASE_SEED = 42


def generate_data(seed):
    rng = np.random.RandomState(seed)
    S = rng.randint(0, K, N)
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, PI[s], mask_s.sum())
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


def invert_mixing(M, tau_obs, se_obs, cond_threshold=100):
    cond = np.linalg.cond(M)
    if cond > cond_threshold or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


def true_confusion_matrix(pi_s):
    return np.array([[1 - pi_s, pi_s],
                     [pi_s, 1 - pi_s]])


def fit_cf_and_predict(Y, T, X_features, X_eval):
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=20,
        random_state=42,
    )
    cf.fit(Y, T, X=X_features, W=None)

    cate_vals = cf.effect(X_eval).flatten()
    try:
        inf = cf.effect_inference(X_eval)
        se_raw = inf.stderr
        if hasattr(se_raw, 'values'):
            se_raw = se_raw.values
        se_vals = np.asarray(se_raw).flatten()
    except Exception:
        se_vals = np.full(len(X_eval), np.nan)

    return cate_vals, se_vals


def run_one_mc(seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed)

    X_eval = np.array([[z, s] for s in range(K) for z in [0, 1]], dtype=float)

    X_oracle = np.column_stack([Z_true, S]).astype(float)
    oracle_cate, oracle_se = fit_cf_and_predict(Y, T, X_oracle, X_eval)

    X_zhat = np.column_stack([Z_hat, S]).astype(float)
    zhat_cate, zhat_se = fit_cf_and_predict(Y, T, X_zhat, X_eval)

    M_true_sub = {}
    for s in range(K):
        C_s_true = true_confusion_matrix(PI[s])
        M_true_sub[s] = build_mixing_matrix(C_s_true)

    C_global_true = sum(SUB_WEIGHTS[s] * true_confusion_matrix(PI[s]) for s in range(K))
    M_global_true = build_mixing_matrix(C_global_true)

    M_est_sub = {}
    for s in range(K):
        mask_s = expert_mask & (S == s)
        if mask_s.sum() >= 4:
            C_s = compute_confusion_matrix(Z_true[mask_s], Z_hat[mask_s])
        else:
            C_s = np.array([[0.5, 0.5], [0.5, 0.5]])
        M_est_sub[s] = build_mixing_matrix(C_s)

    C_global_est = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])
    M_global_est = build_mixing_matrix(C_global_est)

    rows = []
    for s in range(K):
        idx_z0 = s * 2
        idx_z1 = s * 2 + 1

        o_tau = np.array([oracle_cate[idx_z0], oracle_cate[idx_z1]])
        o_se = np.array([oracle_se[idx_z0], oracle_se[idx_z1]])

        tau_obs = np.array([zhat_cate[idx_z0], zhat_cate[idx_z1]])
        se_obs = np.array([zhat_se[idx_z0], zhat_se[idx_z1]])

        tau_ec_true, se_ec_true = invert_mixing(M_true_sub[s], tau_obs, se_obs)
        tau_ec_est, se_ec_est = invert_mixing(M_est_sub[s], tau_obs, se_obs)
        tau_gl_true, se_gl_true = invert_mixing(M_global_true, tau_obs, se_obs)
        tau_gl_est, se_gl_est = invert_mixing(M_global_est, tau_obs, se_obs)

        methods = [
            ('Oracle', o_tau, o_se),
            ('EC-HTE true CM', tau_ec_true, se_ec_true),
            ('EC-HTE estimated CM', tau_ec_est, se_ec_est),
            ('Global true CM', tau_gl_true, se_gl_true),
            ('Global estimated CM', tau_gl_est, se_gl_est),
            ('Naive', tau_obs, se_obs),
        ]

        for method_name, tau_hat, se_hat in methods:
            for zi, z in enumerate([0, 1]):
                th = tau_hat[zi]
                se = se_hat[zi]
                if np.isnan(se) or np.isnan(th):
                    covers = np.nan
                else:
                    ci_lo = th - 1.96 * se
                    ci_hi = th + 1.96 * se
                    covers = int(ci_lo <= TAU_TRUE[z] <= ci_hi)
                rows.append({
                    'method': method_name,
                    'subgroup': s,
                    'z': z,
                    'tau_hat': th,
                    'se': se,
                    'tau_true': TAU_TRUE[z],
                    'covers': covers,
                    'mc_run': seed,
                })

    return rows


def aggregate_results(df):
    grp_cols = ['method', 'subgroup', 'z']
    rows = []
    for keys, g in df.groupby(grp_cols):
        method, subgroup, z = keys
        err = g['tau_hat'].values - g['tau_true'].values
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        cov = g['covers'].mean()
        se_mean = g['se'].mean()
        n_mc = len(g)
        rows.append({
            'method': method,
            'subgroup': int(subgroup),
            'z': int(z),
            'bias': bias,
            'rmse': rmse,
            'coverage': cov,
            'se_mean': se_mean,
            'mc_run_count': n_mc,
        })
    return pd.DataFrame(rows)


def print_summary(df_agg):
    method_order = [
        'Oracle', 'EC-HTE true CM', 'EC-HTE estimated CM',
        'Global true CM', 'Global estimated CM', 'Naive',
    ]

    print("\n" + "=" * 80)
    print("exp-027b CausalForest Oracle Coverage (K=2, extreme)")
    print("=" * 80)

    print(f"\n{'Method':<28s} | {'|Bias|':>7s} | {'RMSE':>7s} | {'Coverage':>8s} | {'Mean SE':>7s}")
    print(f"{'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

    for method in method_order:
        sub = df_agg[df_agg['method'] == method]
        abs_bias = sub['bias'].abs().mean()
        rmse = sub['rmse'].mean()
        cov = sub['coverage'].mean()
        se_mean = sub['se_mean'].mean()
        print(f"{method:<28s} | {abs_bias:7.4f} | {rmse:7.4f} | {cov:8.4f} | {se_mean:7.4f}")

    print(f"\n\nPer-subgroup detail:")
    print(f"{'Method':<28s} | {'Sub':>3s} | {'z':>1s} | {'Bias':>8s} | {'RMSE':>7s} | {'Coverage':>8s} | {'Mean SE':>7s}")
    print(f"{'-'*28}-+-{'-'*3}-+-{'-'*1}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

    for method in method_order:
        for s in range(K):
            for z in [0, 1]:
                r = df_agg[(df_agg['method'] == method) &
                           (df_agg['subgroup'] == s) &
                           (df_agg['z'] == z)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{method:<28s} | {s:>3d} | {z:>1d} | {r['bias']:8.4f} | {r['rmse']:7.4f} | {r['coverage']:8.4f} | {r['se_mean']:7.4f}")
        print()

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    oracle_cov = df_agg[df_agg['method'] == 'Oracle']['coverage'].mean()
    ec_true_cov = df_agg[df_agg['method'] == 'EC-HTE true CM']['coverage'].mean()
    ec_est_cov = df_agg[df_agg['method'] == 'EC-HTE estimated CM']['coverage'].mean()
    print(f"  Oracle coverage:           {oracle_cov:.4f}")
    print(f"  EC-HTE true CM coverage:   {ec_true_cov:.4f}")
    print(f"  EC-HTE estimated coverage: {ec_est_cov:.4f}")
    print()
    if oracle_cov < 0.90:
        print("  -> Oracle < 90%: CF SE is inherently under-calibrated")
    else:
        print("  -> Oracle >= 90%: CF SE is reasonably calibrated")
    if abs(ec_true_cov - ec_est_cov) < 0.03:
        print("  -> EC-HTE true CM ~ estimated CM: gap is NOT from CM estimation error")
        print("     Coverage gap comes from CF inherent SE under-calibration")
    elif ec_true_cov > ec_est_cov + 0.05:
        print("  -> EC-HTE true CM >> estimated CM: CM estimation error contributes to gap")
    else:
        print("  -> Mixed signal: both CF SE calibration and CM estimation may contribute")


if __name__ == '__main__':
    print(f"exp-027b CausalForest Oracle Coverage (K=2, extreme)")
    print(f"N={N}, K={K}, pi={PI}, weights={SUB_WEIGHTS}")
    print(f"tau(0)={TAU_TRUE[0]}, tau(1)={TAU_TRUE[1]}")
    print(f"n_expert={N_EXPERT}, N_MC={N_MC}, seed_base={BASE_SEED}")

    print("\nTiming single run...")
    t_test = time.time()
    _ = run_one_mc(seed=9999)
    t_per = time.time() - t_test
    print(f"  {t_per:.2f}s/run (2 forest fits per run)")
    print(f"  Estimated total: {t_per * N_MC:.0f}s ({t_per * N_MC / 60:.1f} min)\n")

    all_rows = []
    t0 = time.time()

    for mc in range(N_MC):
        seed = BASE_SEED + mc
        try:
            rows = run_one_mc(seed)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  WARN: mc={mc} seed={seed} failed: {e}")

        if (mc + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (mc + 1) * (N_MC - mc - 1)
            print(f"  [{mc+1}/{N_MC}] {el:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.1f}s ({N_MC} runs, {elapsed/N_MC:.1f}s/run)")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate_results(df_raw)

    os.makedirs('results', exist_ok=True)
    df_agg.to_csv('results/exp_cf_oracle_coverage_k2.csv', index=False)
    print(f"Saved: results/exp_cf_oracle_coverage_k2.csv ({len(df_agg)} rows)")

    print_summary(df_agg)
