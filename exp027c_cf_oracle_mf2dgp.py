#!/usr/bin/env python3
"""exp027c_cf_oracle_mf2dgp.py — CausalForest Oracle Coverage under MF2 DGP (exp-027c)

Replicates the exact DGP from exp_mf2_cf_bootstrap_se.py and runs 6 methods:
  1. Oracle (true Z): CausalForestDML with Z_true
  2. EC-HTE true CM: Z_hat forest + per-subgroup correction with true C_s
  3. EC-HTE estimated CM (HB): Z_hat forest + per-subgroup correction with estimated C_s
  4. Global true CM: Z_hat forest + global correction with true C_bar
  5. Global estimated CM: Z_hat forest + global correction with estimated C_bar
  6. Naive: Z_hat forest, no correction

DGP (matching MF2 exactly):
  K=2, π=[0.05, 0.25], τ(z=0)=0.10, τ(z=1)=0.40
  N=5000, n_expert=250, X∈R^5
  Y = 1.0 + τ(Z)*T + 0.5*X₀ + ε, ε~N(0,1)
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

# ── MF2 DGP Parameters (exact copy) ──
N = 5000
K = 2
N_EXPERT = 250
MISCLASS = [0.05, 0.25]
TAU_Z0 = 0.10
TAU_Z1 = 0.40
TAU_TRUE = np.array([TAU_Z0, TAU_Z1])
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 100
BASE_SEED = 42


def generate_data(seed):
    """MF2-identical data generation (same RNG draw order)."""
    rng = np.random.RandomState(seed)
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    X = rng.randn(N, 5)
    tau = np.where(Z == 1, TAU_Z1, TAU_Z0)
    Y = 1.0 + tau * T + 0.5 * X[:, 0] + rng.randn(N)

    mc_rates = np.where(S == 0, MISCLASS[0], MISCLASS[1])
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)

    idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True

    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, expert_mask=expert_mask)


def estimate_cm(z_true, z_hat):
    """MF2-identical CM estimation with Laplace smoothing."""
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mask = z_true == z
        n = mask.sum()
        for zh in [0, 1]:
            C[z, zh] = ((z_hat[mask] == zh).sum() + 1) / (n + 2) if n > 0 else 0.5
    return C


def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        M[zh, 0] = C[0, zh] / cs if cs > 0 else 0.5
        M[zh, 1] = C[1, zh] / cs if cs > 0 else 0.5
    return M


def invert_mixing(M, tau_obs, se_obs):
    if np.linalg.cond(M) > 100 or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


def true_confusion_matrix(pi_s):
    return np.array([[1 - pi_s, pi_s],
                     [pi_s, 1 - pi_s]])


def fit_cf(Y, T, X_features, seed):
    """CausalForestDML with MF2-identical hyperparameters."""
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
    )
    cf.fit(Y, T, X=X_features)
    return cf


def get_subgroup_cates_and_se(cf, X_features, S, Z_col, X):
    """Per-(subgroup, z) CATEs and SEs following MF2 approach."""
    cate_all = cf.effect(X_features).flatten()

    sub_tau = {}
    for s in range(K):
        sm = S == s
        vals = []
        for z in [0, 1]:
            mask = sm & (Z_col == z)
            if mask.sum() == 0:
                vals.append(np.nan)
            else:
                vals.append(cate_all[mask].mean())
        sub_tau[s] = np.array(vals)

    X_mean = X.mean(axis=0)
    X_eval = np.array([
        np.concatenate([[0], X_mean]),
        np.concatenate([[1], X_mean]),
    ])
    try:
        inf = cf.effect_inference(X_eval)
        se_raw = np.asarray(inf.stderr).flatten()
        if np.any(np.isnan(se_raw)) or np.any(se_raw <= 0):
            se_raw = np.full(2, np.nan)
    except Exception:
        se_raw = np.full(2, np.nan)

    sub_se = {s: se_raw.copy() for s in range(K)}
    return sub_tau, sub_se


def run_one_mc(seed):
    data = generate_data(seed)
    Y, T, Z, Z_hat, S, X = (
        data['Y'], data['T'], data['Z'], data['Z_hat'], data['S'], data['X']
    )
    em = data['expert_mask']

    # ── Fit two forests ──
    X_oracle = np.column_stack([Z, X]).astype(float)
    X_zhat = np.column_stack([Z_hat, X]).astype(float)

    cf_oracle = fit_cf(Y, T, X_oracle, seed)
    cf_zhat = fit_cf(Y, T, X_zhat, seed)

    # ── CATEs and SEs ──
    oracle_tau, oracle_se = get_subgroup_cates_and_se(cf_oracle, X_oracle, S, Z, X)
    zhat_tau, zhat_se = get_subgroup_cates_and_se(cf_zhat, X_zhat, S, Z_hat, X)

    # ── Confusion matrices ──
    M_true_sub = {}
    for s in range(K):
        C_s_true = true_confusion_matrix(MISCLASS[s])
        M_true_sub[s] = build_mixing_matrix(C_s_true)

    C_global_true = sum(SUB_WEIGHTS[s] * true_confusion_matrix(MISCLASS[s])
                        for s in range(K))
    M_global_true = build_mixing_matrix(C_global_true)

    M_est_sub = {}
    for s in range(K):
        mask_s = em & (S == s)
        if mask_s.sum() >= 4:
            C_s = estimate_cm(Z[mask_s], Z_hat[mask_s])
        else:
            C_s = estimate_cm(Z[em], Z_hat[em])
        M_est_sub[s] = build_mixing_matrix(C_s)

    C_global_est = estimate_cm(Z[em], Z_hat[em])
    M_global_est = build_mixing_matrix(C_global_est)

    # ── Per-subgroup corrections and coverage ──
    rows = []
    for s in range(K):
        tau_obs = zhat_tau[s]
        se_obs = zhat_se[s]

        tau_ec_true, se_ec_true = invert_mixing(M_true_sub[s], tau_obs, se_obs)
        tau_ec_est, se_ec_est = invert_mixing(M_est_sub[s], tau_obs, se_obs)
        tau_gl_true, se_gl_true = invert_mixing(M_global_true, tau_obs, se_obs)
        tau_gl_est, se_gl_est = invert_mixing(M_global_est, tau_obs, se_obs)

        methods = [
            ('Oracle', oracle_tau[s], oracle_se[s]),
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
    print("=== exp-027c: MF2 DGP (τ(0)=0.10, τ(1)=0.40) ===")
    print("=" * 80)

    print(f"\n{'Method':<28s} | {'Coverage':>8s} | {'|Bias|':>7s} | {'RMSE':>7s} | {'Mean SE':>7s}")
    print(f"{'-' * 28}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")

    for method in method_order:
        sub = df_agg[df_agg['method'] == method]
        cov = sub['coverage'].mean()
        abs_bias = sub['bias'].abs().mean()
        rmse = sub['rmse'].mean()
        se_mean = sub['se_mean'].mean()
        print(f"{method:<28s} | {cov:8.4f} | {abs_bias:7.4f} | {rmse:7.4f} | {se_mean:7.4f}")

    print(f"\nPer-subgroup detail:")
    hdr = (f"{'Method':<28s} | {'Sub':>3s} | {'z':>1s} | "
           f"{'Bias':>8s} | {'RMSE':>7s} | {'Coverage':>8s} | {'Mean SE':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for method in method_order:
        for s in range(K):
            for z in [0, 1]:
                r = df_agg[(df_agg['method'] == method) &
                           (df_agg['subgroup'] == s) &
                           (df_agg['z'] == z)]
                if r.empty:
                    continue
                r = r.iloc[0]
                print(f"{method:<28s} | {s:>3d} | {z:>1d} | "
                      f"{r['bias']:8.4f} | {r['rmse']:7.4f} | "
                      f"{r['coverage']:8.4f} | {r['se_mean']:7.4f}")
        print()


def print_diagnostics(df_agg):
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPARISON")
    print("=" * 80)

    oracle_cov_c = df_agg[df_agg['method'] == 'Oracle']['coverage'].mean()
    ec_est_cov_c = df_agg[df_agg['method'] == 'EC-HTE estimated CM']['coverage'].mean()

    b_path = 'results/exp_cf_oracle_coverage_k2.csv'
    if os.path.exists(b_path):
        df_b = pd.read_csv(b_path)
        oracle_cov_b = df_b[df_b['method'] == 'Oracle']['coverage'].mean()
        ec_est_cov_b = df_b[df_b['method'] == 'EC-HTE estimated CM']['coverage'].mean()

        print(f"\n1. Oracle coverage:")
        print(f"   exp-027b (τ(0)=0.00, τ(1)=0.30): {oracle_cov_b:.4f}")
        print(f"   exp-027c (τ(0)=0.10, τ(1)=0.40): {oracle_cov_c:.4f}")

        print(f"\n2. EC-HTE estimated CM coverage:")
        print(f"   exp-027b (τ(0)=0.00, τ(1)=0.30): {ec_est_cov_b:.4f}")
        print(f"   exp-027c (τ(0)=0.10, τ(1)=0.40): {ec_est_cov_c:.4f}")
    else:
        print(f"\n  exp-027b results not found at {b_path}")
        print(f"  exp-027c Oracle coverage: {oracle_cov_c:.4f}")
        print(f"  exp-027c EC-HTE est CM coverage: {ec_est_cov_c:.4f}")

    mf2_path = 'results/exp_mf2_cf_bootstrap_se.csv'
    if os.path.exists(mf2_path):
        df_mf2 = pd.read_csv(mf2_path)
        mf2_250 = df_mf2[(df_mf2['n_expert'] == 250) & (df_mf2['se_method'] == 'plugin')]
        if not mf2_250.empty:
            mf2_cov = mf2_250['coverage_avg'].values[0]
            mf2_z0 = mf2_250['coverage_z0'].values[0]
            mf2_z1 = mf2_250['coverage_z1'].values[0]
            print(f"\n3. MF2 original EC-HTE plugin coverage (n_expert=250):")
            print(f"   z0={mf2_z0:.2f}, z1={mf2_z1:.2f}, avg={mf2_cov:.2f}")
    else:
        print(f"\n3. MF2 results not found")
        print(f"   (Reference: MF2 reported EC-HTE plugin coverage ≈ 0.83)")


if __name__ == '__main__':
    print(f"exp-027c CausalForest Oracle Coverage — MF2 DGP")
    print(f"N={N}, K={K}, pi={MISCLASS}, weights={SUB_WEIGHTS}")
    print(f"tau(0)={TAU_Z0}, tau(1)={TAU_Z1}")
    print(f"n_expert={N_EXPERT}, N_MC={N_MC}, seed_base={BASE_SEED}")
    print(f"Y = 1.0 + tau*T + 0.5*X0 + N(0,1)")
    print(f"X_features = [Z, X] (6 columns)")

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
            print(f"  [{mc + 1}/{N_MC}] {el:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.1f}s ({N_MC} runs, {elapsed / N_MC:.1f}s/run)")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate_results(df_raw)

    os.makedirs('results', exist_ok=True)
    csv_path = 'results/exp_cf_oracle_coverage_mf2dgp.csv'
    df_agg.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df_agg)} rows)")

    print_summary(df_agg)
    print_diagnostics(df_agg)
