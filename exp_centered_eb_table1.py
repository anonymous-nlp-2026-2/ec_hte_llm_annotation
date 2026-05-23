#!/usr/bin/env python3
"""exp_centered_eb_table1.py — Rerun Table 1 with Centered EB replacing original HB.

Centered EB: Dir(α·ĉ_global_j) instead of Dir(α/2, α/2).
α optimized via marginal likelihood maximization.

Configs (matching current Table 1 DGPs):
1. DIM, K=2, n=250, extreme (exp001 DGP)
2. DIM, K=4, n=500, extreme (exp001 DGP)
3. CausalForest, K=4, n=500, extreme (exp011 DGP)

Output: results/exp_centered_eb_table1.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

N = 5000
N_MC = 100

# ── DIM DGP (from exp001) ────────────────────────────────────────────────────

DIM_CONFIGS = {
    'DIM_K2_extreme': {
        'k': 2, 'misclass': [0.25, 0.05],
        'tau_z1': 0.5, 'tau_z0': 0.1,
        'n_expert': 250, 'labels': ['A', 'B'],
    },
    'DIM_K4_extreme': {
        'k': 4, 'misclass': [0.30, 0.20, 0.10, 0.03],
        'tau_z1': 0.5, 'tau_z0': 0.1,
        'n_expert': 500, 'labels': ['A', 'B', 'C', 'D'],
    },
}

# ── CF DGP (from exp011) ─────────────────────────────────────────────────────

CF_CONFIG = {
    'K': 4, 'regime': 'extreme', 'n_expert': 500,
    'misclass': [0.05, 0.15, 0.25, 0.40],
    'tau_true': {0: 0.5, 1: 1.5},
}


# ── Shared utilities ─────────────────────────────────────────────────────────

def diff_in_means(Y, T, mask):
    t1, t0 = mask & (T == 1), mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


def get_counts(z_true, z_hat, subgroups, labels):
    k = len(labels)
    counts = np.zeros((k, 2, 2), dtype=int)
    for i, s in enumerate(labels):
        ms = subgroups == s
        for z in [0, 1]:
            mz = ms & (z_true == z)
            counts[i, z, 0] = (z_hat[mz] == 0).sum()
            counts[i, z, 1] = (z_hat[mz] == 1).sum()
    return counts


def compute_global_cm(counts):
    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    return C_gl


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


def invert_mixing(M, tau_obs, se_obs):
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    cond = np.linalg.cond(M)
    if cond > cond_threshold:
        return tau_obs.copy(), se_obs.copy()
    return invert_mixing(M, tau_obs, se_obs)


# ── CM estimators ────────────────────────────────────────────────────────────

def estimate_cm_mle(counts, labels):
    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            n = counts[i, z, :].sum()
            for zh in [0, 1]:
                C_s[z, zh] = (counts[i, z, zh] + 1) / (n + 2) if n > 0 else 0.5
        C[s] = C_s
    return C


def _dirmult_neg_ll_centered(alpha, counts_z, theta_g):
    if alpha < 1e-10:
        return 1e10
    a0 = alpha * theta_g[0]
    a1 = alpha * theta_g[1]
    if a0 < 1e-10 or a1 < 1e-10:
        return 1e10
    ll = 0.0
    for c0, c1 in counts_z:
        n = c0 + c1
        if n == 0:
            continue
        ll += gammaln(c0 + a0) + gammaln(c1 + a1) - gammaln(n + alpha)
        ll -= gammaln(a0) + gammaln(a1) - gammaln(alpha)
    return -ll


def estimate_cm_centered_eb(counts, labels, global_cm):
    k = len(labels)
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        theta_g = global_cm[z]
        cl = [(int(counts[i, z, 0]), int(counts[i, z, 1])) for i in range(k)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a, tg=theta_g: _dirmult_neg_ll_centered(a, cl, tg),
            bounds=(0.01, 5000), method='bounded')
        alpha_opt[z] = res.x
    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            a = alpha_opt[z]
            n = counts[i, z, :].sum()
            d = n + a
            C_s[z, 0] = (counts[i, z, 0] + a * global_cm[z, 0]) / d if d > 0 else 0.5
            C_s[z, 1] = (counts[i, z, 1] + a * global_cm[z, 1]) / d if d > 0 else 0.5
        C[s] = C_s
    return C, alpha_opt


# ── DIM DGP generator ────────────────────────────────────────────────────────

def generate_dim_data(cfg, seed):
    rng = np.random.RandomState(seed)
    k = cfg['k']
    labels = cfg['labels']
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(labels, N, p=[1.0 / k] * k)
    X = rng.randn(N, 5)
    tau = np.where(Z == 1, cfg['tau_z1'], cfg['tau_z0'])
    Y = 1 + tau * T + 0.5 * X[:, 0] + rng.randn(N)
    mc_rates = np.zeros(N)
    for i, s in enumerate(labels):
        mc_rates[S == s] = cfg['misclass'][i]
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)
    idx = rng.choice(N, min(cfg['n_expert'], N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True
    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


# ── DIM single MC run ────────────────────────────────────────────────────────

def run_dim_mc(config_name, cfg, seed):
    data = generate_dim_data(cfg, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = cfg['labels']
    tau_z0, tau_z1 = cfg['tau_z0'], cfg['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)
    global_cm = compute_global_cm(counts)
    M_gl = build_mixing_matrix(global_cm)

    C_mle = estimate_cm_mle(counts, labels)
    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}

    C_hb, alpha_opt = estimate_cm_centered_eb(counts, labels, global_cm)
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    base = {'config': config_name, 'estimator': 'DIM', 'mc_run': seed}
    rows = []

    for s in labels:
        sm = S == s
        th0_or, se0_or = diff_in_means(Y, T, sm & (Z == 0))
        th1_or, se1_or = diff_in_means(Y, T, sm & (Z == 1))
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))

        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        tgc, sgc = invert_mixing(M_gl, tau_obs, se_obs)
        tml, sml = invert_mixing(M_mle[s], tau_obs, se_obs)
        thb, shb = invert_mixing(M_hb[s], tau_obs, se_obs)

        def mr(method, th0, th1, se0, se1):
            return {**base, 'subgroup': s, 'method': method,
                    'tau_hat_z0': th0, 'tau_hat_z1': th1,
                    'se_z0': se0, 'se_z1': se1,
                    'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1}

        rows.append(mr('oracle', th0_or, th1_or, se0_or, se1_or))
        rows.append(mr('naive', th0_n, th1_n, se0_n, se1_n))
        rows.append(mr('global_corrected', tgc[0], tgc[1], sgc[0], sgc[1]))
        rows.append(mr('mle_laplace', tml[0], tml[1], sml[0], sml[1]))
        rows.append(mr('hb_centered_eb', thb[0], thb[1], shb[0], shb[1]))

    return rows


# ── CF DGP generator ─────────────────────────────────────────────────────────

def generate_cf_data(seed):
    cfg = CF_CONFIG
    K = cfg['K']
    rng = np.random.RandomState(seed)
    S = rng.randint(0, K, N)
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)
    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, cfg['misclass'][s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])
    tau = np.where(Z_true == 1, cfg['tau_true'][1], cfg['tau_true'][0])
    baseline = 1.0 + 0.5 * Z_true
    Y = baseline + tau * T + rng.normal(0, 1, N)
    expert_idx = rng.choice(N, cfg['n_expert'], replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True
    return Y, T, Z_true, Z_hat, S, expert_mask


def compute_cm_simple(z_true, z_hat):
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mz = z_true == z
        C[z, 0] = (z_hat[mz] == 0).sum() + 1
        C[z, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


# ── CF single MC run ─────────────────────────────────────────────────────────

def run_cf_mc(seed):
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LinearRegression, LogisticRegression

    cfg = CF_CONFIG
    K = cfg['K']
    Y, T, Z_true, Z_hat, S, expert_mask = generate_cf_data(seed)

    X_hat = np.column_stack([Z_hat, S]).astype(float)
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200, min_samples_leaf=20, random_state=42)
    cf.fit(Y, T, X=X_hat, W=None)

    z_exp, zh_exp, s_exp = Z_true[expert_mask], Z_hat[expert_mask], S[expert_mask]

    # Global CM
    C_global = compute_cm_simple(z_exp, zh_exp)
    M_global = build_mixing_matrix(C_global)

    # Per-subgroup centered EB CMs
    labels = list(range(K))
    counts = get_counts(z_exp, zh_exp, s_exp, labels)
    global_cm = compute_global_cm(counts)
    C_eb, _ = estimate_cm_centered_eb(counts, labels, global_cm)
    M_eb = {s: build_mixing_matrix(C_eb[s]) for s in labels}

    rows = []
    X_eval_all = np.array([[z, s] for s in range(K) for z in [0, 1]], dtype=float)
    cate_vals = cf.effect(X_eval_all).flatten()
    try:
        inf = cf.effect_inference(X_eval_all)
        se_raw = inf.stderr
        if hasattr(se_raw, 'values'):
            se_raw = se_raw.values
        se_vals = np.asarray(se_raw).flatten()
    except Exception:
        se_vals = np.full(len(X_eval_all), np.nan)

    cate, cate_se = {}, {}
    idx = 0
    for s_val in range(K):
        for z in [0, 1]:
            cate[(z, s_val)] = float(cate_vals[idx])
            cate_se[(z, s_val)] = float(se_vals[idx])
            idx += 1

    rows = []
    for s_val in range(K):
        tau_obs = np.array([cate[(0, s_val)], cate[(1, s_val)]])
        se_obs = np.array([cate_se[(0, s_val)], cate_se[(1, s_val)]])

        tau_gl, se_gl = invert_mixing_safe(M_global, tau_obs, se_obs)
        tau_eb, se_eb = invert_mixing_safe(M_eb[s_val], tau_obs, se_obs)

        base = {'config': 'CF_K4_extreme', 'estimator': 'CausalForest',
                'mc_run': seed, 'subgroup': s_val}
        for z in [0, 1]:
            tau_true_val = cfg['tau_true'][z]
            rows.append({**base, 'method': 'naive_forest', 'z_level': z,
                         'tau_hat': tau_obs[z], 'se': se_obs[z], 'tau_true': tau_true_val})
            rows.append({**base, 'method': 'global_forest', 'z_level': z,
                         'tau_hat': tau_gl[z], 'se': se_gl[z], 'tau_true': tau_true_val})
            rows.append({**base, 'method': 'ec_hte_forest', 'z_level': z,
                         'tau_hat': tau_eb[z], 'se': se_eb[z], 'tau_true': tau_true_val})

    return rows


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_dim(df):
    rows = []
    for keys, grp in df.groupby(['config', 'subgroup', 'method']):
        config, subgroup, method = keys
        row = {'config': config, 'subgroup': subgroup, 'method': method, 'n_mc': len(grp)}
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'].values - grp[f'tau_true_{z}'].values
            bias = err.mean()
            rmse = np.sqrt((err ** 2).mean())
            ci_lo = grp[f'tau_hat_{z}'].values - 1.96 * grp[f'se_{z}'].values
            ci_hi = grp[f'tau_hat_{z}'].values + 1.96 * grp[f'se_{z}'].values
            cov = ((ci_lo <= grp[f'tau_true_{z}'].values) &
                   (grp[f'tau_true_{z}'].values <= ci_hi)).mean()
            row[f'abs_bias_{z}'] = abs(bias)
            row[f'rmse_{z}'] = rmse
            row[f'coverage_{z}'] = cov
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_cf(df):
    rows = []
    for keys, grp in df.groupby(['config', 'subgroup', 'method', 'z_level']):
        config, subgroup, method, z_level = keys
        err = grp['tau_hat'].values - grp['tau_true'].values
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        ci_lo = grp['tau_hat'].values - 1.96 * grp['se'].values
        ci_hi = grp['tau_hat'].values + 1.96 * grp['se'].values
        cov = ((ci_lo <= grp['tau_true'].values) &
               (grp['tau_true'].values <= ci_hi)).mean()
        rows.append({'config': config, 'subgroup': subgroup, 'method': method,
                     'z_level': z_level, 'abs_bias': abs(bias), 'rmse': rmse,
                     'coverage': cov, 'n_mc': len(grp)})
    return pd.DataFrame(rows)


def table1_summary(dim_agg, cf_agg):
    print("\n" + "=" * 80)
    print("TABLE 1 SUMMARY (averages over subgroups and z-levels)")
    print("=" * 80)

    for config_name in DIM_CONFIGS:
        sub = dim_agg[dim_agg['config'] == config_name]
        methods = ['naive', 'global_corrected', 'mle_laplace', 'hb_centered_eb']
        print(f"\n{config_name}:")
        print(f"  {'Method':<20s} {'|Bias|':>8s} {'RMSE':>8s} {'Cov':>8s}")
        for m in methods:
            ms = sub[sub['method'] == m]
            if ms.empty:
                continue
            ab = ms[['abs_bias_z0', 'abs_bias_z1']].values.mean()
            rm = ms[['rmse_z0', 'rmse_z1']].values.mean()
            cv = ms[['coverage_z0', 'coverage_z1']].values.mean()
            print(f"  {m:<20s} {ab:8.3f} {rm:8.3f} {cv:8.2f}")

    if cf_agg is not None and not cf_agg.empty:
        print(f"\nCF_K4_extreme:")
        print(f"  {'Method':<20s} {'|Bias|':>8s} {'RMSE':>8s} {'Cov':>8s}")
        for m in ['naive_forest', 'global_forest', 'ec_hte_forest']:
            ms = cf_agg[cf_agg['method'] == m]
            if ms.empty:
                continue
            ab = ms['abs_bias'].mean()
            rm = ms['rmse'].mean()
            cv = ms['coverage'].mean()
            print(f"  {m:<20s} {ab:8.3f} {rm:8.3f} {cv:8.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse; _p = argparse.ArgumentParser(); _p.add_argument("n_mc", type=int, nargs="?", default=N_MC); _p.add_argument("--skip-cf", action="store_true"); _args = _p.parse_args(); n_mc = _args.n_mc
    skip_cf = _args.skip_cf

    print(f"exp_centered_eb_table1 | N={N}, N_MC={n_mc}")
    print(f"DIM configs: {list(DIM_CONFIGS.keys())}")
    if not skip_cf:
        print(f"CF config: K={CF_CONFIG['K']} {CF_CONFIG['regime']} n={CF_CONFIG['n_expert']}")
    print()

    # ── DIM experiments ──
    dim_rows = []
    t0 = time.time()
    total_dim = len(DIM_CONFIGS) * n_mc
    count = 0

    for config_name, cfg in DIM_CONFIGS.items():
        for mc in range(n_mc):
            rows = run_dim_mc(config_name, cfg, seed=mc)
            dim_rows.extend(rows)
            count += 1
            if count % max(1, total_dim // 10) == 0:
                el = time.time() - t0
                eta = el / count * (total_dim - count)
                print(f"  DIM [{count}/{total_dim}] {el:.0f}s elapsed, ETA {eta:.0f}s | {config_name}")

    dim_elapsed = time.time() - t0
    print(f"DIM done: {dim_elapsed:.1f}s")

    df_dim = pd.DataFrame(dim_rows)
    dim_agg = aggregate_dim(df_dim)

    # ── CF experiment ──
    cf_agg = None
    if not skip_cf:
        cf_rows = []
        t1 = time.time()
        for mc in range(n_mc):
            try:
                rows = run_cf_mc(seed=mc)
                cf_rows.extend(rows)
            except Exception as e:
                print(f"  CF mc={mc} failed: {e}")
            if (mc + 1) % max(1, n_mc // 10) == 0:
                el = time.time() - t1
                eta = el / (mc + 1) * (n_mc - mc - 1)
                print(f"  CF [{mc+1}/{n_mc}] {el:.0f}s elapsed, ETA {eta:.0f}s")

        cf_elapsed = time.time() - t1
        print(f"CF done: {cf_elapsed:.1f}s")
        df_cf = pd.DataFrame(cf_rows)
        cf_agg = aggregate_cf(df_cf)

    # ── Save and print ──
    os.makedirs('results', exist_ok=True)
    dim_agg.to_csv('results/exp_centered_eb_table1_dim.csv', index=False)
    if cf_agg is not None:
        cf_agg.to_csv('results/exp_centered_eb_table1_cf.csv', index=False)

    table1_summary(dim_agg, cf_agg)

    total_elapsed = time.time() - t0
    print(f"\nTotal: {total_elapsed:.1f}s")
    print(f"Saved: results/exp_centered_eb_table1_dim.csv")
    if cf_agg is not None:
        print(f"Saved: results/exp_centered_eb_table1_cf.csv")
