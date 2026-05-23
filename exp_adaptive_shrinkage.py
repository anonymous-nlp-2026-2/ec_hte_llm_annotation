#!/usr/bin/env python3
"""exp_adaptive_shrinkage.py — Adaptive Shrinkage for EC-HTE

Tests adaptive shrinkage to make EC-HTE competitive with Global correction
under moderate heterogeneity while preserving extreme-heterogeneity gains.

Key insight: original HB uses symmetric Dir(α/2, α/2), shrinking toward (0.5,0.5).
Actual CMs are far from uniform (e.g. 0.9/0.1). Centering the Dirichlet on the
global CM lets the EB pick large α under moderate heterogeneity (≈ Global correction)
and small α under extreme heterogeneity (≈ per-subgroup EC-HTE).

Methods:
  oracle           — true Z labels
  naive            — raw Z_hat labels
  global_corrected — pooled CM correction
  hb_original      — original EB with symmetric Dirichlet
  hb_centered      — EB with Dirichlet centered on global CM
  pretest          — LR test for CM homogeneity → Global or centered EB

Output: results/exp_adaptive_shrinkage.csv, results/exp_adaptive_shrinkage_raw.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

warnings.filterwarnings('ignore')

N = 5000
N_MC = 100

CONFIGS = {
    'K2_moderate': {
        'k': 2, 'misclass': [0.10, 0.15], 'tau_z1': 0.3, 'tau_z0': 0.15,
        'n_expert': 250, 'labels': ['A', 'B'],
    },
    'K2_extreme': {
        'k': 2, 'misclass': [0.05, 0.25], 'tau_z1': 0.5, 'tau_z0': 0.1,
        'n_expert': 250, 'labels': ['A', 'B'],
    },
    'K4_moderate': {
        'k': 4, 'misclass': [0.10, 0.12, 0.14, 0.16], 'tau_z1': 0.3, 'tau_z0': 0.15,
        'n_expert': 500, 'labels': ['A', 'B', 'C', 'D'],
    },
    'K4_extreme': {
        'k': 4, 'misclass': [0.05, 0.10, 0.15, 0.25], 'tau_z1': 0.5, 'tau_z0': 0.1,
        'n_expert': 500, 'labels': ['A', 'B', 'C', 'D'],
    },
}


# ── DGP ──────────────────────────────────────────────────────────────────────

def generate_data(cfg, seed):
    rng = np.random.RandomState(seed)
    k = cfg['k']
    labels = cfg['labels']
    n_expert = cfg['n_expert']
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
    idx = rng.choice(N, min(n_expert, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True
    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


# ── Utilities ────────────────────────────────────────────────────────────────

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


def compute_global_cm(counts):
    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    return C_gl


# ── Original HB EB (symmetric Dirichlet) ─────────────────────────────────────

def _dirmult_neg_ll_symmetric(alpha, counts_z):
    a = alpha / 2.0
    if a < 1e-10:
        return 1e10
    ll = 0.0
    for c0, c1 in counts_z:
        n = c0 + c1
        if n == 0:
            continue
        ll += gammaln(c0 + a) + gammaln(c1 + a) - gammaln(n + 2 * a)
        ll -= 2 * gammaln(a) - gammaln(2 * a)
    return -ll


def estimate_cm_hb_eb_original(counts, labels):
    k = len(labels)
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        cl = [(int(counts[i, z, 0]), int(counts[i, z, 1])) for i in range(k)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a: _dirmult_neg_ll_symmetric(a, cl),
            bounds=(0.01, 500), method='bounded')
        alpha_opt[z] = res.x
    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            a = alpha_opt[z] / 2.0
            n = counts[i, z, :].sum()
            d = n + 2 * a
            C_s[z, 0] = (counts[i, z, 0] + a) / d if d > 0 else 0.5
            C_s[z, 1] = (counts[i, z, 1] + a) / d if d > 0 else 0.5
        C[s] = C_s
    return C


# ── Centered EB (Dirichlet centered on global CM) ────────────────────────────

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


# ── Pre-test selector ────────────────────────────────────────────────────────

def cm_homogeneity_lr_test(counts):
    k = counts.shape[0]
    pooled = counts.sum(axis=0)
    lr_stat = 0.0
    df = 0
    for z in [0, 1]:
        n_pooled = pooled[z, :].sum()
        if n_pooled == 0:
            continue
        p_pooled = pooled[z, :].astype(float) / n_pooled
        for i in range(k):
            n_i = counts[i, z, :].sum()
            if n_i == 0:
                continue
            for j in [0, 1]:
                if counts[i, z, j] > 0 and p_pooled[j] > 0:
                    expected = n_i * p_pooled[j]
                    lr_stat += 2.0 * counts[i, z, j] * np.log(counts[i, z, j] / expected)
        df += (k - 1)
    if df == 0:
        return 0.0, 1.0, 0
    p_value = chi2.sf(max(lr_stat, 0), df)
    return lr_stat, p_value, df


def estimate_cm_pretest(counts, labels, global_cm, test_alpha=0.05):
    _, p_value, _ = cm_homogeneity_lr_test(counts)
    if p_value > test_alpha:
        C = {s: global_cm.copy() for s in labels}
        return C, 'global', p_value
    else:
        C, _ = estimate_cm_centered_eb(counts, labels, global_cm)
        return C, 'stratified', p_value


# ── Single MC run ────────────────────────────────────────────────────────────

def run_one_mc(config_name, cfg, seed):
    data = generate_data(cfg, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = cfg['labels']
    tau_z0, tau_z1 = cfg['tau_z0'], cfg['tau_z1']
    counts = get_counts(Z[em], Z_hat[em], S[em], labels)

    global_cm = compute_global_cm(counts)
    M_gl = build_mixing_matrix(global_cm)

    C_hb_orig = estimate_cm_hb_eb_original(counts, labels)
    M_hb_orig = {s: build_mixing_matrix(C_hb_orig[s]) for s in labels}

    C_centered, _ = estimate_cm_centered_eb(counts, labels, global_cm)
    M_centered = {s: build_mixing_matrix(C_centered[s]) for s in labels}

    C_pretest, pt_decision, pt_pval = estimate_cm_pretest(counts, labels, global_cm)
    M_pretest = {s: build_mixing_matrix(C_pretest[s]) for s in labels}

    base = {'config': config_name, 'mc_run': seed}
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
        tho, sho = invert_mixing(M_hb_orig[s], tau_obs, se_obs)
        thc, shc = invert_mixing(M_centered[s], tau_obs, se_obs)
        tpt, spt = invert_mixing(M_pretest[s], tau_obs, se_obs)

        def mr(method, th0, th1, se0, se1, **extra):
            return {**base, 'subgroup': s, 'method': method,
                    'tau_hat_z0': th0, 'tau_hat_z1': th1,
                    'se_z0': se0, 'se_z1': se1,
                    'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1, **extra}

        rows.append(mr('oracle', th0_or, th1_or, se0_or, se1_or))
        rows.append(mr('naive', th0_n, th1_n, se0_n, se1_n))
        rows.append(mr('global_corrected', tgc[0], tgc[1], sgc[0], sgc[1]))
        rows.append(mr('hb_original', tho[0], tho[1], sho[0], sho[1]))
        rows.append(mr('hb_centered', thc[0], thc[1], shc[0], shc[1]))
        rows.append(mr('pretest', tpt[0], tpt[1], spt[0], spt[1],
                        pretest_decision=pt_decision, pretest_pval=pt_pval))

    return rows


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_results(df):
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
            row[f'bias_{z}'] = bias
            row[f'abs_bias_{z}'] = abs(bias)
            row[f'rmse_{z}'] = rmse
            row[f'coverage_{z}'] = cov
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else N_MC

    print(f"exp_adaptive_shrinkage | N={N}, N_MC={n_mc}")
    print(f"Configs: {list(CONFIGS.keys())}\n")

    all_rows = []
    t0 = time.time()
    total = len(CONFIGS) * n_mc
    count = 0

    for config_name, cfg in CONFIGS.items():
        for mc in range(n_mc):
            rows = run_one_mc(config_name, cfg, seed=mc)
            all_rows.extend(rows)
            count += 1
            if count % max(1, total // 20) == 0:
                el = time.time() - t0
                eta = el / count * (total - count) if count > 0 else 0
                print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s | {config_name} mc={mc}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate_results(df_raw)

    os.makedirs('results', exist_ok=True)
    df_raw.to_csv('results/exp_adaptive_shrinkage_raw.csv', index=False)
    df_agg.to_csv('results/exp_adaptive_shrinkage.csv', index=False)

    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    method_order = ['oracle', 'naive', 'global_corrected', 'hb_original', 'hb_centered', 'pretest']

    for config_name in CONFIGS:
        print(f"\n{'=' * 140}")
        cfg = CONFIGS[config_name]
        print(f"Config: {config_name} | K={cfg['k']} misclass={cfg['misclass']} "
              f"tau_z0={cfg['tau_z0']} tau_z1={cfg['tau_z1']} n_expert={cfg['n_expert']}")
        print(f"{'=' * 140}")

        sub = df_agg[df_agg['config'] == config_name]
        avg = sub.groupby('method').agg({
            'abs_bias_z0': 'mean', 'abs_bias_z1': 'mean',
            'rmse_z0': 'mean', 'rmse_z1': 'mean',
            'coverage_z0': 'mean', 'coverage_z1': 'mean',
        }).reset_index()
        avg.columns = ['method', '|bias|_z0', '|bias|_z1', 'RMSE_z0', 'RMSE_z1', 'cov_z0', 'cov_z1']
        avg['_order'] = avg['method'].map({m: i for i, m in enumerate(method_order)})
        avg = avg.sort_values('_order').drop('_order', axis=1)
        print(avg.to_string(index=False))

    if 'pretest_decision' in df_raw.columns:
        print(f"\n{'=' * 140}")
        print("Pre-test decisions (fraction choosing 'stratified'):")
        for config_name in CONFIGS:
            pt_rows = df_raw[(df_raw['config'] == config_name) &
                             (df_raw['method'] == 'pretest')].drop_duplicates(subset=['mc_run'])
            if not pt_rows.empty:
                frac_strat = (pt_rows['pretest_decision'] == 'stratified').mean()
                print(f"  {config_name}: {frac_strat:.1%} stratified")

    print(f"\nSaved: results/exp_adaptive_shrinkage.csv ({len(df_agg)} rows)")
    print(f"Saved: results/exp_adaptive_shrinkage_raw.csv ({len(df_raw)} rows)")
