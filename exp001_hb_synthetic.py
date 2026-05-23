#!/usr/bin/env python3
"""
exp001_hb_synthetic.py - Hierarchical Bayesian EC-HTE Synthetic Validation (exp-001)

Compares confusion matrix estimators for CATE correction under heterogeneous
misclassification across subgroups:
  Oracle, Naive, Global-corrected, MLE (Laplace), HB Dirichlet (partial pooling)

Input: Synthetic DGP parameters (regimes, expert budgets, subgroup counts)
Output: results/exp001_hb_results.csv, results/exp001_hb_diagnostics.csv
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

N = 5000
SUBGROUP_LABELS = {2: ['A', 'B'], 4: ['A', 'B', 'C', 'D']}

REGIMES = {
    2: {
        'extreme':     {'misclass': [0.25, 0.05], 'tau_z1': 0.5, 'tau_z0': 0.1},
        'moderate':    {'misclass': [0.15, 0.10], 'tau_z1': 0.3, 'tau_z0': 0.15},
        'homogeneous': {'misclass': [0.12, 0.12], 'tau_z1': 0.25, 'tau_z0': 0.20},
    },
    4: {
        'extreme':     {'misclass': [0.30, 0.20, 0.10, 0.03], 'tau_z1': 0.5, 'tau_z0': 0.1},
        'moderate':    {'misclass': [0.20, 0.15, 0.10, 0.05], 'tau_z1': 0.3, 'tau_z0': 0.15},
        'homogeneous': {'misclass': [0.12, 0.12, 0.12, 0.12], 'tau_z1': 0.25, 'tau_z0': 0.20},
    },
}

DEFAULT_N_EXPERTS = [50, 100, 250, 500, 1000, 2000]
DEFAULT_K_LIST = [2, 4]


# ── DGP ───────────────────────────────────────────────────────────────────────

def generate_data(regime_params, n_expert, k, seed):
    rng = np.random.RandomState(seed)
    labels = SUBGROUP_LABELS[k]
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(labels, N, p=[1.0 / k] * k)
    X = rng.randn(N, 5)
    tau = np.where(Z == 1, regime_params['tau_z1'], regime_params['tau_z0'])
    Y = 1 + tau * T + 0.5 * X[:, 0] + rng.randn(N)
    mc_rates = np.zeros(N)
    for i, s in enumerate(labels):
        mc_rates[S == s] = regime_params['misclass'][i]
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)
    idx = rng.choice(N, min(n_expert, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True
    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


# ── Helpers ───────────────────────────────────────────────────────────────────

def diff_in_means(Y, T, mask):
    t1, t0 = mask & (T == 1), mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


def get_counts(z_true, z_hat, subgroups, labels):
    """Confusion matrix counts per subgroup: (k, 2, 2) = [subgroup, true_z, z_hat]."""
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
    """C[z, z_hat] → M[z_hat, z] via Bayes rule (assuming P(Z=0)=P(Z=1)=0.5).

    NOTE: This assumes P(Z=0)=P(Z=1)=0.5, which is correct for this experiment's
    DGP. For general P(Z), use the version in exp007 that takes p_z and p_z_hat.
    """
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


def safe_invert_correction(M_s, tau_obs, cond_threshold=100, fallback='naive'):
    """Invert mixing matrix with condition number guard.

    When cond(M_s) > threshold, matrix inversion is numerically unstable.
    Falls back to naive (no correction) to avoid catastrophic amplification.
    """
    cond = np.linalg.cond(M_s)
    if cond > cond_threshold:
        return tau_obs.copy()
    return np.linalg.solve(M_s, tau_obs)


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    """Like invert_mixing but with condition number guard."""
    cond = np.linalg.cond(M)
    if cond > cond_threshold:
        return tau_obs.copy(), se_obs.copy()
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


# ── Confusion Matrix Estimators ───────────────────────────────────────────────

def estimate_cm_mle(counts, labels):
    """Per-subgroup MLE with Laplace smoothing (add-1)."""
    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            n = counts[i, z, :].sum()
            for zh in [0, 1]:
                C_s[z, zh] = (counts[i, z, zh] + 1) / (n + 2) if n > 0 else 0.5
        C[s] = C_s
    return C


def compute_global_cm(counts):
    """Pooled confusion matrix with Laplace smoothing."""
    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    return C_gl


def _dirmult_neg_ll(alpha, counts_z, theta_g):
    """Negative log marginal likelihood for centered Dirichlet-Multinomial."""
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


def estimate_cm_hb_eb(counts, labels, global_cm):
    """Centered Empirical Bayes: Dirichlet centered on global pooled CM."""
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
            lambda a, tg=theta_g: _dirmult_neg_ll(a, cl, tg),
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

    diag = {'rhat_max': np.nan, 'ess_min': np.nan, 'divergent_count': 0,
            'runtime_seconds': 0.0, 'hb_method': 'eb',
            'alpha_0_z0': alpha_opt[0], 'alpha_0_z1': alpha_opt[1]}
    return C, diag


def _split_rhat(chains):
    """Split R-hat for a scalar parameter. chains: (n_chains, n_draws)."""
    parts = []
    for c in range(chains.shape[0]):
        half = chains.shape[1] // 2
        parts.append(chains[c, :half])
        parts.append(chains[c, half:2 * half])
    parts = np.array(parts)
    m = parts.shape[0]
    n = parts.shape[1]
    chain_means = parts.mean(axis=1)
    W = np.mean(np.var(parts, axis=1, ddof=1))
    B = n * np.var(chain_means, ddof=1)
    if W < 1e-20:
        return 1.0
    var_hat = (1 - 1.0 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def _bulk_ess(chains):
    """Approximate bulk ESS via initial positive sequence. chains: (n_chains, n_draws)."""
    x = chains.ravel()
    n = len(x)
    mu = x.mean()
    xc = x - mu
    max_lag = min(n // 2, 300)
    c0 = np.dot(xc, xc)
    if c0 < 1e-20:
        return float(n)
    tau = 1.0
    for lag in range(1, max_lag - 1, 2):
        r1 = np.dot(xc[:n - lag], xc[lag:]) / c0
        r2 = np.dot(xc[:n - lag - 1], xc[lag + 1:]) / c0 if lag + 1 < n else 0
        pair = r1 + r2
        if pair < 0:
            break
        tau += 2 * pair
    return float(n / tau)


def _fast_diagnostics(alpha_arr, theta_arr, labels):
    """Compute R-hat and ESS over all parameters without arviz."""
    rhats, esss = [], []
    for d in range(alpha_arr.shape[2]):
        rhats.append(_split_rhat(alpha_arr[:, :, d]))
        esss.append(_bulk_ess(alpha_arr[:, :, d]))
    k = len(labels)
    for i in range(k):
        for z in [0, 1]:
            for zh in range(theta_arr.shape[4]):
                ch = theta_arr[:, :, i, z, zh]
                rhats.append(_split_rhat(ch))
                esss.append(_bulk_ess(ch))
    return max(rhats), min(esss)


def _log_target_alpha(alpha, theta_z, theta_g, hp_a0=3.0, hp_b0=1.0):
    """Log posterior for alpha_0[z] with centered Dirichlet prior."""
    if alpha < 1e-10:
        return -1e10
    a0 = alpha * theta_g[0]
    a1 = alpha * theta_g[1]
    if a0 < 1e-10 or a1 < 1e-10:
        return -1e10
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - gammaln(a0) - gammaln(a1))
    lp += (a0 - 1.0) * np.sum(np.log(np.clip(theta_z[:, 0], 1e-300, None)))
    lp += (a1 - 1.0) * np.sum(np.log(np.clip(theta_z[:, 1], 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, labels, global_cm, seed=42, n_iter=4000, n_warmup=2000, n_chains=4,
                         hp_a0=3.0, hp_b0=1.0):
    """Gibbs sampler for HB Dirichlet-Multinomial (conjugate + adaptive MH for alpha_0).

    hp_a0, hp_b0: Gamma hyperprior shape/rate on alpha_0 (default Gamma(3,1)).
    """
    k = len(labels)
    chain_inits_pool = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                        np.array([0.5, 0.5]), np.array([10.0, 10.0])]
    chain_inits = chain_inits_pool[:n_chains]
    all_alpha, all_theta = [], []
    t0 = time.time()

    for chain in range(n_chains):
        rng = np.random.RandomState(seed * 31 + chain * 7919 + 1)
        alpha = chain_inits[chain].copy()
        theta = np.ones((k, 2, 2)) * 0.5
        mh_scale = np.array([0.3, 0.3])
        accept_ct = np.array([0, 0])
        attempt_ct = np.array([0, 0])

        alpha_samples = np.zeros((n_iter, 2))
        theta_samples = np.zeros((n_iter, k, 2, 2))

        for it in range(n_warmup + n_iter):
            for z in [0, 1]:
                post_a = counts[:, z, :].astype(float) + alpha[z] * global_cm[z]
                post_a = np.maximum(post_a, 1e-6)
                g = rng.gamma(post_a)
                g_sum = g.sum(axis=1, keepdims=True)
                theta[:, z, :] = g / np.maximum(g_sum, 1e-300)

            for z in [0, 1]:
                x_prop = np.log(alpha[z]) + rng.normal(0, mh_scale[z])
                a_prop = np.exp(x_prop)
                log_r = (_log_target_alpha(a_prop, theta[:, z, :], global_cm[z], hp_a0, hp_b0) -
                         _log_target_alpha(alpha[z], theta[:, z, :], global_cm[z], hp_a0, hp_b0))
                accepted = np.log(rng.rand()) < log_r
                if accepted:
                    alpha[z] = a_prop
                if it < n_warmup:
                    accept_ct[z] += int(accepted)
                    attempt_ct[z] += 1

            if it < n_warmup and it > 0 and it % 100 == 0:
                for z in [0, 1]:
                    rate = accept_ct[z] / max(attempt_ct[z], 1)
                    if rate < 0.2:
                        mh_scale[z] *= 0.7
                    elif rate > 0.5:
                        mh_scale[z] *= 1.3
                    accept_ct[z] = 0
                    attempt_ct[z] = 0

            if it >= n_warmup:
                idx = it - n_warmup
                alpha_samples[idx] = alpha.copy()
                theta_samples[idx] = theta.copy()

        all_alpha.append(alpha_samples)
        all_theta.append(theta_samples)

    runtime = time.time() - t0
    alpha_arr = np.stack(all_alpha)
    theta_arr = np.stack(all_theta)

    C = {}
    for i, s in enumerate(labels):
        C[s] = theta_arr[:, :, i, :, :].mean(axis=(0, 1))

    rhat_max, ess_min = _fast_diagnostics(alpha_arr, theta_arr, labels)

    diag = {'rhat_max': rhat_max, 'ess_min': ess_min, 'divergent_count': 0,
            'runtime_seconds': runtime, 'hb_method': 'gibbs'}
    return C, diag


def estimate_cm_hb_mcmc(counts, labels, seed=42, draws=1000, tune=500, chains=2):
    """Full Bayesian HB with PyMC MCMC."""
    k = len(labels)
    n_per_sz = counts.sum(axis=2)

    with pm.Model():
        alpha_0 = pm.Gamma('alpha_0', alpha=2, beta=1, shape=2)
        for z in [0, 1]:
            for i, s in enumerate(labels):
                nsz = int(n_per_sz[i, z])
                if nsz == 0:
                    continue
                a = pt.stack([alpha_0[z] / 2, alpha_0[z] / 2])
                theta = pm.Dirichlet(f'theta_{s}_z{z}', a=a)
                pm.Multinomial(f'obs_{s}_z{z}', n=nsz, p=theta,
                               observed=counts[i, z, :].astype('int32'))
        t0 = time.time()
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          random_seed=seed, progressbar=False)
        runtime = time.time() - t0

    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            vn = f'theta_{s}_z{z}'
            if vn in trace.posterior:
                C_s[z, :] = trace.posterior[vn].values.mean(axis=(0, 1))
            else:
                C_s[z, :] = 0.5
        C[s] = C_s

    all_vars = list(trace.posterior.data_vars)
    try:
        summ = az.summary(trace, var_names=all_vars)
        rhat_max = float(summ['r_hat'].max())
        ess_min = float(summ['ess_bulk'].min())
    except Exception:
        rhat_max = ess_min = np.nan
    try:
        divergent = int(trace.sample_stats['diverging'].values.sum())
    except Exception:
        divergent = -1

    diag = {'rhat_max': rhat_max, 'ess_min': ess_min,
            'divergent_count': divergent, 'runtime_seconds': runtime,
            'hb_method': 'mcmc'}
    return C, diag


# ── Single MC Run ─────────────────────────────────────────────────────────────

def run_one_mc(regime_name, regime_params, n_expert, k, seed, hb_method='eb'):
    data = generate_data(regime_params, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]
    tau_z0, tau_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)
    global_cm = compute_global_cm(counts)

    C_mle = estimate_cm_mle(counts, labels)

    if hb_method == 'gibbs':
        C_hb, hb_diag = estimate_cm_hb_gibbs(counts, labels, global_cm, seed=seed)
    elif hb_method == 'mcmc' and HAS_PYMC:
        try:
            C_hb, hb_diag = estimate_cm_hb_mcmc(counts, labels, seed=seed)
        except Exception:
            C_hb, hb_diag = estimate_cm_hb_eb(counts, labels, global_cm)
            hb_diag['hb_method'] = 'eb_fallback'
    else:
        C_hb, hb_diag = estimate_cm_hb_eb(counts, labels, global_cm)

    # Global CM (pool all expert labels, Laplace)
    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5

    M_gl = build_mixing_matrix(C_gl)
    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

    base = {'regime': regime_name, 'n_expert': n_expert, 'k_subgroups': k, 'mc_run': seed}
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
        rows.append(mr('hb_dirichlet', thb[0], thb[1], shb[0], shb[1]))

    return rows, {**base, **hb_diag}


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(df):
    rows = []
    grp_cols = ['regime', 'n_expert', 'k_subgroups', 'subgroup', 'method']
    for keys, grp in df.groupby(grp_cols):
        regime, n_expert, k, subgroup, method = keys
        row = {'regime': regime, 'n_expert': int(n_expert), 'k_subgroups': int(k),
               'subgroup': subgroup, 'method': method}
        n_mc = len(grp)
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'].values - grp[f'tau_true_{z}'].values
            bias = err.mean()
            mse = (err ** 2).mean()
            rmse = np.sqrt(mse)
            ci_lo = grp[f'tau_hat_{z}'].values - 1.96 * grp[f'se_{z}'].values
            ci_hi = grp[f'tau_hat_{z}'].values + 1.96 * grp[f'se_{z}'].values
            cov = ((ci_lo <= grp[f'tau_true_{z}'].values) &
                   (grp[f'tau_true_{z}'].values <= ci_hi)).mean()
            row[f'bias_{z}'] = bias
            row[f'rmse_{z}'] = rmse
            row[f'coverage_{z}'] = cov
            row[f'bias_{z}_se'] = err.std() / np.sqrt(n_mc)
            mse_se = (err ** 2).std() / np.sqrt(n_mc)
            row[f'rmse_{z}_se'] = mse_se / (2 * rmse) if rmse > 1e-10 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(df_agg, df_diag):
    pd.set_option('display.float_format', '{:.4f}'.format)

    # MCMC / Gibbs diagnostics
    if 'hb_method' in df_diag.columns:
        for method_label in ['gibbs', 'mcmc']:
            mrows = df_diag[df_diag['hb_method'] == method_label]
            if mrows.empty:
                continue
            print(f"\n── {method_label.upper()} Diagnostics ──────────────────────────────")
            print(f"  Runs:           {len(mrows)}")
            rh = mrows['rhat_max'].dropna()
            es = mrows['ess_min'].dropna()
            if not rh.empty:
                print(f"  R-hat max:      {rh.max():.4f} (target < 1.01)")
            if not es.empty:
                print(f"  ESS min:        {es.min():.0f} (target > 200)")
            print(f"  Divergent total:{int(mrows['divergent_count'].sum())}")
            print(f"  Runtime median: {mrows['runtime_seconds'].median():.2f}s")

    # Per-config HB vs MLE comparison
    print("\n── HB vs MLE RMSE Comparison (subgroup-level, averaged) ──")
    for k in sorted(df_agg['k_subgroups'].unique()):
        print(f"\n  K = {int(k)} subgroups")
        for regime in ['extreme', 'moderate', 'homogeneous']:
            sub = df_agg[(df_agg['k_subgroups'] == k) & (df_agg['regime'] == regime)]
            if sub.empty:
                continue
            print(f"    {regime}:")
            print(f"    {'n_exp':>6s}  {'oracle':>8s}  {'naive':>8s}  {'global':>8s}  "
                  f"{'mle':>8s}  {'hb':>8s}  {'HB vs MLE':>10s}")
            for ne in sorted(sub['n_expert'].unique()):
                vals = {}
                for m in ['oracle', 'naive', 'global_corrected', 'mle_laplace', 'hb_dirichlet']:
                    ms = sub[(sub['method'] == m) & (sub['n_expert'] == ne)]
                    if ms.empty:
                        vals[m] = np.nan
                    else:
                        vals[m] = np.sqrt((ms['rmse_z0'].values ** 2 +
                                           ms['rmse_z1'].values ** 2).mean())
                mle_v, hb_v = vals['mle_laplace'], vals['hb_dirichlet']
                pct = (mle_v - hb_v) / mle_v * 100 if mle_v > 1e-10 else 0
                tag = f"{pct:+.1f}%"
                print(f"    {int(ne):6d}  {vals['oracle']:8.4f}  {vals['naive']:8.4f}  "
                      f"{vals['global_corrected']:8.4f}  {mle_v:8.4f}  {hb_v:8.4f}  {tag:>10s}")


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(regime_names, n_experts, k_list, n_mc,
              hb_method='eb', mcmc_diag_runs=5, verbose=True):
    all_rows, all_diag = [], []
    total = sum(1 for k in k_list for r in regime_names if r in REGIMES[k]) * len(n_experts) * n_mc
    count = 0
    t0 = time.time()

    for k in k_list:
        for rn in regime_names:
            if rn not in REGIMES[k]:
                continue
            rp = REGIMES[k][rn]
            for ne in n_experts:
                for mc in range(n_mc):
                    use = hb_method
                    if hb_method == 'auto':
                        use = 'mcmc' if mc < mcmc_diag_runs else 'eb'
                    try:
                        rows, diag = run_one_mc(rn, rp, ne, k, seed=mc, hb_method=use)
                        all_rows.extend(rows)
                        all_diag.append(diag)
                    except Exception as e:
                        if verbose:
                            print(f"  WARN: k={k} {rn} ne={ne} mc={mc} failed: {e}")
                        all_diag.append({
                            'regime': rn, 'n_expert': ne, 'k_subgroups': k,
                            'mc_run': mc, 'hb_method': 'error',
                            'rhat_max': np.nan, 'ess_min': np.nan,
                            'divergent_count': -1, 'runtime_seconds': 0.0})
                    count += 1
                    if verbose and count % max(1, total // 20) == 0:
                        el = time.time() - t0
                        eta = el / count * (total - count)
                        print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s | "
                              f"k={k} {rn} ne={ne} mc={mc}")

    if verbose:
        print(f"  Sweep done: {time.time() - t0:.1f}s ({count} runs)")
    return pd.DataFrame(all_rows), pd.DataFrame(all_diag)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--hb-method', choices=['auto', 'mcmc', 'gibbs', 'eb'], default='auto')
    parser.add_argument('--mcmc-diag-runs', type=int, default=5)
    args = parser.parse_args()

    n_mc = 5 if args.dry_run else args.n_mc
    regime_names = ['moderate'] if args.dry_run else ['extreme', 'moderate', 'homogeneous']
    n_experts = [100] if args.dry_run else DEFAULT_N_EXPERTS
    k_list = [2] if args.dry_run else DEFAULT_K_LIST

    print(f"exp-001 HB Synthetic Validation | PyMC={HAS_PYMC}")
    print(f"N={N} n_mc={n_mc} hb={args.hb_method}")
    print(f"Regimes={regime_names} n_experts={n_experts} K={k_list}\n")

    # Timing test
    effective_method = args.hb_method
    d = generate_data(REGIMES[2]['moderate'], 100, 2, seed=9999)
    em = d['expert_mask']
    tc = get_counts(d['Z'][em], d['Z_hat'][em], d['S'][em], ['A', 'B'])

    if args.hb_method == 'auto':
        print("Timing Gibbs sampler...")
        t0 = time.time()
        _gc = compute_global_cm(tc)
        _, dg = estimate_cm_hb_gibbs(tc, ['A', 'B'], _gc, seed=9999)
        gt = time.time() - t0
        print(f"  Gibbs: {gt:.2f}s | rhat={dg['rhat_max']:.3f} "
              f"ess={dg['ess_min']:.0f}")
        if gt < 5:
            effective_method = 'gibbs'
            print(f"  Fast enough → using Gibbs for all runs")
        else:
            effective_method = 'eb'
            print(f"  Slow → using EB")
    elif args.hb_method == 'mcmc' and HAS_PYMC:
        print("Timing PyMC MCMC...")
        t0 = time.time()
        try:
            _, dg = estimate_cm_hb_mcmc(tc, ['A', 'B'], seed=9999)
            mt = time.time() - t0
            print(f"  MCMC: {mt:.1f}s | rhat={dg['rhat_max']:.3f} "
                  f"ess={dg['ess_min']:.0f} div={dg['divergent_count']}")
        except Exception as e:
            print(f"  MCMC failed: {e} → falling back to Gibbs")
            effective_method = 'gibbs'

    print(f"\nEffective method: {effective_method}\n")

    df_raw, df_diag = run_sweep(
        regime_names, n_experts, k_list, n_mc,
        hb_method=effective_method, mcmc_diag_runs=args.mcmc_diag_runs)

    df_agg = aggregate_results(df_raw)

    os.makedirs('results', exist_ok=True)
    agg_cols = ['regime', 'n_expert', 'k_subgroups', 'subgroup', 'method',
                'bias_z0', 'rmse_z0', 'coverage_z0', 'bias_z1', 'rmse_z1', 'coverage_z1',
                'bias_z0_se', 'rmse_z0_se', 'bias_z1_se', 'rmse_z1_se']
    df_agg[agg_cols].to_csv('results/exp001_hb_results.csv', index=False)

    diag_cols = ['regime', 'n_expert', 'k_subgroups', 'mc_run',
                 'rhat_max', 'ess_min', 'divergent_count', 'runtime_seconds']
    save_cols = [c for c in diag_cols if c in df_diag.columns]
    if 'hb_method' in df_diag.columns:
        save_cols.insert(4, 'hb_method')
    df_diag[save_cols].to_csv('results/exp001_hb_diagnostics.csv', index=False)

    print_summary(df_agg, df_diag)

    print(f"\nSaved: results/exp001_hb_results.csv ({len(df_agg)} rows)")
    print(f"Saved: results/exp001_hb_diagnostics.csv ({len(df_diag)} rows)")
