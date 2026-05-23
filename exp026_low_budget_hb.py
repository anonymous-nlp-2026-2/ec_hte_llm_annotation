#!/usr/bin/env python3
"""
exp026_low_budget_hb.py - Low Budget HB vs Stratified MLE (exp-026)

Validates HB partial pooling advantage over Stratified MLE at very small
expert budgets (n_expert=25-250). When per-subgroup expert counts are tiny
(6-18 labels), MLE CM estimates have extreme variance. HB borrows strength
across subgroups via a shared Dirichlet concentration parameter.

Two configurations:
  K=2: π=[0.05, 0.25], equal weights, Δτ=0.3
  K=4: π=[0.05, 0.10, 0.20, 0.30], equal weights, Δτ=0.3

Output: results/exp_low_budget_hb.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────

N = 5000
SUBGROUP_LABELS = {2: ['A', 'B'], 4: ['A', 'B', 'C', 'D']}

CONFIGS = {
    2: {
        'misclass': [0.05, 0.25],
        'weights': [0.5, 0.5],
        'tau_z0': 0.0,
        'tau_z1': 0.3,
    },
    4: {
        'misclass': [0.05, 0.10, 0.20, 0.30],
        'weights': [0.25, 0.25, 0.25, 0.25],
        'tau_z0': 0.0,
        'tau_z1': 0.3,
    },
}

N_EXPERTS = [25, 50, 75, 100, 150, 250]
N_MC = 100
SEED_BASE = 42
COND_THRESHOLD = 100
N_WORKERS = max(1, cpu_count())

MCMC_N_CHAINS = 2
MCMC_N_ITER = 500
MCMC_N_WARMUP = 200
MCMC_HP_A0 = 3.0
MCMC_HP_B0 = 1.0
CHAIN_INITS_POOL = [np.array([1.0, 1.0]), np.array([5.0, 5.0])]


# ── DGP ──────────────────────────────────────────────────────────────────────

def generate_data(config, n_expert, k, seed):
    rng = np.random.RandomState(seed)
    labels = SUBGROUP_LABELS[k]
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(labels, N, p=config['weights'])
    Y0 = rng.randn(N)
    tau = np.where(Z == 1, config['tau_z1'], config['tau_z0'])
    Y = Y0 + tau * T

    mc_rates = np.zeros(N)
    for i, s in enumerate(labels):
        mc_rates[S == s] = config['misclass'][i]
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)

    idx = rng.choice(N, min(n_expert, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True

    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, Y=Y, tau=tau, expert_mask=expert_mask)


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def invert_mixing_safe(M, tau_obs, se_obs):
    cond = np.linalg.cond(M)
    if cond > COND_THRESHOLD:
        return tau_obs.copy(), se_obs.copy()
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


# ── CM Estimators ────────────────────────────────────────────────────────────

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


def _dirmult_neg_ll(alpha, counts_z):
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


def estimate_cm_hb_eb(counts, labels):
    k = len(labels)
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        cl = [(int(counts[i, z, 0]), int(counts[i, z, 1])) for i in range(k)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a, _cl=cl: _dirmult_neg_ll(a, _cl),
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


def _log_target_alpha_vec(alpha, theta_z, hp_a0, hp_b0):
    safe = np.maximum(alpha, 1e-10)
    a = safe / 2.0
    k = theta_z.shape[1]
    lp = (hp_a0 - 1.0) * np.log(safe) - hp_b0 * safe
    lp += k * (gammaln(safe) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)), axis=(1, 2))
    return lp


def estimate_cm_hb_gibbs(counts, labels, seed=42):
    k = len(labels)
    nc = MCMC_N_CHAINS
    rng = np.random.RandomState(seed)
    counts_f = counts.astype(float)

    alpha = np.stack(CHAIN_INITS_POOL[:nc]).copy()
    theta = np.ones((nc, k, 2, 2)) * 0.5
    mh_scale = np.full((nc, 2), 0.3)
    accept_ct = np.zeros((nc, 2), dtype=int)
    attempt_ct = np.zeros((nc, 2), dtype=int)

    theta_samples = np.zeros((nc, MCMC_N_ITER, k, 2, 2))

    for it in range(MCMC_N_WARMUP + MCMC_N_ITER):
        for z in [0, 1]:
            post_a = counts_f[None, :, z, :] + alpha[:, z:z + 1, None] / 2.0
            post_a = np.maximum(post_a, 1e-6)
            g = rng.gamma(post_a)
            g_sum = g.sum(axis=2, keepdims=True)
            theta[:, :, z, :] = g / np.maximum(g_sum, 1e-300)

        for z in [0, 1]:
            x_prop = np.log(alpha[:, z]) + rng.normal(0, 1, nc) * mh_scale[:, z]
            a_prop = np.exp(x_prop)
            log_r = (_log_target_alpha_vec(
                         a_prop, theta[:, :, z, :], MCMC_HP_A0, MCMC_HP_B0) -
                     _log_target_alpha_vec(
                         alpha[:, z], theta[:, :, z, :], MCMC_HP_A0, MCMC_HP_B0))
            u = np.log(rng.rand(nc))
            accepted = u < log_r
            alpha[accepted, z] = a_prop[accepted]
            if it < MCMC_N_WARMUP:
                accept_ct[:, z] += accepted.astype(int)
                attempt_ct[:, z] += 1

        if it < MCMC_N_WARMUP and it > 0 and it % 100 == 0:
            for c in range(nc):
                for z in [0, 1]:
                    rate = accept_ct[c, z] / max(attempt_ct[c, z], 1)
                    if rate < 0.2:
                        mh_scale[c, z] *= 0.7
                    elif rate > 0.5:
                        mh_scale[c, z] *= 1.3
                    accept_ct[c, z] = 0
                    attempt_ct[c, z] = 0

        if it >= MCMC_N_WARMUP:
            idx = it - MCMC_N_WARMUP
            theta_samples[:, idx] = theta.copy()

    C = {}
    for i, s in enumerate(labels):
        C[s] = theta_samples[:, :, i, :, :].mean(axis=(0, 1))
    return C


# ── Single MC Run ────────────────────────────────────────────────────────────

def run_one_mc(config, n_expert, k, seed, hb_method):
    data = generate_data(config, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]
    tau_z0, tau_z1 = config['tau_z0'], config['tau_z1']

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)

    C_mle = estimate_cm_mle(counts, labels)
    if hb_method == 'gibbs':
        C_hb = estimate_cm_hb_gibbs(counts, labels, seed=seed)
    else:
        C_hb = estimate_cm_hb_eb(counts, labels)

    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5

    M_gl = build_mixing_matrix(C_gl)
    M_mle = {s: build_mixing_matrix(C_mle[s]) for s in labels}
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in labels}

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

        tgc, sgc = invert_mixing_safe(M_gl, tau_obs, se_obs)
        tml, sml = invert_mixing_safe(M_mle[s], tau_obs, se_obs)
        thb, shb = invert_mixing_safe(M_hb[s], tau_obs, se_obs)

        estimates = {
            'oracle':           (np.array([th0_or, th1_or]), np.array([se0_or, se1_or])),
            'naive':            (tau_obs, se_obs),
            'global_corrected': (tgc, sgc),
            'stratified_mle':   (tml, sml),
            'hb_ec_hte':        (thb, shb),
        }

        base = {'K': k, 'regime': 'extreme', 'n_expert': n_expert,
                'subgroup': s, 'mc_run': seed}

        for method, (tau_hat, se_hat) in estimates.items():
            for zi in [0, 1]:
                tau_true = tau_z0 if zi == 0 else tau_z1
                if np.isnan(tau_hat[zi]):
                    continue
                rows.append({**base, 'method': method, 'z': zi,
                             'tau_hat': tau_hat[zi], 'se': se_hat[zi],
                             'tau_true': tau_true})

    return rows


def _run_wrapper(args):
    config, n_expert, k, seed, hb_method = args
    try:
        return run_one_mc(config, n_expert, k, seed, hb_method), None
    except Exception as e:
        return [], str(e)


# ── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(hb_method):
    tasks = []
    for k in [2, 4]:
        config = CONFIGS[k]
        for ne in N_EXPERTS:
            for mc in range(N_MC):
                tasks.append((config, ne, k, SEED_BASE + mc, hb_method))

    total = len(tasks)
    all_rows = []
    t0 = time.time()
    errors = 0

    print(f"Running {total} MC iterations with {N_WORKERS} workers...", flush=True)

    with Pool(N_WORKERS) as pool:
        for i, (rows, err) in enumerate(
                pool.imap_unordered(_run_wrapper, tasks, chunksize=4)):
            all_rows.extend(rows)
            if err:
                errors += 1
            done = i + 1
            if done % max(1, total // 20) == 0:
                el = time.time() - t0
                eta = el / done * (total - done)
                print(f"  [{done}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s",
                      flush=True)

    elapsed = time.time() - t0
    print(f"Sweep done: {elapsed:.1f}s ({total} runs, {errors} errors)",
          flush=True)
    return pd.DataFrame(all_rows)


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_results(df_raw):
    rows = []
    grp_cols = ['K', 'regime', 'n_expert', 'method', 'subgroup', 'z']
    for keys, grp in df_raw.groupby(grp_cols):
        K, regime, n_expert, method, subgroup, z = keys
        err = grp['tau_hat'].values - grp['tau_true'].values
        n_mc = len(grp)
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        ci_lo = grp['tau_hat'].values - 2.0 * grp['se'].values
        ci_hi = grp['tau_hat'].values + 2.0 * grp['se'].values
        cov = ((ci_lo <= grp['tau_true'].values) &
               (grp['tau_true'].values <= ci_hi)).mean()
        se_bias = err.std() / np.sqrt(n_mc)
        mse_se = (err ** 2).std() / np.sqrt(n_mc)
        se_rmse = mse_se / (2 * rmse) if rmse > 1e-10 else np.nan
        rows.append({
            'K': int(K), 'regime': regime, 'n_expert': int(n_expert),
            'method': method, 'subgroup': subgroup, 'z': int(z),
            'bias': bias, 'rmse': rmse, 'coverage': cov,
            'se_bias': se_bias, 'se_rmse': se_rmse,
        })
    return pd.DataFrame(rows)


# ── Output ───────────────────────────────────────────────────────────────────

def print_comparison(df_agg):
    for k in [2, 4]:
        print(f"\n=== K={k} extreme ===")
        print(f"{'n_expert':>8s} | {'MLE RMSE':>8s} | {'HB RMSE':>8s} | "
              f"{'Δ%':>6s} | {'MLE Cov':>7s} | {'HB Cov':>7s} | {'Δpp':>6s}")
        print("---------+----------+----------+-------+"
              "---------+---------+-------")

        sub = df_agg[df_agg['K'] == k]
        for ne in N_EXPERTS:
            ne_sub = sub[sub['n_expert'] == ne]
            mle_rows = ne_sub[ne_sub['method'] == 'stratified_mle']
            hb_rows = ne_sub[ne_sub['method'] == 'hb_ec_hte']

            mle_rmse = mle_rows['rmse'].mean() if not mle_rows.empty else np.nan
            hb_rmse = hb_rows['rmse'].mean() if not hb_rows.empty else np.nan
            mle_cov = mle_rows['coverage'].mean() if not mle_rows.empty else np.nan
            hb_cov = hb_rows['coverage'].mean() if not hb_rows.empty else np.nan

            dpct = ((mle_rmse - hb_rmse) / mle_rmse * 100
                    if mle_rmse > 1e-10 else 0)
            dpp = (hb_cov - mle_cov) * 100

            blowup = " *** BLOWUP" if mle_rmse > 10 else ""
            print(f"{ne:8d} | {mle_rmse:8.4f} | {hb_rmse:8.4f} | "
                  f"{dpct:+5.1f}% | {mle_cov:7.3f} | {hb_cov:7.3f} | "
                  f"{dpp:+5.1f}{blowup}")

    print("\n── All Methods Summary ──")
    for k in [2, 4]:
        print(f"\n  K={k}")
        sub = df_agg[df_agg['K'] == k]
        print(f"  {'n_expert':>8s}  {'oracle':>8s}  {'naive':>8s}  "
              f"{'global':>8s}  {'mle':>8s}  {'hb':>8s}")
        for ne in N_EXPERTS:
            ne_sub = sub[sub['n_expert'] == ne]
            vals = {}
            for m in ['oracle', 'naive', 'global_corrected',
                      'stratified_mle', 'hb_ec_hte']:
                ms = ne_sub[ne_sub['method'] == m]
                vals[m] = ms['rmse'].mean() if not ms.empty else np.nan
            print(f"  {ne:8d}  {vals['oracle']:8.4f}  {vals['naive']:8.4f}  "
                  f"{vals['global_corrected']:8.4f}  "
                  f"{vals['stratified_mle']:8.4f}  {vals['hb_ec_hte']:8.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("exp-026 Low Budget HB vs Stratified MLE")
    print(f"N={N} N_MC={N_MC} n_experts={N_EXPERTS}")
    print(f"K=2: π={CONFIGS[2]['misclass']}  K=4: π={CONFIGS[4]['misclass']}")
    print(f"τ(0)={CONFIGS[2]['tau_z0']}, τ(1)={CONFIGS[2]['tau_z1']}, "
          f"Δτ={CONFIGS[2]['tau_z1'] - CONFIGS[2]['tau_z0']}")
    total_runs = 2 * len(N_EXPERTS) * N_MC
    print(f"Total: 2 configs × {len(N_EXPERTS)} budgets × {N_MC} MC "
          f"= {total_runs} runs")
    print(f"Workers: {N_WORKERS}")

    print("\nTiming Gibbs sampler...")
    d = generate_data(CONFIGS[2], 50, 2, seed=9999)
    em = d['expert_mask']
    tc = get_counts(d['Z'][em], d['Z_hat'][em], d['S'][em], ['A', 'B'])
    t0 = time.time()
    _ = estimate_cm_hb_gibbs(tc, ['A', 'B'], seed=9999)
    gt = time.time() - t0
    print(f"  Gibbs: {gt:.2f}s per call")
    est_minutes = gt * total_runs / 60 / N_WORKERS
    print(f"  Estimated total ({N_WORKERS} workers): {est_minutes:.0f} minutes")

    hb_method = 'gibbs'
    print("  → Using Gibbs sampler (no EB fallback)\n")

    df_raw = run_sweep(hb_method)

    df_agg = aggregate_results(df_raw)

    out_cols = ['K', 'regime', 'n_expert', 'method', 'subgroup', 'z',
                'bias', 'rmse', 'coverage', 'se_bias', 'se_rmse']
    df_agg[out_cols].to_csv('results/exp_low_budget_hb.csv', index=False)

    print_comparison(df_agg)

    print(f"\nSaved: results/exp_low_budget_hb.csv ({len(df_agg)} rows)")
