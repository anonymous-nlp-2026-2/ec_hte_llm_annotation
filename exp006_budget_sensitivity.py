#!/usr/bin/env python3
"""
exp006_budget_sensitivity.py - Expert Budget Sensitivity Sweep (exp-006)

Systematically quantifies how expert budget (n_expert) affects each correction
method's performance, finding the crossover point where HB partial pooling
starts outperforming Global correction.

Input:  Synthetic DGP with heterogeneous misclassification and subgroup-varying CATE
Output: results/exp006_budget_sensitivity.csv  (aggregated metrics)
        results/exp006_crossover.csv            (crossover analysis)
        results/exp006_run.log                  (run log)
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────

N = 5000
SUBGROUP_LABELS = {2: ['A', 'B'], 4: ['A', 'B', 'C', 'D']}

REGIMES = {
    2: {
        'extreme': {
            'misclass': [0.05, 0.20],
            'tau_z1': [0.5, 1.0],
            'tau_z0': [0.1, 0.2],
        },
        'moderate': {
            'misclass': [0.10, 0.15],
            'tau_z1': [0.5, 1.0],
            'tau_z0': [0.1, 0.2],
        },
    },
    4: {
        'extreme': {
            'misclass': [0.05, 0.15, 0.25, 0.40],
            'tau_z1': [0.3, 0.5, 0.8, 1.2],
            'tau_z0': [0.05, 0.1, 0.15, 0.25],
        },
        'moderate': {
            'misclass': [0.08, 0.12, 0.16, 0.20],
            'tau_z1': [0.3, 0.5, 0.8, 1.2],
            'tau_z0': [0.05, 0.1, 0.15, 0.25],
        },
    },
}

N_EXPERTS = [50, 100, 250, 500, 1000, 2000]
K_LIST = [2, 4]
N_MC = 100

MCMC_N_CHAINS = 4
MCMC_N_ITER = 4000
MCMC_N_WARMUP = 2000
MCMC_HP_A0 = 3.0
MCMC_HP_B0 = 1.0
CHAIN_INITS_POOL = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                    np.array([0.5, 0.5]), np.array([10.0, 10.0])]
COND_THRESHOLD = 100
N_WORKERS = max(1, cpu_count())


# ── DGP ──────────────────────────────────────────────────────────────────────

def generate_data(regime_params, n_expert, k, seed):
    """Generate synthetic data with subgroup-varying CATE and misclassification."""
    rng = np.random.RandomState(seed)
    labels = SUBGROUP_LABELS[k]
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(labels, N, p=[1.0 / k] * k)
    X = rng.randn(N, 5)

    tau = np.zeros(N)
    for i, s in enumerate(labels):
        ms = S == s
        tau[ms] = np.where(Z[ms] == 1,
                           regime_params['tau_z1'][i],
                           regime_params['tau_z0'][i])

    Y = 1.0 + tau * T + 0.5 * X[:, 0] + rng.randn(N)

    mc_rates = np.zeros(N)
    for i, s in enumerate(labels):
        mc_rates[S == s] = regime_params['misclass'][i]
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)

    idx = rng.choice(N, min(n_expert, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True

    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


# ── Helpers (from exp-001) ───────────────────────────────────────────────────

def diff_in_means(Y, T, mask):
    """Difference-in-means estimator with SE."""
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
    """C[z, z_hat] -> M[z_hat, z] via Bayes rule assuming P(Z=0)=P(Z=1)=0.5."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        if cs > 0:
            M[zh, 0] = C[0, zh] / cs
            M[zh, 1] = C[1, zh] / cs
        else:
            M[zh, :] = 0.5
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    """Invert mixing matrix with condition number guard (cond_threshold=100)."""
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


# ── Confusion Matrix Estimators ──────────────────────────────────────────────

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


def _split_rhat(chains):
    """Split R-hat for a scalar parameter. chains: (n_chains, n_draws)."""
    parts = []
    for c in range(chains.shape[0]):
        half = chains.shape[1] // 2
        parts.append(chains[c, :half])
        parts.append(chains[c, half:2 * half])
    parts = np.array(parts)
    n = parts.shape[1]
    chain_means = parts.mean(axis=1)
    W = np.mean(np.var(parts, axis=1, ddof=1))
    B = n * np.var(chain_means, ddof=1)
    if W < 1e-20:
        return 1.0
    var_hat = (1 - 1.0 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def _bulk_ess(chains):
    """Approximate bulk ESS via initial positive sequence."""
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
    """Compute R-hat and ESS over all parameters."""
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


def _log_target_alpha(alpha, theta_z, hp_a0=3.0, hp_b0=1.0):
    """Log posterior for alpha_0[z] in log-alpha space."""
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def _log_target_alpha_vec(alpha, theta_z, hp_a0, hp_b0):
    """Vectorized log posterior across chains. alpha: (nc,), theta_z: (nc, k, 2)."""
    safe = np.maximum(alpha, 1e-10)
    a = safe / 2.0
    k = theta_z.shape[1]
    lp = (hp_a0 - 1.0) * np.log(safe) - hp_b0 * safe
    lp += k * (gammaln(safe) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)), axis=(1, 2))
    return lp


def estimate_cm_hb_gibbs(counts, labels, seed=42):
    """Gibbs sampler for HB Dirichlet-Multinomial, vectorized across chains."""
    k = len(labels)
    nc = MCMC_N_CHAINS
    rng = np.random.RandomState(seed)
    counts_f = counts.astype(float)

    alpha = np.stack(CHAIN_INITS_POOL[:nc]).copy()  # (nc, 2)
    theta = np.ones((nc, k, 2, 2)) * 0.5
    mh_scale = np.full((nc, 2), 0.3)
    accept_ct = np.zeros((nc, 2), dtype=int)
    attempt_ct = np.zeros((nc, 2), dtype=int)

    alpha_samples = np.zeros((nc, MCMC_N_ITER, 2))
    theta_samples = np.zeros((nc, MCMC_N_ITER, k, 2, 2))

    t0 = time.time()

    for it in range(MCMC_N_WARMUP + MCMC_N_ITER):
        for z in [0, 1]:
            post_a = counts_f[None, :, z, :] + alpha[:, z:z+1, None] / 2.0  # (nc, k, 2)
            post_a = np.maximum(post_a, 1e-6)
            g = rng.gamma(post_a)  # (nc, k, 2)
            g_sum = g.sum(axis=2, keepdims=True)
            theta[:, :, z, :] = g / np.maximum(g_sum, 1e-300)

        for z in [0, 1]:
            x_prop = np.log(alpha[:, z]) + rng.normal(0, 1, nc) * mh_scale[:, z]
            a_prop = np.exp(x_prop)
            log_r = (_log_target_alpha_vec(a_prop, theta[:, :, z, :], MCMC_HP_A0, MCMC_HP_B0) -
                     _log_target_alpha_vec(alpha[:, z], theta[:, :, z, :], MCMC_HP_A0, MCMC_HP_B0))
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
            alpha_samples[:, idx] = alpha.copy()
            theta_samples[:, idx] = theta.copy()

    runtime = time.time() - t0

    C = {}
    for i, s in enumerate(labels):
        C[s] = theta_samples[:, :, i, :, :].mean(axis=(0, 1))

    rhat_max, ess_min = _fast_diagnostics(alpha_samples, theta_samples, labels)

    diag = {'rhat_max': rhat_max, 'ess_min': ess_min, 'divergent_count': 0,
            'runtime_seconds': runtime, 'hb_method': 'gibbs'}
    return C, diag


# ── Single MC Run ────────────────────────────────────────────────────────────

def run_one_mc(regime_name, regime_params, n_expert, k, seed):
    """Run one Monte Carlo iteration for all 4 methods."""
    data = generate_data(regime_params, n_expert, k, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = SUBGROUP_LABELS[k]

    counts = get_counts(Z[em], Z_hat[em], S[em], labels)

    C_mle = estimate_cm_mle(counts, labels)
    C_hb, hb_diag = estimate_cm_hb_gibbs(counts, labels, seed=seed)

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
    singularity_guards = 0

    for si, s in enumerate(labels):
        sm = S == s
        tau_true_z0 = regime_params['tau_z0'][si]
        tau_true_z1 = regime_params['tau_z1'][si]

        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        row_base = {**base, 'subgroup': s,
                    'tau_true_z0': tau_true_z0, 'tau_true_z1': tau_true_z1}

        rows.append({**row_base, 'method': 'naive',
                     'tau_hat_z0': th0_n, 'tau_hat_z1': th1_n,
                     'se_z0': se0_n, 'se_z1': se1_n})

        tgc, sgc = invert_mixing_safe(M_gl, tau_obs, se_obs, COND_THRESHOLD)
        rows.append({**row_base, 'method': 'global_corrected',
                     'tau_hat_z0': tgc[0], 'tau_hat_z1': tgc[1],
                     'se_z0': sgc[0], 'se_z1': sgc[1]})

        tml, sml = invert_mixing_safe(M_mle[s], tau_obs, se_obs, COND_THRESHOLD)
        rows.append({**row_base, 'method': 'stratified_mle',
                     'tau_hat_z0': tml[0], 'tau_hat_z1': tml[1],
                     'se_z0': sml[0], 'se_z1': sml[1]})

        if np.linalg.cond(M_hb[s]) > COND_THRESHOLD:
            singularity_guards += 1
        thb, shb = invert_mixing_safe(M_hb[s], tau_obs, se_obs, COND_THRESHOLD)
        rows.append({**row_base, 'method': 'hb_ec_hte',
                     'tau_hat_z0': thb[0], 'tau_hat_z1': thb[1],
                     'se_z0': shb[0], 'se_z1': shb[1]})

    diag = {**base, **hb_diag, 'singularity_guards': singularity_guards}
    return rows, diag


def _run_one_mc_wrapper(args):
    """Wrapper for multiprocessing Pool.map."""
    regime_name, regime_params, n_expert, k, seed = args
    try:
        rows, diag = run_one_mc(regime_name, regime_params, n_expert, k, seed)
        return rows, diag, None
    except Exception as e:
        diag = {'regime': regime_name, 'n_expert': n_expert, 'k_subgroups': k,
                'mc_run': seed, 'hb_method': 'error',
                'rhat_max': np.nan, 'ess_min': np.nan,
                'divergent_count': -1, 'runtime_seconds': 0.0,
                'singularity_guards': -1}
        return [], diag, str(e)


# ── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(logger):
    """Run the full budget sensitivity sweep with multiprocessing."""
    tasks = []
    for k in K_LIST:
        for regime_name in ['extreme', 'moderate']:
            rp = REGIMES[k][regime_name]
            for ne in N_EXPERTS:
                for mc in range(N_MC):
                    tasks.append((regime_name, rp, ne, k, mc))

    total = len(tasks)
    all_rows, all_diag = [], []
    t0 = time.time()
    errors = 0

    print(f"Running {total} MC iterations with {N_WORKERS} workers...", flush=True)
    logger.info(f"Starting sweep: {total} tasks, {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:
        for i, (rows, diag, err) in enumerate(pool.imap_unordered(_run_one_mc_wrapper, tasks, chunksize=4)):
            all_rows.extend(rows)
            all_diag.append(diag)
            if err:
                errors += 1
                logger.warning(f"Task failed: {err}")
            done = i + 1
            if done % max(1, total // 20) == 0:
                el = time.time() - t0
                eta = el / done * (total - done)
                msg = f"[{done}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s"
                logger.info(msg)
                print(msg, flush=True)

    elapsed = time.time() - t0
    msg = f"Sweep done: {elapsed:.1f}s ({total} runs, {errors} errors)"
    logger.info(msg)
    print(msg, flush=True)
    return pd.DataFrame(all_rows), pd.DataFrame(all_diag)


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_results(df):
    """Aggregate MC runs into bias, RMSE, coverage per config x method x subgroup."""
    rows = []
    grp_cols = ['regime', 'n_expert', 'k_subgroups', 'subgroup', 'method']
    for keys, grp in df.groupby(grp_cols):
        regime, n_expert, k, subgroup, method = keys
        row = {'regime': regime, 'n_expert': int(n_expert), 'k_subgroups': int(k),
               'subgroup': subgroup, 'method': method, 'n_mc': len(grp)}
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'].values - grp[f'tau_true_{z}'].values
            bias = err.mean()
            rmse = np.sqrt((err ** 2).mean())
            ci_lo = grp[f'tau_hat_{z}'].values - 1.96 * grp[f'se_{z}'].values
            ci_hi = grp[f'tau_hat_{z}'].values + 1.96 * grp[f'se_{z}'].values
            cov = ((ci_lo <= grp[f'tau_true_{z}'].values) &
                   (grp[f'tau_true_{z}'].values <= ci_hi)).mean()
            row[f'bias_{z}'] = bias
            row[f'rmse_{z}'] = rmse
            row[f'coverage_{z}'] = cov
        rows.append(row)
    return pd.DataFrame(rows)


# ── Crossover Analysis ───────────────────────────────────────────────────────

def crossover_analysis(df_agg):
    """Find n_expert where HB RMSE first beats Global RMSE for each (K, regime)."""
    rows = []
    for k in K_LIST:
        for regime in ['extreme', 'moderate']:
            sub = df_agg[(df_agg['k_subgroups'] == k) & (df_agg['regime'] == regime)]
            if sub.empty:
                continue
            for ne in sorted(N_EXPERTS):
                ne_sub = sub[sub['n_expert'] == ne]

                def avg_rmse(d):
                    if d.empty:
                        return np.nan
                    return np.sqrt((d['rmse_z0'].values ** 2 +
                                    d['rmse_z1'].values ** 2).mean())

                rows.append({
                    'k_subgroups': k, 'regime': regime, 'n_expert': ne,
                    'rmse_naive': avg_rmse(ne_sub[ne_sub['method'] == 'naive']),
                    'rmse_global': avg_rmse(ne_sub[ne_sub['method'] == 'global_corrected']),
                    'rmse_mle': avg_rmse(ne_sub[ne_sub['method'] == 'stratified_mle']),
                    'rmse_hb': avg_rmse(ne_sub[ne_sub['method'] == 'hb_ec_hte']),
                })

    df_cross = pd.DataFrame(rows)
    df_cross['hb_beats_global'] = df_cross['rmse_hb'] < df_cross['rmse_global']
    df_cross['hb_beats_mle'] = df_cross['rmse_hb'] < df_cross['rmse_mle']

    print("\n── Crossover Analysis ──────────────────────────────────────────")
    for k in K_LIST:
        for regime in ['extreme', 'moderate']:
            csub = df_cross[(df_cross['k_subgroups'] == k) &
                            (df_cross['regime'] == regime)].sort_values('n_expert')
            crossover_ne = None
            for _, row in csub.iterrows():
                if row['hb_beats_global']:
                    crossover_ne = int(row['n_expert'])
                    break
            status = f"n_expert={crossover_ne}" if crossover_ne else "not reached"
            print(f"  K={k} {regime:>8s}: HB < Global crossover at {status}")
            print(f"    {'n_expert':>8s}  {'naive':>8s}  {'global':>8s}  "
                  f"{'mle':>8s}  {'hb':>8s}")
            for _, row in csub.iterrows():
                marker = " *" if row['hb_beats_global'] else ""
                print(f"    {int(row['n_expert']):8d}  {row['rmse_naive']:8.4f}  "
                      f"{row['rmse_global']:8.4f}  {row['rmse_mle']:8.4f}  "
                      f"{row['rmse_hb']:8.4f}{marker}")

    return df_cross


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    logging.basicConfig(
        filename='results/exp006_run.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('exp006')

    print(f"exp-006 Budget Sensitivity Sweep")
    print(f"N={N} n_mc={N_MC} n_experts={N_EXPERTS} K={K_LIST}")
    print(f"Regimes: extreme, moderate")
    print(f"MCMC: chains={MCMC_N_CHAINS} iter={MCMC_N_ITER} warmup={MCMC_N_WARMUP}")
    print(f"Workers: {N_WORKERS}")
    total_configs = len(K_LIST) * 2 * len(N_EXPERTS)
    total_runs = total_configs * N_MC
    print(f"Total: {total_configs} configs x {N_MC} MC = {total_runs} runs\n")
    logger.info(f"exp-006 start: N={N} n_mc={N_MC} total_runs={total_runs} workers={N_WORKERS}")

    # Timing test
    print("Timing vectorized Gibbs sampler...")
    d = generate_data(REGIMES[2]['moderate'], 100, 2, seed=9999)
    em = d['expert_mask']
    tc = get_counts(d['Z'][em], d['Z_hat'][em], d['S'][em], ['A', 'B'])
    t0 = time.time()
    _, dg = estimate_cm_hb_gibbs(tc, ['A', 'B'], seed=9999)
    gt = time.time() - t0
    print(f"  Gibbs: {gt:.2f}s | rhat={dg['rhat_max']:.3f} ess={dg['ess_min']:.0f}")
    est_minutes = gt * total_runs / 60 / N_WORKERS
    print(f"  Estimated total (with {N_WORKERS} workers): {est_minutes:.0f} minutes\n")
    logger.info(f"Gibbs timing: {gt:.2f}s, est total: {est_minutes:.0f}min")

    # Run sweep
    df_raw, df_diag = run_sweep(logger)

    # Aggregate
    df_agg = aggregate_results(df_raw)

    # Save main results
    agg_cols = ['regime', 'n_expert', 'k_subgroups', 'subgroup', 'method', 'n_mc',
                'bias_z0', 'rmse_z0', 'coverage_z0', 'bias_z1', 'rmse_z1', 'coverage_z1']
    df_agg[agg_cols].to_csv('results/exp006_budget_sensitivity.csv', index=False)

    # Diagnostics summary
    sg_total = df_diag.get('singularity_guards', pd.Series([0])).clip(lower=0).sum()
    print(f"\nSingularity guards triggered: {int(sg_total)}")
    logger.info(f"Singularity guards total: {int(sg_total)}")
    if sg_total > 0:
        sg_detail = df_diag[df_diag['singularity_guards'] > 0].groupby(
            ['k_subgroups', 'regime', 'n_expert'])['singularity_guards'].sum()
        print(sg_detail.to_string())

    rh = df_diag['rhat_max'].dropna()
    es = df_diag['ess_min'].dropna()
    if not rh.empty:
        print(f"\nMCMC Diagnostics:")
        print(f"  R-hat max: {rh.max():.4f}")
        print(f"  ESS min:   {es.min():.0f}")
        print(f"  Runtime median: {df_diag['runtime_seconds'].median():.2f}s")

    # Crossover analysis
    df_cross = crossover_analysis(df_agg)
    df_cross.to_csv('results/exp006_crossover.csv', index=False)

    # Convergence check: n_expert >= 1000, HB should approximate MLE
    print("\n── Convergence Check (n_expert >= 1000): HB vs MLE ──")
    for k in K_LIST:
        for regime in ['extreme', 'moderate']:
            large_n = df_agg[(df_agg['k_subgroups'] == k) &
                             (df_agg['regime'] == regime) &
                             (df_agg['n_expert'] >= 1000)]
            for ne in sorted(large_n['n_expert'].unique()):
                hb = large_n[(large_n['method'] == 'hb_ec_hte') & (large_n['n_expert'] == ne)]
                mle = large_n[(large_n['method'] == 'stratified_mle') &
                              (large_n['n_expert'] == ne)]
                if hb.empty or mle.empty:
                    continue
                hb_rmse = np.sqrt((hb['rmse_z0'].values ** 2 +
                                   hb['rmse_z1'].values ** 2).mean())
                mle_rmse = np.sqrt((mle['rmse_z0'].values ** 2 +
                                    mle['rmse_z1'].values ** 2).mean())
                diff_pct = abs(hb_rmse - mle_rmse) / mle_rmse * 100
                print(f"  K={k} {regime} ne={ne}: HB={hb_rmse:.4f} MLE={mle_rmse:.4f} "
                      f"diff={diff_pct:.1f}%")

    print(f"\nSaved: results/exp006_budget_sensitivity.csv ({len(df_agg)} rows)")
    print(f"Saved: results/exp006_crossover.csv ({len(df_cross)} rows)")
    print(f"Log:   results/exp006_run.log")
    logger.info("exp-006 complete")
