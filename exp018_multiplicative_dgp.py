#!/usr/bin/env python3
"""
exp018_multiplicative_dgp.py — Multiplicative CATE DGP validation for EC-HTE

Validates EC-HTE bias correction under a multiplicative treatment effect model:
  Y(1) = Y(0) * exp(tau(Z))

Methods: oracle, naive, global_corrected, ec_hte (HB Gibbs)
Config: K=2, extreme misclass (A=0.05, B=0.25), n_expert=500, 50 MC seeds
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

N = 5000
N_EXPERT = 500
N_MC = 50
LABELS = ['A', 'B']
MISCLASS = [0.05, 0.25]
TAU_Z1 = 0.3


def compute_true_cate():
    rng = np.random.RandomState(99999)
    n = 1_000_000
    X = rng.randn(n, 5)
    S = np.where(X[:, 0] > 0, 'B', 'A')
    Y0 = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) + 1
    true_cate = {}
    for s in LABELS:
        mask = S == s
        ey0 = Y0[mask].mean()
        true_cate[(s, 0)] = 0.0
        true_cate[(s, 1)] = ey0 * (np.exp(TAU_Z1) - 1)
    return true_cate


def generate_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 5)
    Z = rng.binomial(1, 0.5, N)
    S = np.where(X[:, 0] > 0, 'B', 'A')

    Z_hat = Z.copy()
    for i, s in enumerate(LABELS):
        mask = S == s
        flip = rng.binomial(1, MISCLASS[i], mask.sum())
        Z_hat[mask] = np.where(flip, 1 - Z[mask], Z[mask])

    T = rng.binomial(1, 0.5, N)
    Y0 = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) + 1
    tau_z = TAU_Z1 * Z
    Y1 = Y0 * np.exp(tau_z)
    Y = T * Y1 + (1 - T) * Y0 + rng.randn(N) * 0.5

    idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True

    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, Y=Y, expert_mask=expert_mask)


def diff_in_means(Y, T, mask):
    t1 = mask & (T == 1)
    t0 = mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


def get_counts(z_true, z_hat, subgroups):
    counts = np.zeros((2, 2, 2), dtype=int)
    for i, s in enumerate(LABELS):
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


def invert_mixing(M, tau_obs, se_obs, cond_threshold=100):
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


# ── HB Gibbs Sampler ────────────────────────────────────────────────────────

def _log_target_alpha(alpha, theta_z, hp_a0=3.0, hp_b0=1.0):
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, seed=42, n_iter=4000, n_warmup=2000, n_chains=4):
    k = len(LABELS)
    chain_inits = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                   np.array([0.5, 0.5]), np.array([10.0, 10.0])]
    all_theta = []

    for chain in range(n_chains):
        rng = np.random.RandomState(seed * 31 + chain * 7919 + 1)
        alpha = chain_inits[chain].copy()
        theta = np.ones((k, 2, 2)) * 0.5
        mh_scale = np.array([0.3, 0.3])
        accept_ct = np.array([0, 0])
        attempt_ct = np.array([0, 0])
        theta_samples = np.zeros((n_iter, k, 2, 2))

        for it in range(n_warmup + n_iter):
            for z in [0, 1]:
                post_a = counts[:, z, :].astype(float) + alpha[z] / 2.0
                post_a = np.maximum(post_a, 1e-6)
                g = rng.gamma(post_a)
                g_sum = g.sum(axis=1, keepdims=True)
                theta[:, z, :] = g / np.maximum(g_sum, 1e-300)

            for z in [0, 1]:
                x_prop = np.log(alpha[z]) + rng.normal(0, mh_scale[z])
                a_prop = np.exp(x_prop)
                log_r = (_log_target_alpha(a_prop, theta[:, z, :]) -
                         _log_target_alpha(alpha[z], theta[:, z, :]))
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
                theta_samples[it - n_warmup] = theta.copy()

        all_theta.append(theta_samples)

    theta_arr = np.stack(all_theta)
    C = {}
    for i, s in enumerate(LABELS):
        C[s] = theta_arr[:, :, i, :, :].mean(axis=(0, 1))
    return C


# ── Single MC Run ───────────────────────────────────────────────────────────

def run_one_mc(seed, true_cate):
    data = generate_data(seed)
    Y, T, Z, Z_hat, S, em = (data['Y'], data['T'], data['Z'],
                               data['Z_hat'], data['S'], data['expert_mask'])

    counts = get_counts(Z[em], Z_hat[em], S[em])

    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    M_gl = build_mixing_matrix(C_gl)

    C_hb = estimate_cm_hb_gibbs(counts, seed=seed)
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in LABELS}

    rows = []
    for s in LABELS:
        sm = S == s
        th_or, se_or = np.zeros(2), np.zeros(2)
        th_na, se_na = np.zeros(2), np.zeros(2)
        for z in [0, 1]:
            th_or[z], se_or[z] = diff_in_means(Y, T, sm & (Z == z))
            th_na[z], se_na[z] = diff_in_means(Y, T, sm & (Z_hat == z))

        if np.any(np.isnan(th_na)):
            continue

        th_gc, se_gc = invert_mixing(M_gl, th_na, se_na)
        th_hb, se_hb = invert_mixing(M_hb[s], th_na, se_na)

        methods = {
            'oracle': (th_or, se_or),
            'naive': (th_na, se_na),
            'global_corrected': (th_gc, se_gc),
            'ec_hte': (th_hb, se_hb),
        }

        for z in [0, 1]:
            ct = true_cate[(s, z)]
            for method, (th, se) in methods.items():
                est = th[z]
                bias = est - ct
                ci_lo = est - 1.96 * se[z]
                ci_hi = est + 1.96 * se[z]
                cov = int(ci_lo <= ct <= ci_hi)
                rows.append({
                    'seed': seed,
                    'subgroup': s,
                    'z': z,
                    'method': method,
                    'cate_est': est,
                    'cate_true': ct,
                    'bias': bias,
                    'coverage': cov,
                })
    return rows


# ── Summary ─────────────────────────────────────────────────────────────────

def generate_summary(df):
    lines = ['# exp-018: Multiplicative DGP — Analysis', '']
    lines.append('## Configuration')
    lines.append(f'- N={N}, K=2, n_expert={N_EXPERT}, MC seeds={df["seed"].nunique()}')
    lines.append(f'- Misclass: A(S0)={MISCLASS[0]}, B(S1)={MISCLASS[1]} (extreme)')
    lines.append(f'- DGP: Y(1) = Y(0) * exp(tau(Z)), tau(Z=1)={TAU_Z1}, tau(Z=0)=0')
    lines.append('')

    lines.append('## Per-method per-(subgroup, z) results')
    lines.append('')
    lines.append('| Subgroup | Z | Method | Avg |Bias| | RMSE | Coverage |')
    lines.append('|----------|---|--------|---------|------|----------|')

    summary_rows = []
    for (s, z, m), grp in df.groupby(['subgroup', 'z', 'method']):
        bias_vals = grp['bias'].values
        avg_abs_bias = np.abs(bias_vals).mean()
        rmse = np.sqrt((bias_vals ** 2).mean())
        coverage = grp['coverage'].mean()
        lines.append(f'| {s} | {z} | {m} | {avg_abs_bias:.4f} | {rmse:.4f} | {coverage:.3f} |')
        summary_rows.append({
            'subgroup': s, 'z': int(z), 'method': m,
            'avg_abs_bias': avg_abs_bias, 'rmse': rmse, 'coverage': coverage
        })

    lines.append('')
    lines.append('## Method-level summary (averaged across subgroups and z)')
    lines.append('')
    lines.append('| Method | Avg |Bias| | Avg RMSE | Avg Coverage |')
    lines.append('|--------|---------|----------|--------------|')

    sr = pd.DataFrame(summary_rows)
    method_summary = {}
    for m, grp in sr.groupby('method'):
        ab = grp['avg_abs_bias'].mean()
        rm = grp['rmse'].mean()
        cv = grp['coverage'].mean()
        lines.append(f'| {m} | {ab:.4f} | {rm:.4f} | {cv:.3f} |')
        method_summary[m] = {'avg_abs_bias': ab, 'rmse': rm, 'coverage': cv}

    # Compare with exp-001 additive DGP
    lines.append('')
    lines.append('## Comparison with exp-001 (additive DGP, K=2/extreme/n_expert=500)')
    lines.append('')

    exp001_path = 'results/exp001_hb_results.csv'
    if os.path.exists(exp001_path):
        df1 = pd.read_csv(exp001_path)
        ref = df1[(df1['regime'] == 'extreme') & (df1['n_expert'] == 500) &
                  (df1['k_subgroups'] == 2)]

        lines.append('| Method (exp-001 name) | exp-001 Avg RMSE | exp-018 Avg RMSE | exp-001 Avg Cov | exp-018 Avg Cov |')
        lines.append('|-----------------------|------------------|------------------|-----------------|-----------------|')

        method_map = {
            'oracle': 'oracle',
            'naive': 'naive',
            'global_corrected': 'global_corrected',
            'hb_dirichlet': 'ec_hte',
        }
        for m1, m18 in method_map.items():
            r1 = ref[ref['method'] == m1]
            if r1.empty or m18 not in method_summary:
                continue
            rmse_001 = np.sqrt((r1['rmse_z0'].values ** 2 + r1['rmse_z1'].values ** 2).mean() / 2)
            cov_001 = ((r1['coverage_z0'].values + r1['coverage_z1'].values) / 2).mean()
            rmse_018 = method_summary[m18]['rmse']
            cov_018 = method_summary[m18]['coverage']
            lines.append(f'| {m1} → {m18} | {rmse_001:.4f} | {rmse_018:.4f} | {cov_001:.3f} | {cov_018:.3f} |')
    else:
        lines.append('exp-001 results not found for comparison.')

    lines.append('')
    lines.append('## Conclusion')
    lines.append('')

    ec = method_summary.get('ec_hte', {})
    naive = method_summary.get('naive', {})
    oracle = method_summary.get('oracle', {})

    ec_rmse = ec.get('rmse', np.nan)
    naive_rmse = naive.get('rmse', np.nan)
    ec_cov = ec.get('coverage', np.nan)
    ec_bias = ec.get('avg_abs_bias', np.nan)
    naive_bias = naive.get('avg_abs_bias', np.nan)

    rmse_improv = (naive_rmse - ec_rmse) / naive_rmse * 100 if naive_rmse > 0 else 0
    bias_improv = (naive_bias - ec_bias) / naive_bias * 100 if naive_bias > 0 else 0

    lines.append(f'- EC-HTE vs naive: RMSE improvement = {rmse_improv:+.1f}%, |bias| improvement = {bias_improv:+.1f}%')
    lines.append(f'- EC-HTE avg coverage: {ec_cov:.3f} (target: 0.95)')
    lines.append(f'- Oracle avg coverage: {oracle.get("coverage", np.nan):.3f}')
    lines.append('')

    if rmse_improv > 5 and ec_cov > 0.85:
        lines.append('**EC-HTE is effective under the multiplicative DGP.** '
                     'Bias correction successfully reduces RMSE and maintains reasonable coverage '
                     'even when the treatment effect is multiplicative rather than additive. '
                     'This demonstrates that the linear mixing-matrix correction framework is robust '
                     'to the functional form of the treatment effect, since misclassification bias '
                     'in diff-in-means operates at the level of conditional means regardless of the '
                     'underlying DGP.')
    else:
        lines.append('**EC-HTE shows limited effectiveness under the multiplicative DGP.** '
                     'The bias correction does not substantially improve over naive estimation.')

    summary_text = '\n'.join(lines)
    with open('results/exp018_multiplicative_analysis.md', 'w') as f:
        f.write(summary_text)
    print(f"\nSaved results/exp018_multiplicative_analysis.md")
    print('\n' + summary_text)

    return method_summary


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=N_MC)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else args.n_mc

    print(f"exp-018: Multiplicative DGP | N={N} K=2 n_expert={N_EXPERT} n_mc={n_mc}")
    print(f"Misclass: A={MISCLASS[0]}, B={MISCLASS[1]} (extreme)")

    print("Computing true CATE (large-sample MC)...")
    true_cate = compute_true_cate()
    for s in LABELS:
        for z in [0, 1]:
            print(f"  CATE(Z={z}, S={s}) = {true_cate[(s, z)]:.4f}")

    t0 = time.time()
    all_rows = []
    for mc in range(n_mc):
        rows = run_one_mc(mc, true_cate)
        all_rows.extend(rows)
        if (mc + 1) % max(1, n_mc // 10) == 0:
            el = time.time() - t0
            eta = el / (mc + 1) * (n_mc - mc - 1)
            print(f"  [{mc+1}/{n_mc}] {el:.1f}s elapsed, ETA {eta:.0f}s")

    df = pd.DataFrame(all_rows)
    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s, {len(df)} rows")

    os.makedirs('results', exist_ok=True)
    df.to_csv('results/exp018_multiplicative_results.csv', index=False)
    print(f"Saved results/exp018_multiplicative_results.csv")

    method_summary = generate_summary(df)
    return method_summary


if __name__ == '__main__':
    main()
