#!/usr/bin/env python3
"""
exp019_pz1_sweep.py — P(Z=1) Sweep Experiment (exp-019)

Validates Theorem 1 bias decomposition and EC-HTE correction across
varying P(Z=1) ∈ {0.1, 0.2, 0.3, 0.4, 0.5}. Addresses reviewer comment M2
(A2+A3 assumptions too strong when class prevalence is imbalanced).

Methods: oracle, naive, global_corrected, hb_ec_hte (EB)
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

# ── Config ───────────────────────────────────────────────────────────────────

N = 5000
P_Z1_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
K_VALUES = [2, 4]
N_EXPERT = 250
N_MC = 100

SUBGROUP_LABELS = {2: ['A', 'B'], 4: ['A', 'B', 'C', 'D']}

MISCLASS_RATES = {
    2: [0.05, 0.25],
    4: [0.05, 0.10, 0.20, 0.30],
}

TAU_Z0 = 0.5
TAU_Z1 = 1.0


# ── DGP ──────────────────────────────────────────────────────────────────────

def generate_data(p_z1, k, seed):
    rng = np.random.RandomState(seed)
    labels = SUBGROUP_LABELS[k]
    misclass = MISCLASS_RATES[k]

    X = rng.randn(N, 5)

    # Subgroups based on X[:,0] quantiles
    quantiles = np.linspace(0, 1, k + 1)[1:-1]
    thresholds = np.quantile(X[:, 0], quantiles)
    S_idx = np.digitize(X[:, 0], thresholds)
    S = np.array([labels[i] for i in S_idx])

    # Z ~ Bernoulli(p_z1)
    Z_true = rng.binomial(1, p_z1, N)

    # Symmetric misclassification per subgroup
    Z_hat = Z_true.copy()
    for i, s in enumerate(labels):
        mask = S == s
        pi_s = misclass[i]
        flip = rng.binomial(1, pi_s, mask.sum()).astype(bool)
        Z_hat[mask] = np.where(flip, 1 - Z_true[mask], Z_true[mask])

    # Treatment (RCT)
    T = rng.binomial(1, 0.5, N)

    # Outcome: additive DGP
    tau = np.where(Z_true == 1, TAU_Z1, TAU_Z0)
    Y = 1.0 + 0.5 * X[:, 0] + tau * T + rng.randn(N)

    # Expert subsample
    idx = rng.choice(N, min(N_EXPERT, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True

    return dict(T=T, Z_true=Z_true, Z_hat=Z_hat, S=S, X=X, Y=Y,
                tau=tau, expert_mask=expert_mask, labels=labels)


# ── Helpers ──────────────────────────────────────────────────────────────────

def diff_in_means(Y, T, mask):
    t1 = mask & (T == 1)
    t0 = mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
    return est, se


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


def build_mixing_matrix(C, p_z, p_z_hat):
    """M[z_hat, z] = P(Z=z | Z_hat=z_hat) via Bayes' rule with general P(Z)."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        denom = p_z_hat[zh]
        if denom < 1e-12:
            M[zh, :] = 0.5
        else:
            for z in [0, 1]:
                M[zh, z] = C[z, zh] * p_z[z] / denom
    return M


def compute_p_z_hat(C, p_z):
    """P(Z_hat=zh) = sum_z P(Z_hat=zh|Z=z) P(Z=z)."""
    p_zh = np.zeros(2)
    for zh in [0, 1]:
        for z in [0, 1]:
            p_zh[zh] += C[z, zh] * p_z[z]
    return p_zh


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


# ── CM Estimators ────────────────────────────────────────────────────────────

def estimate_cm_global(counts):
    """Pool all subgroups, Laplace smoothing."""
    cg = counts.sum(axis=0)
    C = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
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
    """Empirical Bayes HB: optimize shared Dirichlet concentration, posterior mean."""
    k = len(labels)
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        cl = [(int(counts[i, z, 0]), int(counts[i, z, 1])) for i in range(k)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a: _dirmult_neg_ll(a, cl), bounds=(0.01, 500), method='bounded')
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


# ── Single MC Run ────────────────────────────────────────────────────────────

def run_one_mc(p_z1, k, seed):
    data = generate_data(p_z1, k, seed)
    Y, T = data['Y'], data['T']
    Z_true, Z_hat, S = data['Z_true'], data['Z_hat'], data['S']
    em = data['expert_mask']
    labels = data['labels']

    p_z = np.array([1.0 - p_z1, p_z1])

    counts = get_counts(Z_true[em], Z_hat[em], S[em], labels)

    # Oracle CM (true rates from full population)
    counts_oracle = get_counts(Z_true, Z_hat, S, labels)

    # Global CM (pooled expert labels)
    C_global = estimate_cm_global(counts)

    # HB EB per-subgroup CM
    C_hb = estimate_cm_hb_eb(counts, labels)

    rows = []
    for si, s in enumerate(labels):
        sm = S == s

        # Oracle observed CATE (stratified by true Z)
        th0_or, se0_or = diff_in_means(Y, T, sm & (Z_true == 0))
        th1_or, se1_or = diff_in_means(Y, T, sm & (Z_true == 1))

        # Naive observed CATE (stratified by Z_hat)
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))

        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)):
            continue

        # Global corrected
        p_zh_global = compute_p_z_hat(C_global, p_z)
        M_global = build_mixing_matrix(C_global, p_z, p_zh_global)
        tgc, sgc = invert_mixing_safe(M_global, tau_obs, se_obs)

        # HB EC-HTE corrected
        C_hb_s = C_hb[s]
        p_zh_hb = compute_p_z_hat(C_hb_s, p_z)
        M_hb = build_mixing_matrix(C_hb_s, p_z, p_zh_hb)
        thb, shb = invert_mixing_safe(M_hb, tau_obs, se_obs)

        # Oracle corrected (using true full-pop CM)
        C_or_s = np.zeros((2, 2))
        for z in [0, 1]:
            n = counts_oracle[si, z, :].sum()
            for zh in [0, 1]:
                C_or_s[z, zh] = counts_oracle[si, z, zh] / n if n > 0 else 0.5
        p_zh_or = compute_p_z_hat(C_or_s, p_z)
        M_or = build_mixing_matrix(C_or_s, p_z, p_zh_or)
        tor, sor = invert_mixing_safe(M_or, tau_obs, se_obs)

        base = dict(p_z1=p_z1, K=k, seed=seed, subgroup=s)

        for z_val, z_label in [(0, 0), (1, 1)]:
            true_cate = TAU_Z0 if z_val == 0 else TAU_Z1

            def make_row(method, est, se):
                bias = est - true_cate
                covered = int(abs(bias) <= 1.96 * se) if se > 0 and not np.isnan(se) else np.nan
                return {**base, 'z': z_val, 'method': method,
                        'cate_est': est, 'cate_true': true_cate,
                        'bias': bias, 'se': se, 'coverage': covered}

            rows.append(make_row('oracle', [th0_or, th1_or][z_val], [se0_or, se1_or][z_val]))
            rows.append(make_row('naive', tau_obs[z_val], se_obs[z_val]))
            rows.append(make_row('global_corrected', tgc[z_val], sgc[z_val]))
            rows.append(make_row('hb_ec_hte', thb[z_val], shb[z_val]))

            # Also store oracle-corrected for reference
            rows.append(make_row('oracle_corrected', tor[z_val], sor[z_val]))

    return rows


# ── Theorem 1 Analysis ───────────────────────────────────────────────────────

def compute_theorem1_r2(df_raw, p_z1_val, k_val):
    """
    Theorem 1: global correction residual bias for subgroup s is
      bias(s) = (M_global^{-1} M_s - I) τ_true
    where M is the mixing matrix built from the true CM and P(Z).

    Compute R² of analytical predicted bias vs mean empirical bias,
    across (subgroup, z) cells averaged over MC seeds.
    """
    sub = df_raw[(df_raw['p_z1'] == p_z1_val) & (df_raw['K'] == k_val)].copy()
    if sub.empty:
        return np.nan

    gc = sub[sub['method'] == 'global_corrected'].copy()
    if gc.empty:
        return np.nan

    misclass = MISCLASS_RATES[k_val]
    labels = SUBGROUP_LABELS[k_val]
    p_z = np.array([1.0 - p_z1_val, p_z1_val])
    tau_true = np.array([TAU_Z0, TAU_Z1])

    # Build true global CM (equal subgroup sizes → simple average of rates)
    pi_global = np.mean(misclass)
    C_global = np.array([[1 - pi_global, pi_global], [pi_global, 1 - pi_global]])
    p_zh_global = compute_p_z_hat(C_global, p_z)
    M_global = build_mixing_matrix(C_global, p_z, p_zh_global)
    det_g = np.linalg.det(M_global)
    if abs(det_g) < 1e-10:
        return np.nan
    M_global_inv = np.linalg.inv(M_global)

    predicted = []
    empirical = []

    for i, s in enumerate(labels):
        pi_s = misclass[i]
        C_s = np.array([[1 - pi_s, pi_s], [pi_s, 1 - pi_s]])
        p_zh_s = compute_p_z_hat(C_s, p_z)
        M_s = build_mixing_matrix(C_s, p_z, p_zh_s)

        bias_pred = (M_global_inv @ M_s - np.eye(2)) @ tau_true

        for z in [0, 1]:
            mean_bias = gc[(gc['subgroup'] == s) & (gc['z'] == z)]['bias'].mean()
            predicted.append(bias_pred[z])
            empirical.append(mean_bias)

    predicted = np.array(predicted)
    empirical = np.array(empirical)

    if len(predicted) < 3 or np.std(predicted) < 1e-12:
        return np.nan

    ss_res = np.sum((empirical - predicted) ** 2)
    ss_tot = np.sum((empirical - empirical.mean()) ** 2)
    if ss_tot < 1e-20:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ── Main Sweep ───────────────────────────────────────────────────────────────

def run_sweep(n_mc=N_MC, verbose=True):
    all_rows = []
    configs = [(p, k) for p in P_Z1_VALUES for k in K_VALUES]
    total = len(configs) * n_mc
    count = 0
    t0 = time.time()

    for p_z1, k in configs:
        for mc in range(n_mc):
            try:
                rows = run_one_mc(p_z1, k, seed=mc)
                all_rows.extend(rows)
            except Exception as e:
                if verbose:
                    print(f"  WARN: p_z1={p_z1} k={k} mc={mc}: {e}")
            count += 1
            if verbose and count % max(1, total // 20) == 0:
                el = time.time() - t0
                eta = el / count * (total - count)
                print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s | "
                      f"p_z1={p_z1} k={k} mc={mc}")

    if verbose:
        print(f"  Sweep done: {time.time() - t0:.1f}s ({count} runs)")

    return pd.DataFrame(all_rows)


# ── Analysis & Output ────────────────────────────────────────────────────────

def generate_analysis(df):
    os.makedirs('results', exist_ok=True)
    os.makedirs('artifacts/pz1_sweep', exist_ok=True)

    # Save raw CSV
    df.to_csv('results/exp019_pz1_sweep_results.csv', index=False)
    print(f"Saved results/exp019_pz1_sweep_results.csv ({len(df)} rows)")

    methods = ['oracle', 'naive', 'global_corrected', 'hb_ec_hte']

    # ── Summary table: per-(p_z1, K, method) ──
    summary_rows = []
    for p_z1 in P_Z1_VALUES:
        for k in K_VALUES:
            for method in methods:
                sub = df[(df['p_z1'] == p_z1) & (df['K'] == k) & (df['method'] == method)]
                if sub.empty:
                    continue
                avg_abs_bias = sub['bias'].abs().mean()
                rmse = np.sqrt((sub['bias'] ** 2).mean())
                cov = sub['coverage'].mean()
                summary_rows.append(dict(
                    p_z1=p_z1, K=k, method=method,
                    avg_abs_bias=avg_abs_bias, rmse=rmse, coverage=cov))

    df_summary = pd.DataFrame(summary_rows)

    # ── Theorem 1 R² table ──
    r2_rows = []
    for p_z1 in P_Z1_VALUES:
        for k in K_VALUES:
            r2 = compute_theorem1_r2(df, p_z1, k)
            r2_rows.append(dict(p_z1=p_z1, K=k, theorem1_r2=r2))
    df_r2 = pd.DataFrame(r2_rows)

    # ── RMSE ratio: hb / global ──
    ratio_rows = []
    for p_z1 in P_Z1_VALUES:
        for k in K_VALUES:
            gc = df_summary[(df_summary['p_z1'] == p_z1) &
                            (df_summary['K'] == k) &
                            (df_summary['method'] == 'global_corrected')]
            hb = df_summary[(df_summary['p_z1'] == p_z1) &
                            (df_summary['K'] == k) &
                            (df_summary['method'] == 'hb_ec_hte')]
            if gc.empty or hb.empty:
                continue
            rmse_gc = gc['rmse'].values[0]
            rmse_hb = hb['rmse'].values[0]
            ratio = rmse_hb / rmse_gc if rmse_gc > 1e-10 else np.nan
            ratio_rows.append(dict(p_z1=p_z1, K=k,
                                   rmse_global=rmse_gc, rmse_hb=rmse_hb,
                                   rmse_ratio=ratio))
    df_ratio = pd.DataFrame(ratio_rows)

    # ── Generate figures ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: Theorem 1 R² vs P(Z=1)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        for k in K_VALUES:
            sub = df_r2[df_r2['K'] == k]
            ax.plot(sub['p_z1'], sub['theorem1_r2'], 'o-', label=f'K={k}', markersize=8)
        ax.set_xlabel('P(Z=1)')
        ax.set_ylabel('Theorem 1 R²')
        ax.set_title('Theorem 1 Bias Decomposition R² vs P(Z=1)')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.95, ls='--', color='gray', alpha=0.5, label='R²=0.95')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/pz1_sweep/theorem1_r2_vs_pz1.png', dpi=150)
        plt.close()

        # Fig 2: RMSE ratio vs P(Z=1)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        for k in K_VALUES:
            sub = df_ratio[df_ratio['K'] == k]
            ax.plot(sub['p_z1'], sub['rmse_ratio'], 's-', label=f'K={k}', markersize=8)
        ax.set_xlabel('P(Z=1)')
        ax.set_ylabel('RMSE ratio (HB / Global)')
        ax.set_title('EC-HTE RMSE Advantage vs P(Z=1)')
        ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='ratio=1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/pz1_sweep/rmse_ratio_vs_pz1.png', dpi=150)
        plt.close()

        # Fig 3: Coverage by method vs P(Z=1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ki, k in enumerate(K_VALUES):
            ax = axes[ki]
            for method in methods:
                sub = df_summary[(df_summary['K'] == k) & (df_summary['method'] == method)]
                if sub.empty:
                    continue
                ax.plot(sub['p_z1'], sub['coverage'], 'o-', label=method, markersize=6)
            ax.axhline(0.95, ls='--', color='gray', alpha=0.5)
            ax.set_xlabel('P(Z=1)')
            ax.set_ylabel('95% CI Coverage')
            ax.set_title(f'Coverage vs P(Z=1), K={k}')
            ax.set_ylim(0.5, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/pz1_sweep/coverage_vs_pz1.png', dpi=150)
        plt.close()

        # Fig 4: Avg |bias| heatmap-style line plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ki, k in enumerate(K_VALUES):
            ax = axes[ki]
            for method in methods:
                sub = df_summary[(df_summary['K'] == k) & (df_summary['method'] == method)]
                if sub.empty:
                    continue
                ax.plot(sub['p_z1'], sub['avg_abs_bias'], 'o-', label=method, markersize=6)
            ax.set_xlabel('P(Z=1)')
            ax.set_ylabel('Avg |Bias|')
            ax.set_title(f'Average |Bias| vs P(Z=1), K={k}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/pz1_sweep/avg_bias_vs_pz1.png', dpi=150)
        plt.close()

        print("Figures saved to artifacts/pz1_sweep/")
    except ImportError:
        print("matplotlib not available, skipping figures")

    # ── Write analysis markdown ──
    lines = []
    lines.append("# exp-019: P(Z=1) Sweep Analysis\n")
    lines.append("## Configuration")
    lines.append(f"- P(Z=1) ∈ {P_Z1_VALUES}")
    lines.append(f"- K ∈ {K_VALUES}")
    lines.append(f"- Heterogeneity: extreme (K=2: π_s ∈ {MISCLASS_RATES[2]}, K=4: π_s ∈ {MISCLASS_RATES[4]})")
    lines.append(f"- n_expert = {N_EXPERT}, N = {N}, MC seeds = {len(df['seed'].unique())}")
    lines.append(f"- τ(Z=0) = {TAU_Z0}, τ(Z=1) = {TAU_Z1}")
    lines.append(f"- Symmetric CM, T ~ Bernoulli(0.5)\n")

    lines.append("## Theorem 1 R² vs P(Z=1)\n")
    lines.append("| P(Z=1) | K=2 R² | K=4 R² |")
    lines.append("|--------|--------|--------|")
    for p_z1 in P_Z1_VALUES:
        r2_k2 = df_r2[(df_r2['p_z1'] == p_z1) & (df_r2['K'] == 2)]['theorem1_r2'].values
        r2_k4 = df_r2[(df_r2['p_z1'] == p_z1) & (df_r2['K'] == 4)]['theorem1_r2'].values
        v2 = f"{r2_k2[0]:.4f}" if len(r2_k2) > 0 and not np.isnan(r2_k2[0]) else "N/A"
        v4 = f"{r2_k4[0]:.4f}" if len(r2_k4) > 0 and not np.isnan(r2_k4[0]) else "N/A"
        lines.append(f"| {p_z1} | {v2} | {v4} |")

    lines.append("\n## RMSE Ratio (HB / Global) vs P(Z=1)\n")
    lines.append("| P(Z=1) | K=2 ratio | K=4 ratio |")
    lines.append("|--------|-----------|-----------|")
    for p_z1 in P_Z1_VALUES:
        r_k2 = df_ratio[(df_ratio['p_z1'] == p_z1) & (df_ratio['K'] == 2)]['rmse_ratio'].values
        r_k4 = df_ratio[(df_ratio['p_z1'] == p_z1) & (df_ratio['K'] == 4)]['rmse_ratio'].values
        v2 = f"{r_k2[0]:.4f}" if len(r_k2) > 0 else "N/A"
        v4 = f"{r_k4[0]:.4f}" if len(r_k4) > 0 else "N/A"
        lines.append(f"| {p_z1} | {v2} | {v4} |")

    lines.append("\n## Per-(P(Z=1), K, Method) Summary\n")
    lines.append("| P(Z=1) | K | Method | Avg|Bias| | RMSE | Coverage |")
    lines.append("|--------|---|--------|-----------|------|----------|")
    for _, row in df_summary.iterrows():
        lines.append(f"| {row['p_z1']} | {int(row['K'])} | {row['method']} | "
                      f"{row['avg_abs_bias']:.4f} | {row['rmse']:.4f} | {row['coverage']:.4f} |")

    lines.append("\n## Figures\n")
    lines.append("- `artifacts/pz1_sweep/theorem1_r2_vs_pz1.png`")
    lines.append("- `artifacts/pz1_sweep/rmse_ratio_vs_pz1.png`")
    lines.append("- `artifacts/pz1_sweep/coverage_vs_pz1.png`")
    lines.append("- `artifacts/pz1_sweep/avg_bias_vs_pz1.png`")

    md_text = '\n'.join(lines) + '\n'
    with open('results/exp019_pz1_sweep_analysis.md', 'w') as f:
        f.write(md_text)
    print("Saved results/exp019_pz1_sweep_analysis.md")

    return df_summary, df_r2, df_ratio


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=N_MC)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 5 if args.dry_run else args.n_mc

    print(f"exp-019 P(Z=1) Sweep | N={N} n_mc={n_mc} n_expert={N_EXPERT}")
    print(f"P(Z=1)={P_Z1_VALUES} K={K_VALUES}")
    print(f"Misclass K=2: {MISCLASS_RATES[2]}, K=4: {MISCLASS_RATES[4]}")
    print(f"tau(Z=0)={TAU_Z0}, tau(Z=1)={TAU_Z1}\n")

    df = run_sweep(n_mc=n_mc)
    df_summary, df_r2, df_ratio = generate_analysis(df)

    print("\n── Theorem 1 R² ──")
    print(df_r2.to_string(index=False))
    print("\n── RMSE Ratio (HB/Global) ──")
    print(df_ratio.to_string(index=False))
    print("\nDone.")
