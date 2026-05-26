#!/usr/bin/env python3
"""exp_treatment_dep_misclass.py — Treatment-dependent misclassification sensitivity

Tests EC-HTE robustness when P(Ẑ|Z,T) ≠ P(Ẑ|Z).
Violation: P(Ẑ≠Z | Z,S,T) = π_s + γ_T·(T - 0.5)

DGP: K=2, pi=[0.05,0.25], tau(z=0)=0.1, tau(z=1)=0.5, N=5000, n_expert=250
γ_T ∈ {0, 0.05, 0.10, 0.15, 0.20}
100 MC runs per γ_T

Methods: naive, global_corrected, hb_ec_hte (centered EB)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

N = 5000
K = 2
N_EXPERT = 250
PI = [0.05, 0.25]
TAU_TRUE = {0: 0.1, 1: 0.5}
GAMMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
N_MC = 100
LABELS = ['A', 'B']

ARTIFACTS_DIR = os.path.expanduser(
    '~/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts')


def generate_data(seed, gamma_T):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=[0.5, 0.5])
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    tau = np.array([TAU_TRUE[z] for z in Z_true])
    baseline = 1.0 + 0.5 * Z_true
    Y = baseline + tau * T + rng.normal(0, 1, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        pi_eff = PI[s] + gamma_T * (T[mask_s] - 0.5)
        pi_eff = np.clip(pi_eff, 0.01, 0.49)
        flip = rng.binomial(1, pi_eff)
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    expert_idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


# ── CM estimation ───────────────────────────────────────────────────────────

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


# ── DIM estimator ───────────────────────────────────────────────────────────

def diff_in_means(Y, T, mask):
    t1, t0 = mask & (T == 1), mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


# ── Single MC run ───────────────────────────────────────────────────────────

def run_one_mc(seed, gamma_T):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(seed, gamma_T)

    S_label = np.array([LABELS[s] for s in S])
    z_exp = Z_true[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S_label[expert_mask]

    counts = get_counts(z_exp, zh_exp, s_exp, LABELS)
    global_cm = compute_global_cm(counts)
    M_gl = build_mixing_matrix(global_cm)

    C_eb, _ = estimate_cm_centered_eb(counts, LABELS, global_cm)
    M_eb = {s: build_mixing_matrix(C_eb[s]) for s in LABELS}

    rows = []
    for si, s in enumerate(LABELS):
        sm = S_label == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))

        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if np.any(np.isnan(tau_obs)) or np.any(np.isnan(se_obs)):
            continue

        tau_gl, se_gl = invert_mixing(M_gl, tau_obs, se_obs)
        tau_eb, se_eb = invert_mixing(M_eb[s], tau_obs, se_obs)

        methods = [
            ('naive', tau_obs, se_obs),
            ('global_corrected', tau_gl, se_gl),
            ('hb_ec_hte', tau_eb, se_eb),
        ]

        for method_name, tau_hat_vec, se_hat_vec in methods:
            for zi, z in enumerate([0, 1]):
                th = tau_hat_vec[zi]
                se = se_hat_vec[zi]
                ci_lo = th - 1.96 * se
                ci_hi = th + 1.96 * se
                covers = int(ci_lo <= TAU_TRUE[z] <= ci_hi)
                rows.append({
                    'gamma_T': gamma_T,
                    'mc_seed': seed,
                    'subgroup': s,
                    'z_level': z,
                    'method': method_name,
                    'tau_hat': th,
                    'tau_true': TAU_TRUE[z],
                    'bias': th - TAU_TRUE[z],
                    'se': se,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'covers': covers,
                })

    return rows


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    all_rows = []
    for gamma in GAMMAS:
        print(f"gamma_T={gamma:.2f} ...", end=' ', flush=True)
        gt = time.time()
        for mc in range(N_MC):
            seed = 42 + mc
            rows = run_one_mc(seed, gamma)
            all_rows.extend(rows)
        print(f"done ({time.time()-gt:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_treatment_dep_misclass.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_treatment_dep_misclass.csv")

    # ── Summary ──
    summary = df.groupby(['gamma_T', 'method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
        n=('covers', 'count'),
    ).reset_index()

    summary.to_csv(os.path.join(ARTIFACTS_DIR, 'treatment_dep_misclass_summary.csv'), index=False)
    print(f"Wrote summary to artifacts/treatment_dep_misclass_summary.csv")

    # ── Print table ──
    print("\n" + "="*80)
    print("TREATMENT-DEPENDENT MISCLASSIFICATION — AGGREGATE RESULTS")
    print("="*80)
    print(f"{'gamma_T':>8} {'Method':>18} {'|Bias|':>8} {'RMSE':>8} {'Cov':>6} {'Mean SE':>8}")
    print("-"*60)
    for _, row in summary.iterrows():
        print(f"{row['gamma_T']:8.2f} {row['method']:>18} {row['mean_abs_bias']:8.4f} "
              f"{row['rmse']:8.4f} {row['coverage']:6.3f} {row['mean_se']:8.4f}")

    # ── Degradation relative to gamma_T=0 ──
    print("\n" + "="*80)
    print("DEGRADATION RELATIVE TO gamma_T=0")
    print("="*80)
    baseline = summary[summary['gamma_T'] == 0.0].set_index('method')
    for gamma in GAMMAS[1:]:
        print(f"\ngamma_T={gamma:.2f}:")
        current = summary[summary['gamma_T'] == gamma].set_index('method')
        for method in ['naive', 'global_corrected', 'hb_ec_hte']:
            if method in current.index and method in baseline.index:
                b0 = baseline.loc[method, 'mean_abs_bias']
                b1 = current.loc[method, 'mean_abs_bias']
                c0 = baseline.loc[method, 'coverage']
                c1 = current.loc[method, 'coverage']
                print(f"  {method:>18}: |Bias| {b0:.4f}->{b1:.4f} "
                      f"({(b1-b0)/max(b0,1e-6)*100:+.1f}%), "
                      f"Cov {c0:.3f}->{c1:.3f} ({(c1-c0)*100:+.1f}pp)")

    # ── Coverage threshold ──
    print("\n" + "="*80)
    print("COVERAGE THRESHOLDS")
    print("="*80)
    for method in ['naive', 'global_corrected', 'hb_ec_hte']:
        method_data = summary[summary['method'] == method]
        below_93 = method_data[method_data['coverage'] < 0.93]
        below_90 = method_data[method_data['coverage'] < 0.90]
        if len(below_93) > 0:
            g93 = below_93['gamma_T'].min()
            print(f"  {method:>18}: first < 93% at gamma_T={g93:.2f}")
        else:
            print(f"  {method:>18}: always >= 93%")
        if len(below_90) > 0:
            g90 = below_90['gamma_T'].min()
            print(f"  {method:>18}: first < 90% at gamma_T={g90:.2f}")

    # ── Key numbers ──
    print("\n" + "="*80)
    print("KEY NUMBERS")
    print("="*80)
    g15 = summary[summary['gamma_T'] == 0.15]
    for method in ['naive', 'global_corrected', 'hb_ec_hte']:
        row_m = g15[g15['method'] == method]
        if not row_m.empty:
            r = row_m.iloc[0]
            print(f"  gamma_T=0.15 {method:>18}: |Bias|={r['mean_abs_bias']:.4f}, "
                  f"RMSE={r['rmse']:.4f}, Cov={r['coverage']:.3f}")

    # EC-HTE vs global RMSE comparison
    print("\n  EC-HTE vs Global RMSE ratio by gamma_T:")
    for gamma in GAMMAS:
        ec = summary[(summary['gamma_T'] == gamma) & (summary['method'] == 'hb_ec_hte')]
        gl = summary[(summary['gamma_T'] == gamma) & (summary['method'] == 'global_corrected')]
        if not ec.empty and not gl.empty:
            ratio = ec.iloc[0]['rmse'] / gl.iloc[0]['rmse']
            print(f"    gamma_T={gamma:.2f}: EC-HTE/Global = {ratio:.3f}")

    # ── Visualization ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = [('mean_abs_bias', '|Bias|'), ('rmse', 'RMSE'), ('coverage', 'Coverage')]
        method_styles = {
            'naive': ('o-', '#d62728', 'Naive'),
            'global_corrected': ('s--', '#1f77b4', 'Global Corrected'),
            'hb_ec_hte': ('^-', '#2ca02c', 'EC-HTE (Centered EB)'),
        }

        for ax, (metric, label) in zip(axes, metrics):
            for method, (style, color, name) in method_styles.items():
                mdata = summary[summary['method'] == method].sort_values('gamma_T')
                ax.plot(mdata['gamma_T'], mdata[metric], style, color=color,
                        label=name, markersize=7, linewidth=2)
            ax.set_xlabel(r'$\gamma_T$ (treatment-dependent misclass.)', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(label, fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            if metric == 'coverage':
                ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
                ax.axhline(y=0.93, color='gray', linestyle='--', alpha=0.3)
                ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))

        fig.suptitle('Treatment-Dependent Misclassification Sensitivity\n'
                     r'$P(\hat{Z}\neq Z|Z,S,T) = \pi_s + \gamma_T \cdot (T - 0.5)$',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        for ext in ['pdf', 'png']:
            path = os.path.join(ARTIFACTS_DIR, f'treatment_dep_misclass_figure.{ext}')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f"Saved {path}")

        plt.close(fig)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
