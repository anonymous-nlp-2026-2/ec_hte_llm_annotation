"""
exp-015: Semi-Real Confusion Matrix EC-HTE Experiment.

Uses real asymmetric confusion matrices from GPT-4.1 TweetEval pilot annotations.
Ground truth CATE is synthetic, but CMs come from real LLM pilot data.

4 subgroups with real asymmetric CMs (from pilot GPT-4.1 annotations, n>=10 only):
  S0 (False_medium): FPR=0.054, FNR=0.526, weight=0.38
  S1 (False_short):  FPR=0.101, FNR=0.300, weight=0.22
  S2 (True_short):   FPR=0.143, FNR=0.308, weight=0.14
  S3 (True_medium):  FPR=0.194, FNR=0.222, weight=0.25

Output:
  results/exp015_semi_real_raw.csv
  results/exp015_semi_real_summary.csv
  results/exp015_semi_real_analysis.md

Usage: python3 exp015_semi_real_cm.py --n-mc 100
"""

import argparse
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln

warnings.filterwarnings('ignore')

# === Config ===

N = 5000
N_EXPERTS = [100, 250, 500]
SEED_BASE = 0

SUBGROUP_LABELS = ['S0', 'S1', 'S2', 'S3']
SUBGROUP_WEIGHTS = np.array([0.38, 0.22, 0.14, 0.25])
SUBGROUP_WEIGHTS = SUBGROUP_WEIGHTS / SUBGROUP_WEIGHTS.sum()

# Real asymmetric CMs from GPT-4.1 pilot (FPR, FNR per subgroup)
SUBGROUP_CM = {
    'S0': {'fpr': 0.054, 'fnr': 0.526},
    'S1': {'fpr': 0.101, 'fnr': 0.300},
    'S2': {'fpr': 0.143, 'fnr': 0.308},
    'S3': {'fpr': 0.194, 'fnr': 0.222},
}

P_Z1 = 0.18  # P(Z=1) from pilot overall
P_T1 = 0.5


def get_cm_matrix(fpr, fnr):
    """Build 2x2 confusion matrix: M[z_true, z_obs]."""
    return np.array([[1 - fpr, fpr],
                     [fnr, 1 - fnr]])


# True CATE by (z, subgroup)
def true_tau(z, s_idx):
    if z == 0:
        return 0.3 + 0.1 * s_idx
    else:
        return 0.8 - 0.15 * s_idx


# === Core Estimators ===

def diff_in_means(Y, T, mask):
    t1 = mask & (T == 1)
    t0 = mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    tau_hat = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
    return tau_hat, se


def estimate_confusion_matrix(z_true, z_hat):
    C = np.zeros((2, 2))
    for zt in [0, 1]:
        mask = z_true == zt
        n = mask.sum()
        for zh in [0, 1]:
            C[zt, zh] = ((z_hat[mask] == zh).sum() + 1) / (n + 2) if n > 0 else 0.5
    return C


def build_mixing_matrix(C, p_z, p_z_hat):
    """M[z_hat, z] = P(Z=z | Z_hat=z_hat) via Bayes' rule."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        for z in [0, 1]:
            M[zh, z] = C[z, zh] * p_z[z] / p_z_hat[zh] if p_z_hat[zh] > 0 else 0.5
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
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


# === HB Gibbs Sampler ===

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


def _split_rhat(chains):
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


def _log_target_alpha(alpha, theta_z, hp_a0=3.0, hp_b0=1.0):
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, labels, seed=42, n_iter=1000, n_warmup=500, n_chains=4,
                         hp_a0=3.0, hp_b0=1.0):
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
                post_a = counts[:, z, :].astype(float) + alpha[z] / 2.0
                post_a = np.maximum(post_a, 1e-6)
                g = rng.gamma(post_a)
                g_sum = g.sum(axis=1, keepdims=True)
                theta[:, z, :] = g / np.maximum(g_sum, 1e-300)

            for z in [0, 1]:
                x_prop = np.log(alpha[z]) + rng.normal(0, mh_scale[z])
                a_prop = np.exp(x_prop)
                log_r = (_log_target_alpha(a_prop, theta[:, z, :], hp_a0, hp_b0) -
                         _log_target_alpha(alpha[z], theta[:, z, :], hp_a0, hp_b0))
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

    return C, {'runtime_seconds': runtime}


# === DGP ===

def generate_data(n_expert, seed):
    """Generate one MC dataset with real asymmetric CMs."""
    rng = np.random.RandomState(seed)

    # Assign subgroups
    S_idx = rng.choice(4, size=N, p=SUBGROUP_WEIGHTS)
    S = np.array([SUBGROUP_LABELS[i] for i in S_idx])

    # Draw Z_true
    Z = rng.binomial(1, P_Z1, N)

    # Draw T
    T = rng.binomial(1, P_T1, N)

    # Generate Z_obs from Z_true using subgroup-specific CM
    Z_hat = np.empty(N, dtype=int)
    for i in range(4):
        s_mask = S_idx == i
        cm = get_cm_matrix(SUBGROUP_CM[SUBGROUP_LABELS[i]]['fpr'],
                           SUBGROUP_CM[SUBGROUP_LABELS[i]]['fnr'])
        n_s = s_mask.sum()
        z_s = Z[s_mask]
        # For each unit: P(Z_obs=1 | Z_true=z, S=s) = cm[z, 1]
        p_obs1 = np.where(z_s == 0, cm[0, 1], cm[1, 1])
        Z_hat[s_mask] = rng.binomial(1, p_obs1)

    # Generate Y with heterogeneous treatment effects
    tau_vals = np.zeros(N)
    for i in range(4):
        s_mask = S_idx == i
        tau_vals[s_mask & (Z == 0)] = true_tau(0, i)
        tau_vals[s_mask & (Z == 1)] = true_tau(1, i)

    eps = rng.randn(N)
    Y = 1.0 + tau_vals * T + eps

    # Expert mask
    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'S_idx': S_idx,
        'Y': Y, 'tau': tau_vals, 'expert_mask': expert_mask,
    }


# === MC Runner ===

def run_one_mc(data, n_expert, seed):
    """Run one MC iteration: all methods x all (subgroup, z_level) combos."""
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    S_idx = data['S_idx']
    expert_mask = data['expert_mask']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    # Global confusion matrix from expert subset
    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    # HB Dirichlet confusion matrices (partial pooling via Gibbs sampler)
    counts = get_counts(z_exp, zh_exp, s_exp, SUBGROUP_LABELS)
    C_hb, _ = estimate_cm_hb_gibbs(counts, SUBGROUP_LABELS, seed=seed)

    M_hb_sub = {}
    for i, s_val in enumerate(SUBGROUP_LABELS):
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        if sm_exp.sum() >= 4:
            p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()])
        else:
            p_z_s = p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_hb_sub[s_val] = build_mixing_matrix(C_hb[s_val], p_z_s, p_zh_s)

    rows = []

    for s_i, s_val in enumerate(SUBGROUP_LABELS):
        s_mask = S == s_val

        for z_level in [0, 1]:
            oracle_cate = true_tau(z_level, s_i)

            # Oracle: uses Z_true
            th_or, se_or = diff_in_means(Y, T, s_mask & (Z == z_level))
            ci_lo_or = th_or - 1.96 * se_or if not np.isnan(se_or) else np.nan
            ci_hi_or = th_or + 1.96 * se_or if not np.isnan(se_or) else np.nan
            cov_or = 1 if (not np.isnan(ci_lo_or) and ci_lo_or <= oracle_cate <= ci_hi_or) else 0
            rows.append({
                'seed': seed, 'n_expert': n_expert, 'subgroup': s_val,
                'z_level': z_level, 'method': 'oracle',
                'cate': th_or, 'oracle_cate': oracle_cate,
                'bias': th_or - oracle_cate if not np.isnan(th_or) else np.nan,
                'ci_lo': ci_lo_or, 'ci_hi': ci_hi_or,
                'coverage': cov_or,
            })

            # Naive: uses Z_obs
            th_nv, se_nv = diff_in_means(Y, T, s_mask & (Z_hat == z_level))
            ci_lo_nv = th_nv - 1.96 * se_nv if not np.isnan(se_nv) else np.nan
            ci_hi_nv = th_nv + 1.96 * se_nv if not np.isnan(se_nv) else np.nan
            cov_nv = 1 if (not np.isnan(ci_lo_nv) and ci_lo_nv <= oracle_cate <= ci_hi_nv) else 0
            rows.append({
                'seed': seed, 'n_expert': n_expert, 'subgroup': s_val,
                'z_level': z_level, 'method': 'naive',
                'cate': th_nv, 'oracle_cate': oracle_cate,
                'bias': th_nv - oracle_cate if not np.isnan(th_nv) else np.nan,
                'ci_lo': ci_lo_nv, 'ci_hi': ci_hi_nv,
                'coverage': cov_nv,
            })

            # Global corrected
            tau_obs = np.array([
                diff_in_means(Y, T, s_mask & (Z_hat == 0))[0],
                diff_in_means(Y, T, s_mask & (Z_hat == 1))[0],
            ])
            se_obs = np.array([
                diff_in_means(Y, T, s_mask & (Z_hat == 0))[1],
                diff_in_means(Y, T, s_mask & (Z_hat == 1))[1],
            ])
            if not np.any(np.isnan(tau_obs)):
                tau_gc, se_gc = invert_mixing_safe(M_global, tau_obs, se_obs)
                th_gc = tau_gc[z_level]
                se_gc_z = se_gc[z_level]
            else:
                th_gc, se_gc_z = np.nan, np.nan
            ci_lo_gc = th_gc - 1.96 * se_gc_z if not np.isnan(se_gc_z) else np.nan
            ci_hi_gc = th_gc + 1.96 * se_gc_z if not np.isnan(se_gc_z) else np.nan
            cov_gc = 1 if (not np.isnan(ci_lo_gc) and ci_lo_gc <= oracle_cate <= ci_hi_gc) else 0
            rows.append({
                'seed': seed, 'n_expert': n_expert, 'subgroup': s_val,
                'z_level': z_level, 'method': 'global_corrected',
                'cate': th_gc, 'oracle_cate': oracle_cate,
                'bias': th_gc - oracle_cate if not np.isnan(th_gc) else np.nan,
                'ci_lo': ci_lo_gc, 'ci_hi': ci_hi_gc,
                'coverage': cov_gc,
            })

            # HB EC-HTE
            if not np.any(np.isnan(tau_obs)):
                tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], tau_obs, se_obs)
                th_hb = tau_hb[z_level]
                se_hb_z = se_hb[z_level]
            else:
                th_hb, se_hb_z = np.nan, np.nan
            ci_lo_hb = th_hb - 1.96 * se_hb_z if not np.isnan(se_hb_z) else np.nan
            ci_hi_hb = th_hb + 1.96 * se_hb_z if not np.isnan(se_hb_z) else np.nan
            cov_hb = 1 if (not np.isnan(ci_lo_hb) and ci_lo_hb <= oracle_cate <= ci_hi_hb) else 0
            rows.append({
                'seed': seed, 'n_expert': n_expert, 'subgroup': s_val,
                'z_level': z_level, 'method': 'hb_ec_hte',
                'cate': th_hb, 'oracle_cate': oracle_cate,
                'bias': th_hb - oracle_cate if not np.isnan(th_hb) else np.nan,
                'ci_lo': ci_lo_hb, 'ci_hi': ci_hi_hb,
                'coverage': cov_hb,
            })

    return rows


def aggregate_results(df_raw):
    """Aggregate raw results to summary statistics."""
    group_cols = ['n_expert', 'subgroup', 'z_level', 'method']
    rows = []
    for keys, grp in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        valid = grp['bias'].dropna()
        if len(valid) == 0:
            continue
        row['mean_bias'] = valid.mean()
        row['mean_abs_bias'] = valid.abs().mean()
        row['mean_rmse'] = np.sqrt((valid ** 2).mean())
        row['mean_coverage'] = grp['coverage'].mean()
        row['se_bias'] = valid.std() / np.sqrt(len(valid))
        row['se_rmse'] = row['mean_rmse'] / np.sqrt(2 * len(valid))
        rows.append(row)
    return pd.DataFrame(rows)


def write_analysis(df_summary):
    """Write analysis markdown report."""
    lines = []
    lines.append("# exp-015: Semi-Real Confusion Matrix EC-HTE Experiment\n")
    lines.append("## Design\n")
    lines.append("Uses **real asymmetric confusion matrices** from GPT-4.1 TweetEval pilot annotations.")
    lines.append("Ground truth CATE is synthetic; confusion matrices are empirical.\n")
    lines.append("| Subgroup | Weight | FPR | FNR |")
    lines.append("|----------|--------|-----|-----|")
    for i, s in enumerate(SUBGROUP_LABELS):
        cm = SUBGROUP_CM[s]
        lines.append(f"| {s} | {SUBGROUP_WEIGHTS[i]:.2f} | {cm['fpr']:.3f} | {cm['fnr']:.3f} |")
    lines.append("")
    lines.append("**True CATE**: tau(z=0,s) = 0.3 + 0.1*s_idx; tau(z=1,s) = 0.8 - 0.15*s_idx\n")
    lines.append("| Subgroup | tau(z=0) | tau(z=1) |")
    lines.append("|----------|----------|----------|")
    for i, s in enumerate(SUBGROUP_LABELS):
        lines.append(f"| {s} | {true_tau(0,i):.2f} | {true_tau(1,i):.2f} |")
    lines.append("")
    lines.append("N=5000, P(Z=1)=0.18, P(T=1)=0.5, n_expert in {100, 250, 500}, 100 MC seeds.\n")

    lines.append("## Summary Tables\n")
    for ne in N_EXPERTS:
        lines.append(f"### n_expert = {ne}\n")
        sub = df_summary[df_summary['n_expert'] == ne].copy()
        sub = sub.sort_values(['subgroup', 'z_level', 'method'])
        lines.append("| Subgroup | Z | Method | Mean Bias | Mean |Bias| | RMSE | Coverage |")
        lines.append("|----------|---|--------|-----------|-------------|------|----------|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['subgroup']} | {int(r['z_level'])} | {r['method']} | "
                         f"{r['mean_bias']:+.4f} | {r['mean_abs_bias']:.4f} | "
                         f"{r['mean_rmse']:.4f} | {r['mean_coverage']:.3f} |")
        lines.append("")

    # Key comparisons
    lines.append("## Key Comparisons\n")
    lines.append("### HB EC-HTE vs Global Corrected: Bias Reduction\n")
    lines.append("| n_expert | Subgroup | Z | HB Bias | Global Bias | Reduction |")
    lines.append("|----------|----------|---|---------|-------------|-----------|")
    for ne in N_EXPERTS:
        for s in SUBGROUP_LABELS:
            for z in [0, 1]:
                hb = df_summary[(df_summary['n_expert'] == ne) & (df_summary['subgroup'] == s) &
                                (df_summary['z_level'] == z) & (df_summary['method'] == 'hb_ec_hte')]
                gc = df_summary[(df_summary['n_expert'] == ne) & (df_summary['subgroup'] == s) &
                                (df_summary['z_level'] == z) & (df_summary['method'] == 'global_corrected')]
                if len(hb) > 0 and len(gc) > 0:
                    hb_b = hb.iloc[0]['mean_abs_bias']
                    gc_b = gc.iloc[0]['mean_abs_bias']
                    red = (gc_b - hb_b) / gc_b * 100 if gc_b > 0 else 0
                    lines.append(f"| {ne} | {s} | {z} | {hb_b:.4f} | {gc_b:.4f} | {red:+.1f}% |")
    lines.append("")

    # Coverage comparison
    lines.append("### Coverage (95% CI)\n")
    lines.append("| n_expert | Method | Mean Coverage |")
    lines.append("|----------|--------|---------------|")
    for ne in N_EXPERTS:
        for m in ['oracle', 'naive', 'global_corrected', 'hb_ec_hte']:
            sub = df_summary[(df_summary['n_expert'] == ne) & (df_summary['method'] == m)]
            if len(sub) > 0:
                lines.append(f"| {ne} | {m} | {sub['mean_coverage'].mean():.3f} |")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion\n")
    lines.append("Does EC-HTE maintain advantage under realistic asymmetric CMs?\n")

    # Compute overall advantage
    for ne in N_EXPERTS:
        hb_all = df_summary[(df_summary['n_expert'] == ne) & (df_summary['method'] == 'hb_ec_hte')]
        gc_all = df_summary[(df_summary['n_expert'] == ne) & (df_summary['method'] == 'global_corrected')]
        nv_all = df_summary[(df_summary['n_expert'] == ne) & (df_summary['method'] == 'naive')]
        if len(hb_all) > 0 and len(gc_all) > 0:
            hb_rmse = hb_all['mean_rmse'].mean()
            gc_rmse = gc_all['mean_rmse'].mean()
            nv_rmse = nv_all['mean_rmse'].mean()
            lines.append(f"- n_expert={ne}: HB RMSE={hb_rmse:.4f}, Global RMSE={gc_rmse:.4f}, "
                         f"Naive RMSE={nv_rmse:.4f}")

    report = "\n".join(lines)
    with open('results/exp015_semi_real_analysis.md', 'w') as f:
        f.write(report)
    print("Wrote results/exp015_semi_real_analysis.md")


# === Main ===

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=100)
    args = parser.parse_args()
    n_mc = args.n_mc

    print(f"exp-015: Semi-Real Confusion Matrix EC-HTE")
    print(f"N={N}, n_mc={n_mc}, n_experts={N_EXPERTS}")
    print(f"Subgroup CMs (real asymmetric from GPT-4.1 pilot):")
    for s in SUBGROUP_LABELS:
        cm = SUBGROUP_CM[s]
        print(f"  {s}: FPR={cm['fpr']:.3f}, FNR={cm['fnr']:.3f}")
    print(f"Methods: oracle, naive, global_corrected, hb_ec_hte")
    print()

    results = []
    configs = [(ne, mc) for ne in N_EXPERTS for mc in range(n_mc)]
    total = len(configs)
    t0 = time.time()

    for count, (ne, mc) in enumerate(configs, 1):
        seed = SEED_BASE + mc
        data = generate_data(ne, seed)
        mc_rows = run_one_mc(data, ne, seed)
        results.extend(mc_rows)
        if count % max(1, total // 20) == 0:
            elapsed = time.time() - t0
            eta = elapsed / count * (total - count)
            print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | ne={ne} seed={seed}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    df_raw = pd.DataFrame(results)
    df_raw['abs_bias'] = df_raw['bias'].abs()
    df_raw['squared_error'] = df_raw['bias'] ** 2

    df_summary = aggregate_results(df_raw)

    # Save
    df_raw.to_csv('results/exp015_semi_real_raw.csv', index=False)
    df_summary.to_csv('results/exp015_semi_real_summary.csv', index=False)
    print(f"Saved results/exp015_semi_real_raw.csv ({len(df_raw)} rows)")
    print(f"Saved results/exp015_semi_real_summary.csv ({len(df_summary)} rows)")

    write_analysis(df_summary)

    # Print key metrics
    print("\n=== Key Metrics Summary ===\n")
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 200)
    for ne in N_EXPERTS:
        print(f"--- n_expert = {ne} ---")
        sub = df_summary[df_summary['n_expert'] == ne]
        for m in ['naive', 'global_corrected', 'hb_ec_hte']:
            ms = sub[sub['method'] == m]
            print(f"  {m:20s}: mean_abs_bias={ms['mean_abs_bias'].mean():.4f}, "
                  f"rmse={ms['mean_rmse'].mean():.4f}, coverage={ms['mean_coverage'].mean():.3f}")
        print()
