#!/usr/bin/env python3
"""exp_bootstrap_small_n.py — Bootstrap coverage under small n_expert

Tests whether bootstrap SE recovers nominal 95% coverage when the expert
sample is small: n_expert ∈ {25, 50, 75, 100, 200}.

DGP: K=2, π=[0.05, 0.25], τ(0)=0.0, τ(1)=0.30, N=5000
Methods: Naive, Global, Strat MLE, Centered EB EC-HTE
Estimator: CausalForestDML (n_estimators=200)
20 MC runs × 100 bootstrap resamples per run

Output:
  results/exp_bootstrap_small_n.csv          (raw per-MC-run results)
  results/exp_bootstrap_small_n_summary.csv  (aggregated summary)
  results/exp_bootstrap_small_n_analysis.md  (analysis report)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False

N = 5000
K = 2
N_EXPERT_LIST = [25, 50, 75, 100, 200]
PI = [0.05, 0.25]
TAU_Z = {0: 0.0, 1: 0.30}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 20
N_BOOTSTRAP = 100
N_ESTIMATORS = 200
METHODS = ['naive', 'global', 'strat_mle', 'hb_echte']


def generate_data(n_expert, seed):
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng.binomial(1, 0.5, N)
    T = rng.binomial(1, 0.5, N)

    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, PI[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_hat[mask_s])

    tau = np.array([TAU_Z[z] for z in Z_true])
    Y = 1.0 + 0.5 * Z_true + tau * T + rng.normal(0, 1, N)

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


# ── CM estimation ──────────────────────────────────────────────────────────

def get_counts(z_true, z_hat, S_vals):
    counts = np.zeros((K, 2, 2), dtype=int)
    for s in range(K):
        ms = S_vals == s
        for z in [0, 1]:
            mz = ms & (z_true == z)
            counts[s, z, 0] = (z_hat[mz] == 0).sum()
            counts[s, z, 1] = (z_hat[mz] == 1).sum()
    return counts


def compute_global_cm(counts):
    cg = counts.sum(axis=0)
    C = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    return C


def estimate_cm_strat_mle(counts):
    cms = {}
    for s in range(K):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            n = counts[s, z, :].sum()
            for zh in [0, 1]:
                C_s[z, zh] = (counts[s, z, zh] + 1) / (n + 2) if n > 0 else 0.5
        cms[s] = C_s
    return cms


def _dirmult_neg_ll(alpha, counts_z, theta_g):
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


def estimate_cm_centered_eb(counts, global_cm):
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        theta_g = global_cm[z]
        cl = [(int(counts[s, z, 0]), int(counts[s, z, 1])) for s in range(K)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a, tg=theta_g: _dirmult_neg_ll(a, cl, tg),
            bounds=(0.01, 5000), method='bounded')
        alpha_opt[z] = res.x

    cms = {}
    for s in range(K):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            a = alpha_opt[z]
            n = counts[s, z, :].sum()
            d = n + a
            C_s[z, 0] = (counts[s, z, 0] + a * global_cm[z, 0]) / d if d > 0 else 0.5
            C_s[z, 1] = (counts[s, z, 1] + a * global_cm[z, 1]) / d if d > 0 else 0.5
        cms[s] = C_s
    return cms


def estimate_all_cms(z_true_exp, z_hat_exp, S_exp):
    counts = get_counts(z_true_exp, z_hat_exp, S_exp)
    global_cm = compute_global_cm(counts)
    return {
        'global': {s: global_cm for s in range(K)},
        'strat_mle': estimate_cm_strat_mle(counts),
        'hb_echte': estimate_cm_centered_eb(counts, global_cm),
    }


# ── Mixing matrix inversion ───────────────────────────────────────────────

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


def invert_cm(M, tau_obs, se_obs, cond_threshold=100):
    cond = np.linalg.cond(M)
    if cond > cond_threshold or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


# ── CausalForest ───────────────────────────────────────────────────────────

def fit_cf(Y, T, X_feat, seed):
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=20,
        random_state=seed,
    )
    cf.fit(Y, T, X=X_feat, W=None)
    return cf


def get_subgroup_cates(cf, X_feat, Z_labels, S):
    results = {}
    for s in range(K):
        for z in [0, 1]:
            mask = (S == s) & (Z_labels == z)
            if mask.sum() < 5:
                results[(s, z)] = (np.nan, np.nan)
                continue
            te = cf.effect(X_feat[mask])
            tau_hat = te.mean()
            try:
                inf = cf.effect_inference(X_feat[mask])
                se_raw = inf.stderr
                if hasattr(se_raw, 'values'):
                    se_raw = se_raw.values
                se_hat = np.mean(se_raw.flatten())
            except Exception:
                se_hat = te.std() / np.sqrt(len(te))
            results[(s, z)] = (tau_hat, se_hat)
    return results


# ── Single MC run ──────────────────────────────────────────────────────────

def run_one_mc(n_expert, seed):
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(n_expert, seed)
    X_feat = np.column_stack([Z_hat, S])

    all_cms = estimate_all_cms(Z_true[expert_mask], Z_hat[expert_mask], S[expert_mask])

    cf = fit_cf(Y, T, X_feat, seed)
    raw_cates = get_subgroup_cates(cf, X_feat, Z_hat, S)

    plugin_corrected = {}
    for method in METHODS:
        plugin_corrected[method] = {}
        for s in range(K):
            if (s, 0) not in raw_cates or (s, 1) not in raw_cates:
                continue
            t0, t1 = raw_cates[(s, 0)], raw_cates[(s, 1)]
            if np.isnan(t0[0]) or np.isnan(t1[0]):
                continue
            tau_obs = np.array([t0[0], t1[0]])
            se_obs = np.array([t0[1], t1[1]])

            if method == 'naive':
                plugin_corrected[method][s] = {
                    0: (tau_obs[0], se_obs[0]),
                    1: (tau_obs[1], se_obs[1]),
                }
            else:
                M = build_mixing_matrix(all_cms[method][s])
                tau_c, se_c = invert_cm(M, tau_obs, se_obs)
                plugin_corrected[method][s] = {
                    0: (tau_c[0], se_c[0]),
                    1: (tau_c[1], se_c[1]),
                }

    # ── Bootstrap ──
    rng = np.random.RandomState(seed + 10000)
    boot_estimates = {
        m: {(s, z): [] for s in range(K) for z in [0, 1]}
        for m in METHODS
    }

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(N, N, replace=True)
        Y_b, T_b, Z_hat_b, S_b = Y[idx], T[idx], Z_hat[idx], S[idx]
        Z_true_b = Z_true[idx]
        expert_mask_b = expert_mask[idx]

        z_true_exp_b = Z_true_b[expert_mask_b]
        z_hat_exp_b = Z_hat_b[expert_mask_b]
        S_exp_b = S_b[expert_mask_b]

        if any((S_exp_b == s).sum() < 2 for s in range(K)):
            continue

        try:
            all_cms_b = estimate_all_cms(z_true_exp_b, z_hat_exp_b, S_exp_b)
        except Exception:
            continue

        X_feat_b = np.column_stack([Z_hat_b, S_b])
        try:
            cf_b = fit_cf(Y_b, T_b, X_feat_b, seed * 1000 + b)
        except Exception:
            continue

        for s in range(K):
            tau_obs_list = []
            for z in [0, 1]:
                mask_sz = (S_b == s) & (Z_hat_b == z)
                if mask_sz.sum() < 5:
                    tau_obs_list.append(np.nan)
                    continue
                te = cf_b.effect(X_feat_b[mask_sz])
                tau_obs_list.append(te.mean())

            if any(np.isnan(t) for t in tau_obs_list):
                continue

            tau_obs_b = np.array(tau_obs_list)
            se_dummy = np.array([0.1, 0.1])

            for zi, z in enumerate([0, 1]):
                boot_estimates['naive'][(s, z)].append(tau_obs_b[zi])

            for method in ['global', 'strat_mle', 'hb_echte']:
                M_b = build_mixing_matrix(all_cms_b[method][s])
                tau_c_b, _ = invert_cm(M_b, tau_obs_b, se_dummy)
                for zi, z in enumerate([0, 1]):
                    boot_estimates[method][(s, z)].append(tau_c_b[zi])

    boot_se = {}
    for method in METHODS:
        boot_se[method] = {}
        for key, vals in boot_estimates[method].items():
            boot_se[method][key] = np.std(vals, ddof=1) if len(vals) >= 10 else np.nan

    # ── Build output rows ──
    rows = []
    for method in METHODS:
        for s in range(K):
            if s not in plugin_corrected[method]:
                continue
            for z in [0, 1]:
                true_tau = TAU_Z[z]
                tau_hat = plugin_corrected[method][s][z][0]
                se_p = plugin_corrected[method][s][z][1]

                ci_lo_p = tau_hat - 1.96 * se_p
                ci_hi_p = tau_hat + 1.96 * se_p
                covers_p = int(ci_lo_p <= true_tau <= ci_hi_p)

                se_b = boot_se[method].get((s, z), np.nan)
                if not np.isnan(se_b):
                    ci_lo_b = tau_hat - 1.96 * se_b
                    ci_hi_b = tau_hat + 1.96 * se_b
                    covers_b = int(ci_lo_b <= true_tau <= ci_hi_b)
                else:
                    covers_b = np.nan

                rows.append({
                    'mc_seed': seed,
                    'n_expert': n_expert,
                    'subgroup': s,
                    'z_level': z,
                    'pi': PI[s],
                    'method': method,
                    'tau_true': true_tau,
                    'tau_hat': tau_hat,
                    'bias': tau_hat - true_tau,
                    'se_plugin': se_p,
                    'covers_plugin': covers_p,
                    'se_bootstrap': se_b,
                    'covers_bootstrap': covers_b,
                    'ci_width_plugin': 2 * 1.96 * se_p,
                    'ci_width_bootstrap': 2 * 1.96 * se_b if not np.isnan(se_b) else np.nan,
                })
    return rows


# ── Aggregation & reporting ────────────────────────────────────────────────

def build_summary(df):
    agg_rows = []
    sg_rows = []

    for ne in N_EXPERT_LIST:
        for method in METHODS:
            sub = df[(df['n_expert'] == ne) & (df['method'] == method)]
            if sub.empty:
                continue
            agg_rows.append({
                'n_expert': ne,
                'method': method,
                'breakdown': 'aggregate',
                'coverage_plugin': sub['covers_plugin'].mean(),
                'coverage_bootstrap': sub['covers_bootstrap'].dropna().mean()
                    if sub['covers_bootstrap'].notna().any() else np.nan,
                'mean_se_plugin': sub['se_plugin'].mean(),
                'mean_se_bootstrap': sub['se_bootstrap'].dropna().mean()
                    if sub['se_bootstrap'].notna().any() else np.nan,
                'mean_abs_bias': sub['bias'].abs().mean(),
                'n_obs': len(sub),
            })

            for s in range(K):
                ss = sub[sub['subgroup'] == s]
                if ss.empty:
                    continue
                sg_rows.append({
                    'n_expert': ne,
                    'method': method,
                    'breakdown': f's{s}_pi{PI[s]}',
                    'coverage_plugin': ss['covers_plugin'].mean(),
                    'coverage_bootstrap': ss['covers_bootstrap'].dropna().mean()
                        if ss['covers_bootstrap'].notna().any() else np.nan,
                    'mean_se_plugin': ss['se_plugin'].mean(),
                    'mean_se_bootstrap': ss['se_bootstrap'].dropna().mean()
                        if ss['se_bootstrap'].notna().any() else np.nan,
                    'mean_abs_bias': ss['bias'].abs().mean(),
                    'n_obs': len(ss),
                })

    return pd.DataFrame(agg_rows + sg_rows)


def print_results(df_summary):
    agg = df_summary[df_summary['breakdown'] == 'aggregate']

    print("\n" + "=" * 100)
    print("AGGREGATE COVERAGE: plug-in vs bootstrap")
    print("=" * 100)
    for ne in N_EXPERT_LIST:
        print(f"\nn_expert={ne}:")
        print(f"  {'Method':<12s} {'Cov_PI':>8s} {'Cov_BS':>8s} "
              f"{'SE_PI':>8s} {'SE_BS':>8s} {'|Bias|':>8s}")
        for method in METHODS:
            row = agg[(agg['n_expert'] == ne) & (agg['method'] == method)]
            if row.empty:
                continue
            r = row.iloc[0]
            cov_bs = f"{r['coverage_bootstrap']:.3f}" if not np.isnan(r['coverage_bootstrap']) else "   N/A"
            se_bs = f"{r['mean_se_bootstrap']:.4f}" if not np.isnan(r['mean_se_bootstrap']) else "    N/A"
            print(f"  {method:<12s} {r['coverage_plugin']:8.3f} {cov_bs:>8s} "
                  f"{r['mean_se_plugin']:8.4f} {se_bs:>8s} {r['mean_abs_bias']:8.4f}")

    sg = df_summary[df_summary['breakdown'] != 'aggregate']
    print("\n" + "=" * 100)
    print("PER-SUBGROUP COVERAGE (HB EC-HTE)")
    print("=" * 100)
    for ne in N_EXPERT_LIST:
        print(f"\nn_expert={ne}:")
        print(f"  {'Subgroup':<18s} {'Cov_PI':>8s} {'Cov_BS':>8s} "
              f"{'SE_PI':>8s} {'SE_BS':>8s}")
        for s in range(K):
            row = sg[(sg['n_expert'] == ne) & (sg['method'] == 'hb_echte')
                     & (sg['breakdown'] == f's{s}_pi{PI[s]}')]
            if row.empty:
                continue
            r = row.iloc[0]
            label = f"s{s} (π={PI[s]})"
            cov_bs = f"{r['coverage_bootstrap']:.3f}" if not np.isnan(r['coverage_bootstrap']) else "   N/A"
            se_bs = f"{r['mean_se_bootstrap']:.4f}" if not np.isnan(r['mean_se_bootstrap']) else "    N/A"
            print(f"  {label:<18s} {r['coverage_plugin']:8.3f} {cov_bs:>8s} "
                  f"{r['mean_se_plugin']:8.4f} {se_bs:>8s}")


def write_analysis(df, df_summary, total_time):
    agg = df_summary[df_summary['breakdown'] == 'aggregate']
    sg = df_summary[df_summary['breakdown'] != 'aggregate']

    lines = []
    lines.append("# Bootstrap Coverage with Small n_expert\n")
    lines.append("## Configuration\n")
    lines.append(f"- N={N}, K={K}, π={PI}, τ(0)={TAU_Z[0]}, τ(1)={TAU_Z[1]}")
    lines.append(f"- n_expert ∈ {N_EXPERT_LIST}")
    lines.append(f"- N_MC={N_MC}, B={N_BOOTSTRAP}, n_estimators={N_ESTIMATORS}")
    lines.append(f"- CM estimation: Centered EB (Dirichlet prior centered on global CM)")
    lines.append(f"- Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)\n")

    lines.append("## Aggregate Coverage\n")
    lines.append("| n_expert | Method | Cov (plug-in) | Cov (bootstrap) | "
                 "SE (plug-in) | SE (bootstrap) | |Bias| |")
    lines.append("|----------|--------|:---:|:---:|:---:|:---:|:---:|")
    for ne in N_EXPERT_LIST:
        for method in METHODS:
            row = agg[(agg['n_expert'] == ne) & (agg['method'] == method)]
            if row.empty:
                continue
            r = row.iloc[0]
            cov_bs = f"{r['coverage_bootstrap']:.3f}" if not np.isnan(r['coverage_bootstrap']) else "N/A"
            se_bs = f"{r['mean_se_bootstrap']:.4f}" if not np.isnan(r['mean_se_bootstrap']) else "N/A"
            lines.append(
                f"| {ne} | {method} | {r['coverage_plugin']:.3f} | {cov_bs} | "
                f"{r['mean_se_plugin']:.4f} | {se_bs} | {r['mean_abs_bias']:.4f} |")

    lines.append("\n## Per-Subgroup Coverage (HB EC-HTE)\n")
    lines.append("| n_expert | Subgroup | π | Cov (plug-in) | Cov (bootstrap) |")
    lines.append("|----------|----------|:---:|:---:|:---:|")
    for ne in N_EXPERT_LIST:
        for s in range(K):
            row = sg[(sg['n_expert'] == ne) & (sg['method'] == 'hb_echte')
                     & (sg['breakdown'] == f's{s}_pi{PI[s]}')]
            if row.empty:
                continue
            r = row.iloc[0]
            cov_bs = f"{r['coverage_bootstrap']:.3f}" if not np.isnan(r['coverage_bootstrap']) else "N/A"
            lines.append(
                f"| {ne} | s{s} | {PI[s]} | {r['coverage_plugin']:.3f} | {cov_bs} |")

    lines.append("\n## Valid Regime Analysis\n")
    min_valid_bs = None
    for ne in N_EXPERT_LIST:
        row = agg[(agg['n_expert'] == ne) & (agg['method'] == 'hb_echte')]
        if row.empty:
            continue
        r = row.iloc[0]
        cov_pi = r['coverage_plugin']
        cov_bs = r['coverage_bootstrap']
        if np.isnan(cov_bs):
            status = "insufficient bootstrap data"
        elif cov_bs >= 0.93:
            status = "nominal"
            if min_valid_bs is None:
                min_valid_bs = ne
        elif cov_bs >= 0.90:
            status = "marginal"
            if min_valid_bs is None:
                min_valid_bs = ne
        else:
            status = "below nominal"
        lines.append(f"- n_expert={ne}: plug-in={cov_pi:.3f}, "
                     f"bootstrap={'N/A' if np.isnan(cov_bs) else f'{cov_bs:.3f}'} "
                     f"→ {status}")

    lines.append("")
    if min_valid_bs:
        lines.append(f"**Valid regime lower bound: n_expert ≥ {min_valid_bs}** "
                     f"(bootstrap coverage ≥ 90%)")
    else:
        lines.append("**No n_expert value tested achieves ≥ 90% bootstrap coverage "
                     "for HB EC-HTE.**")

    lines.append("\n## Runtime per Configuration\n")
    lines.append("| n_expert | Time (s) |")
    lines.append("|----------|----------|")

    with open('results/exp_bootstrap_small_n_analysis.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t_start = time.time()
    os.makedirs('results', exist_ok=True)

    if not HAS_ECONML:
        print("ERROR: econml required.")
        sys.exit(1)

    print(f"Bootstrap Small n_expert Experiment")
    print(f"N={N}, K={K}, PI={PI}, TAU_Z={TAU_Z}")
    print(f"n_expert={N_EXPERT_LIST}")
    print(f"N_MC={N_MC}, N_BOOTSTRAP={N_BOOTSTRAP}, n_estimators={N_ESTIMATORS}")
    print()

    all_rows = []
    config_times = {}

    for ne in N_EXPERT_LIST:
        ne_t0 = time.time()
        print(f"n_expert={ne}:")
        for mc in range(N_MC):
            mc_t0 = time.time()
            print(f"  MC {mc+1}/{N_MC} ...", end='', flush=True)
            rows = run_one_mc(ne, 42 + mc)
            all_rows.extend(rows)
            elapsed = time.time() - mc_t0
            print(f" {elapsed:.1f}s ({len(rows)} rows)")
        ne_elapsed = time.time() - ne_t0
        config_times[ne] = ne_elapsed
        print(f"  n_expert={ne} done: {ne_elapsed:.1f}s\n")

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_bootstrap_small_n.csv', index=False)
    print(f"Wrote {len(df)} rows to results/exp_bootstrap_small_n.csv")

    df_summary = build_summary(df)
    df_summary.to_csv('results/exp_bootstrap_small_n_summary.csv', index=False)

    total_time = time.time() - t_start

    print_results(df_summary)
    write_analysis(df, df_summary, total_time)

    with open('results/exp_bootstrap_small_n_analysis.md', 'a') as f:
        for ne in N_EXPERT_LIST:
            f.write(f"| {ne} | {config_times.get(ne, 0):.0f} |\n")
        f.write(f"\n**Total: {total_time:.0f}s ({total_time/60:.1f}min)**\n")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("Wrote results/exp_bootstrap_small_n_summary.csv")
    print("Wrote results/exp_bootstrap_small_n_analysis.md")


if __name__ == '__main__':
    main()
