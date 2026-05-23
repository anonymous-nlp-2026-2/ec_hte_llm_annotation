"""
EC-HTE Synthetic Simulation MVP v3

Evaluates CATE estimation at three levels: marginal, subgroup_A, subgroup_B.
EC-HTE's value: global correction is biased at subgroup level when misclassification
rates are heterogeneous; stratified correction (EC-HTE) removes this bias.

Input: regime parameters (misclassification rates, treatment effects)
Output: results/mvp_results_v3.csv, results/mvp_results_v3_raw.csv
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REGIMES = {
    'extreme':     {'misclass_A': 0.25, 'misclass_B': 0.05, 'tau_z1': 0.5,  'tau_z0': 0.1},
    'moderate':    {'misclass_A': 0.15, 'misclass_B': 0.10, 'tau_z1': 0.3,  'tau_z0': 0.15},
    'homogeneous': {'misclass_A': 0.12, 'misclass_B': 0.12, 'tau_z1': 0.25, 'tau_z0': 0.20},
}
N_EXPERTS = [100, 250, 500]
N_MC = 100
N = 5000


def generate_data(regime_params, n_expert, seed):
    rng = np.random.RandomState(seed)
    p = regime_params

    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(['A', 'B'], N, p=[0.5, 0.5])
    X = rng.randn(N, 5)

    tau = np.where(Z == 1, p['tau_z1'], p['tau_z0'])
    eps = rng.randn(N)
    Y = 1 + tau * T + 0.5 * X[:, 0] + eps

    mc_rates = np.where(S == 'A', p['misclass_A'], p['misclass_B'])
    flip = rng.rand(N) < mc_rates
    Z_hat = np.where(flip, 1 - Z, Z)

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


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


def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        col_sum = C[0, zh] + C[1, zh]
        for z in [0, 1]:
            M[zh, z] = C[z, zh] / col_sum if col_sum > 0 else 0.5
    return M


def invert_mixing(M, tau_obs, se_obs):
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    M_inv = np.linalg.inv(M)
    tau_corrected = M_inv @ tau_obs
    sigma_corrected = M_inv @ np.diag(se_obs ** 2) @ M_inv.T
    se_corrected = np.sqrt(np.diag(sigma_corrected))
    return tau_corrected, se_corrected


def run_one_mc(regime_name, regime_params, n_expert, seed):
    data = generate_data(regime_params, n_expert, seed)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    tau_z0, tau_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    base = {'regime': regime_name, 'n_expert': n_expert, 'mc_run': seed}
    rows = []

    # Estimate confusion matrices
    z_exp, zh_exp, s_exp = Z[expert_mask], Z_hat[expert_mask], S[expert_mask]
    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    M_global = build_mixing_matrix(C_global)

    C_sub, M_sub = {}, {}
    for s_val in ['A', 'B']:
        sm = s_exp == s_val
        if sm.sum() >= 4:
            C_sub[s_val] = estimate_confusion_matrix(z_exp[sm], zh_exp[sm])
        else:
            C_sub[s_val] = C_global
        M_sub[s_val] = build_mixing_matrix(C_sub[s_val])

    def make_row(method, level, th0, th1, se0, se1):
        ci_lo0, ci_hi0 = th0 - 1.96 * se0, th0 + 1.96 * se0
        ci_lo1, ci_hi1 = th1 - 1.96 * se1, th1 + 1.96 * se1
        return {
            **base, 'method': method, 'level': level,
            'tau_hat_z0': th0, 'tau_hat_z1': th1,
            'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1,
            'ci_lower_z0': ci_lo0, 'ci_upper_z0': ci_hi0,
            'ci_lower_z1': ci_lo1, 'ci_upper_z1': ci_hi1,
        }

    # --- Precompute subgroup observed and corrected CATEs ---
    sub_obs = {}
    sub_se_obs = {}
    sub_corrected_ec = {}
    sub_se_ec = {}
    sub_n = {}
    for s_val in ['A', 'B']:
        s_mask = S == s_val
        sub_n[s_val] = s_mask.sum()
        th0_obs, se0_obs = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1_obs, se1_obs = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        sub_obs[s_val] = np.array([th0_obs, th1_obs])
        sub_se_obs[s_val] = np.array([se0_obs, se1_obs])
        tau_ec, se_ec = invert_mixing(M_sub[s_val], sub_obs[s_val], sub_se_obs[s_val])
        sub_corrected_ec[s_val] = tau_ec
        sub_se_ec[s_val] = se_ec

    # --- Marginal level ---

    # Oracle marginal: group by true Z
    th0_or, se0_or = diff_in_means(Y, T, Z == 0)
    th1_or, se1_or = diff_in_means(Y, T, Z == 1)
    rows.append(make_row('oracle', 'marginal', th0_or, th1_or, se0_or, se1_or))

    # Naive marginal: group by Z_hat
    th0_naive, se0_naive = diff_in_means(Y, T, Z_hat == 0)
    th1_naive, se1_naive = diff_in_means(Y, T, Z_hat == 1)
    rows.append(make_row('naive', 'marginal', th0_naive, th1_naive, se0_naive, se1_naive))

    # Global-corrected marginal: M_global^{-1} on marginal observed
    tau_obs_m = np.array([th0_naive, th1_naive])
    se_obs_m = np.array([se0_naive, se1_naive])
    tau_gc_m, se_gc_m = invert_mixing(M_global, tau_obs_m, se_obs_m)
    rows.append(make_row('global_corrected', 'marginal', tau_gc_m[0], tau_gc_m[1], se_gc_m[0], se_gc_m[1]))

    # EC-HTE stratified marginal: weighted average of subgroup-corrected
    total_n = sum(sub_n.values())
    w = {s: sub_n[s] / total_n for s in ['A', 'B']}
    tau_ec_m = w['A'] * sub_corrected_ec['A'] + w['B'] * sub_corrected_ec['B']
    se_ec_m = np.sqrt((w['A'] * sub_se_ec['A']) ** 2 + (w['B'] * sub_se_ec['B']) ** 2)
    rows.append(make_row('ec_hte_stratified', 'marginal', tau_ec_m[0], tau_ec_m[1], se_ec_m[0], se_ec_m[1]))

    # --- Subgroup levels ---
    for s_val in ['A', 'B']:
        s_mask = S == s_val
        level = f'subgroup_{s_val}'

        tau_obs = sub_obs[s_val]
        se_obs = sub_se_obs[s_val]
        if np.any(np.isnan(tau_obs)):
            continue

        # Oracle
        th0_or, se0_or = diff_in_means(Y, T, s_mask & (Z == 0))
        th1_or, se1_or = diff_in_means(Y, T, s_mask & (Z == 1))
        rows.append(make_row('oracle', level, th0_or, th1_or, se0_or, se1_or))

        # Naive
        rows.append(make_row('naive', level, tau_obs[0], tau_obs[1], se_obs[0], se_obs[1]))

        # Global-corrected: global M^{-1} applied to subgroup observed
        tau_gc, se_gc = invert_mixing(M_global, tau_obs, se_obs)
        rows.append(make_row('global_corrected', level, tau_gc[0], tau_gc[1], se_gc[0], se_gc[1]))

        # EC-HTE stratified: subgroup-specific M^{-1}
        rows.append(make_row('ec_hte_stratified', level, sub_corrected_ec[s_val][0], sub_corrected_ec[s_val][1],
                             sub_se_ec[s_val][0], sub_se_ec[s_val][1]))

    return rows


def run_simulation(n_mc, verbose=True):
    results = []
    total = len(REGIMES) * len(N_EXPERTS) * n_mc
    count = 0
    t0 = time.time()

    for regime_name, regime_params in REGIMES.items():
        for n_expert in N_EXPERTS:
            for mc in range(n_mc):
                rows = run_one_mc(regime_name, regime_params, n_expert, seed=mc)
                results.extend(rows)
                count += 1
                if verbose and count % max(1, total // 20) == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"Total time: {elapsed:.1f}s")
    return pd.DataFrame(results), elapsed


def aggregate_results(df):
    rows = []
    for keys, grp in df.groupby(['regime', 'n_expert', 'method', 'level']):
        regime, n_expert, method, level = keys
        row = {'regime': regime, 'n_expert': n_expert, 'method': method, 'level': level}
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'] - grp[f'tau_true_{z}']
            row[f'bias_{z}'] = err.mean()
            row[f'rmse_{z}'] = np.sqrt((err ** 2).mean())
            covered = ((grp[f'ci_lower_{z}'] <= grp[f'tau_true_{z}']) &
                       (grp[f'tau_true_{z}'] <= grp[f'ci_upper_{z}']))
            row[f'coverage_{z}'] = covered.mean()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else N_MC
    print(f"Running EC-HTE simulation v3: N={N}, N_MC={n_mc}")
    print(f"Regimes: {list(REGIMES.keys())}")
    print(f"Expert budgets: {N_EXPERTS}\n")

    df_raw, elapsed = run_simulation(n_mc)
    df_agg = aggregate_results(df_raw)

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 260)
    pd.set_option('display.float_format', '{:.4f}'.format)

    for regime in REGIMES:
        p = REGIMES[regime]
        print(f"\n{'='*160}")
        print(f"Regime: {regime} | tau(Z=0)={p['tau_z0']}, tau(Z=1)={p['tau_z1']} | misclass_A={p['misclass_A']}, misclass_B={p['misclass_B']}")
        print(f"{'='*160}")
        for level in ['marginal', 'subgroup_A', 'subgroup_B']:
            sub = df_agg[(df_agg['regime'] == regime) & (df_agg['level'] == level)]
            sub = sub.sort_values(['n_expert', 'method'])
            cols = ['n_expert', 'method', 'level', 'bias_z0', 'bias_z1', 'rmse_z0', 'rmse_z1', 'coverage_z0', 'coverage_z1']
            print(sub[cols].to_string(index=False))
            print()

    raw_path = 'results/mvp_results_v3_raw.csv'
    agg_path = 'results/mvp_results_v3.csv'
    raw_cols = ['regime', 'n_expert', 'mc_run', 'method', 'level',
                'tau_hat_z0', 'tau_hat_z1', 'tau_true_z0', 'tau_true_z1',
                'ci_lower_z0', 'ci_upper_z0', 'ci_lower_z1', 'ci_upper_z1']
    agg_cols = ['regime', 'n_expert', 'method', 'level',
                'bias_z0', 'rmse_z0', 'coverage_z0', 'bias_z1', 'rmse_z1', 'coverage_z1']
    df_raw[raw_cols].to_csv(raw_path, index=False)
    df_agg[agg_cols].to_csv(agg_path, index=False)
    print(f"Raw results saved to {raw_path}")
    print(f"Aggregated results saved to {agg_path}")
    print(f"Total runtime: {elapsed:.1f}s")
