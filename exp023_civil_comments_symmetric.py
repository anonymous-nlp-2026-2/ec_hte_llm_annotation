"""
exp-023 Phase 2b: Semi-synthetic DGP on Civil Comments with SYMMETRIC misclassification.

Same setup as exp-003 (TweetEval) but with Civil Comments subgroup structure.
Uses observed overall misclassification rates per subgroup from real LLM annotations,
but applies them as symmetric flip probabilities (FPR = FNR = pi_s).

This isolates the misclassification heterogeneity phenomenon on a second dataset,
matching the paper's main experimental design (Table 1).
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import gammaln

warnings.filterwarnings('ignore')

PROJ = Path(__file__).parent
RESULTS = PROJ / "results"

N = 5000
N_MC = 50
N_EXPERT_LIST = [100, 250, 500]
SEED_BASE = 42
SUBGROUP_LABELS = ['S0', 'S1', 'S2', 'S3']

TAU_Z0 = 0.10
DELTA_S = {'S0': 0.4, 'S1': 0.3, 'S2': 0.2, 'S3': 0.1}
P_T = 0.5

df_pool = pd.read_parquet(RESULTS / "exp023_civil_comments_pool.parquet")
sg_counts = df_pool['subgroup'].value_counts()
SUBGROUP_WEIGHTS = np.array([sg_counts.get(s, 0) for s in SUBGROUP_LABELS], dtype=float)
SUBGROUP_WEIGHTS /= SUBGROUP_WEIGHTS.sum()
P_Z1 = df_pool['y_true'].mean()

cms = json.loads((RESULTS / "exp023_confusion_matrices.json").read_text())

print("=== Observed per-subgroup misclassification rates ===")
MISCLASS_CONFIGS = {}
for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
    mc = cms[model]
    config = {}
    for s in SUBGROUP_LABELS:
        if s in mc:
            n = mc[s]['n']
            fp = mc[s]['fp']
            fn = mc[s]['fn']
            pi_s = (fp + fn) / n if n > 0 else 0.10
            config[s] = round(pi_s, 3)
    delta_pi = max(config.values()) - min(config.values())
    MISCLASS_CONFIGS[model] = config
    print(f"  {model}: {config}, Δπ={delta_pi:.3f}")

MISCLASS_CONFIGS['extreme'] = {'S0': 0.05, 'S1': 0.10, 'S2': 0.15, 'S3': 0.25}

print(f"\nPool: P(Z=1)={P_Z1:.3f}, weights={dict(zip(SUBGROUP_LABELS, SUBGROUP_WEIGHTS))}")


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


def _log_target_alpha(alpha, theta_z, hp_a0=3.0, hp_b0=1.0):
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, labels, seed=42, n_iter=1000, n_warmup=500, n_chains=4):
    k = len(labels)
    chain_inits = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                   np.array([0.5, 0.5]), np.array([10.0, 10.0])][:n_chains]
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
    for i, s in enumerate(labels):
        C[s] = theta_arr[:, :, i, :, :].mean(axis=(0, 1))
    return C


def generate_data(misclass_config, n_expert, seed):
    rng = np.random.RandomState(seed)

    S_idx = rng.choice(4, size=N, p=SUBGROUP_WEIGHTS)
    S = np.array([SUBGROUP_LABELS[i] for i in S_idx])

    Z = rng.binomial(1, P_Z1, N)
    T = rng.binomial(1, P_T, N)

    mc_rates = np.array([misclass_config[S[i]] for i in range(N)])
    flip = rng.rand(N) < mc_rates
    Z_hat = np.where(flip, 1 - Z, Z)

    delta_s = np.array([DELTA_S[S[i]] for i in range(N)])
    tau_vals = np.where(Z == 1, TAU_Z0 + delta_s, TAU_Z0)
    eps = rng.randn(N)
    Y = 1.0 + tau_vals * T + eps

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S,
        'Y': Y, 'tau': tau_vals, 'expert_mask': expert_mask,
    }


def run_one_mc(data, seed):
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    C_mle = {}
    M_mle_sub = {}
    for s_val in SUBGROUP_LABELS:
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        if sm_exp.sum() >= 4:
            C_mle[s_val] = estimate_confusion_matrix(z_exp[sm_exp], zh_exp[sm_exp])
            p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()])
        else:
            C_mle[s_val] = C_global
            p_z_s = p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_mle_sub[s_val] = build_mixing_matrix(C_mle[s_val], p_z_s, p_zh_s)

    counts = get_counts(z_exp, zh_exp, s_exp, SUBGROUP_LABELS)
    C_hb = estimate_cm_hb_gibbs(counts, SUBGROUP_LABELS, seed=seed)

    M_hb_sub = {}
    for s_val in SUBGROUP_LABELS:
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        if sm_exp.sum() >= 4:
            p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()])
        else:
            p_z_s = p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_hb_sub[s_val] = build_mixing_matrix(C_hb[s_val], p_z_s, p_zh_s)

    rows = []
    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val
        true_tau_z0 = TAU_Z0
        true_tau_z1 = TAU_Z0 + DELTA_S[s_val]

        tau_obs = np.array([
            diff_in_means(Y, T, s_mask & (Z_hat == 0))[0],
            diff_in_means(Y, T, s_mask & (Z_hat == 1))[0],
        ])
        se_obs = np.array([
            diff_in_means(Y, T, s_mask & (Z_hat == 0))[1],
            diff_in_means(Y, T, s_mask & (Z_hat == 1))[1],
        ])

        for z_level in [0, 1]:
            true_cate = true_tau_z1 if z_level == 1 else true_tau_z0

            def make_row(method, th, se):
                ci_lo = th - 1.96 * se if not np.isnan(se) else np.nan
                ci_hi = th + 1.96 * se if not np.isnan(se) else np.nan
                cov = 1 if (not np.isnan(ci_lo) and ci_lo <= true_cate <= ci_hi) else 0
                return {
                    'seed': seed, 'method': method, 'subgroup': s_val,
                    'z_level': z_level,
                    'bias': th - true_cate if not np.isnan(th) else np.nan,
                    'cate_est': th, 'cate_true': true_cate, 'coverage': cov,
                }

            th_or, se_or = diff_in_means(Y, T, s_mask & (Z == z_level))
            rows.append(make_row('oracle', th_or, se_or))

            th_nv, se_nv = diff_in_means(Y, T, s_mask & (Z_hat == z_level))
            rows.append(make_row('naive', th_nv, se_nv))

            if not np.any(np.isnan(tau_obs)):
                tau_gc, se_gc = invert_mixing_safe(M_global, tau_obs, se_obs)
                rows.append(make_row('global_corrected', tau_gc[z_level], se_gc[z_level]))

                tau_ml, se_ml = invert_mixing_safe(M_mle_sub[s_val], tau_obs, se_obs)
                rows.append(make_row('stratified_mle', tau_ml[z_level], se_ml[z_level]))

                tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], tau_obs, se_obs)
                rows.append(make_row('hb_ec_hte', tau_hb[z_level], se_hb[z_level]))
            else:
                for m in ['global_corrected', 'stratified_mle', 'hb_ec_hte']:
                    rows.append(make_row(m, np.nan, np.nan))

    return rows


def run_experiment(config_name, misclass_config):
    all_rows = []
    for n_expert in N_EXPERT_LIST:
        t0 = time.time()
        for mc in range(N_MC):
            seed = SEED_BASE + mc
            data = generate_data(misclass_config, n_expert, seed)
            mc_rows = run_one_mc(data, seed)
            for r in mc_rows:
                r['config'] = config_name
                r['n_expert'] = n_expert
            all_rows.extend(mc_rows)
        elapsed = time.time() - t0
        print(f"  {config_name}/n_expert={n_expert}: {N_MC} MC done in {elapsed:.1f}s")
    return all_rows


def aggregate(df_raw):
    group_cols = ['config', 'n_expert', 'subgroup', 'z_level', 'method']
    rows = []
    for keys, grp in df_raw.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        valid = grp['bias'].dropna()
        if len(valid) == 0:
            continue
        row['mean_bias'] = valid.mean()
        row['mean_abs_bias'] = valid.abs().mean()
        row['rmse'] = np.sqrt((valid ** 2).mean())
        row['coverage'] = grp['coverage'].mean()
        row['n_mc'] = len(valid)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []
    for config_name, misclass_config in MISCLASS_CONFIGS.items():
        print(f"\n=== Config: {config_name} ===")
        rows = run_experiment(config_name, misclass_config)
        all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(RESULTS / "exp023_symmetric_raw.csv", index=False)

    df_summary = aggregate(df_raw)
    df_summary.to_csv(RESULTS / "exp023_symmetric_summary.csv", index=False)

    print("\n\n===== RESULTS =====")
    for config in df_summary['config'].unique():
        csub = df_summary[df_summary.config == config]
        pi = MISCLASS_CONFIGS[config]
        delta = max(pi.values()) - min(pi.values())
        print(f"\n--- {config}: {pi}, Δπ={delta:.3f} ---")
        for ne in N_EXPERT_LIST:
            esub = csub[csub.n_expert == ne]
            print(f"\n  n_expert={ne}:")
            for m in ['oracle', 'naive', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
                ms = esub[esub.method == m]
                if len(ms) > 0:
                    print(f"    {m:20s}: |Bias|={ms.mean_abs_bias.mean():.4f}, "
                          f"RMSE={ms.rmse.mean():.4f}, Cov={ms.coverage.mean():.3f}")

    print("\nDone.")
