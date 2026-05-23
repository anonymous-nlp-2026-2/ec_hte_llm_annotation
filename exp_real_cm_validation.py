"""
exp_real_cm_validation.py — End-to-end MC validation with real LLM confusion matrices.

Uses real per-subgroup FPR/FNR from 13 LLM-dataset combinations (10 on TweetEval,
3 on CivilComments) to run Monte Carlo simulations comparing:
  - Naive (no correction)
  - Global corrected (single CM for all subgroups)
  - EC-HTE (HB per-subgroup CM estimation)
  - Oracle EC-HTE (true per-subgroup CMs, no estimation noise)

Data sources:
  - results/exp008_multi_llm_comparison.csv (GPT models on TweetEval)
  - results/exp008_checkpoint_gpt-5.4.json (GPT-5.4 on TweetEval)
  - results/exp023_confusion_matrices.json (GPT models on CivilComments)
  - artifacts/c1_multi_llm/*_cm_profile.md (open-source LLMs on TweetEval)
  - artifacts/c1_multi_llm/llama_analysis.md (Llama-3.1-8B on TweetEval)

Outputs:
  - results/exp_real_cm_validation_raw.csv
  - results/exp_real_cm_validation.csv (summary)
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln

warnings.filterwarnings('ignore')

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

N = 5000
N_EXPERTS = [100, 250, 500]
N_MC = 100
SEED_BASE = 42
P_T1 = 0.5
DELTA_TAU = 0.30


# ============================================================
# Real LLM confusion matrix configs
# ============================================================

def load_all_configs():
    configs = {}

    tw_subgroups = ['False_medium', 'False_short', 'True_medium', 'True_short']
    tw_weights = [0.382, 0.222, 0.255, 0.141]
    tw_pz1 = 0.652

    csv_path = os.path.join(RESULTS_DIR, 'exp008_multi_llm_comparison.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[~df['subgroup'].str.contains('long')]
        for model in df['model'].unique():
            mdf = df[df['model'] == model].sort_values('subgroup')
            if len(mdf) < 4:
                continue
            configs[f'{model}_tweet'] = {
                'llm': model, 'dataset': 'TweetEval', 'p_z1': tw_pz1,
                'subgroups': mdf['subgroup'].tolist(),
                'weights': tw_weights,
                'fpr': mdf['FPR'].tolist(),
                'fnr': mdf['FNR'].tolist(),
            }

    configs['gpt-5.4_tweet'] = {
        'llm': 'gpt-5.4', 'dataset': 'TweetEval', 'p_z1': tw_pz1,
        'subgroups': tw_subgroups, 'weights': tw_weights,
        'fpr': [0.0655, 0.0787, 0.2143, 0.1429],
        'fnr': [0.4211, 0.3000, 0.2222, 0.1538],
    }
    configs['gemma-2-9b_tweet'] = {
        'llm': 'Gemma-2-9B', 'dataset': 'TweetEval', 'p_z1': tw_pz1,
        'subgroups': tw_subgroups, 'weights': tw_weights,
        'fpr': [0.110, 0.138, 0.033, 0.091],
        'fnr': [0.437, 0.300, 0.263, 0.207],
    }
    configs['mistral-7b_tweet'] = {
        'llm': 'Mistral-7B', 'dataset': 'TweetEval', 'p_z1': tw_pz1,
        'subgroups': tw_subgroups, 'weights': tw_weights,
        'fpr': [0.180, 0.241, 0.233, 0.091],
        'fnr': [0.299, 0.188, 0.168, 0.121],
    }
    configs['qwen-2.5-7b_tweet'] = {
        'llm': 'Qwen2.5-7B', 'dataset': 'TweetEval', 'p_z1': tw_pz1,
        'subgroups': tw_subgroups, 'weights': tw_weights,
        'fpr': [0.1800, 0.2414, 0.2333, 0.0000],
        'fnr': [0.2874, 0.1750, 0.2105, 0.1379],
    }
    configs['llama-3.1-8b_tweet'] = {
        'llm': 'Llama-3.1-8B', 'dataset': 'TweetEval', 'p_z1': tw_pz1,
        'subgroups': tw_subgroups, 'weights': tw_weights,
        'fpr': [0.235, 0.310, 0.219, 0.091],
        'fnr': [0.319, 0.212, 0.258, 0.190],
    }

    json_path = os.path.join(RESULTS_DIR, 'exp023_confusion_matrices.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            cm = json.load(f)
        for model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
            if model not in cm:
                continue
            md = cm[model]
            subs = sorted(k for k in md if not k.startswith('_'))
            total_n = sum(md[s]['n'] for s in subs)
            total_pos = sum(md[s]['n_pos'] for s in subs)
            configs[f'{model}_civil'] = {
                'llm': model, 'dataset': 'CivilComments', 'p_z1': total_pos / total_n,
                'subgroups': subs,
                'weights': [md[s]['n'] / total_n for s in subs],
                'fpr': [md[s]['fpr'] for s in subs],
                'fnr': [md[s]['fnr'] for s in subs],
            }

    return configs


# ============================================================
# Core estimators
# ============================================================

def get_cm_matrix(fpr, fnr):
    return np.array([[1 - fpr, fpr],
                     [fnr, 1 - fnr]])


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


def build_mixing_from_rates(fpr, fnr, p_z1):
    """Build mixing matrix directly from FPR, FNR, P(Z=1)."""
    p0, p1 = 1.0 - p_z1, p_z1
    p_zh0 = (1 - fpr) * p0 + fnr * p1
    p_zh1 = fpr * p0 + (1 - fnr) * p1
    return np.array([
        [(1 - fpr) * p0 / p_zh0, fnr * p1 / p_zh0],
        [fpr * p0 / p_zh1, (1 - fnr) * p1 / p_zh1],
    ])


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


# ============================================================
# HB Gibbs Sampler
# ============================================================

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


# ============================================================
# DGP and MC runner
# ============================================================

def generate_data(cfg, n_expert, seed):
    rng = np.random.RandomState(seed)
    K = len(cfg['subgroups'])
    w = np.array(cfg['weights'], dtype=float)
    w /= w.sum()
    p_z1 = cfg['p_z1']
    fpr_list = cfg['fpr']
    fnr_list = cfg['fnr']

    S_idx = rng.choice(K, size=N, p=w)
    S = np.array([cfg['subgroups'][i] for i in S_idx])
    Z = rng.binomial(1, p_z1, N)
    T = rng.binomial(1, P_T1, N)

    Z_hat = np.empty(N, dtype=int)
    for s in range(K):
        s_mask = S_idx == s
        cm = get_cm_matrix(fpr_list[s], fnr_list[s])
        z_s = Z[s_mask]
        p_obs1 = np.where(z_s == 0, cm[0, 1], cm[1, 1])
        Z_hat[s_mask] = rng.binomial(1, p_obs1)

    # Constant Δτ = DELTA_TAU across all subgroups
    tau_vals = np.where(Z == 0, 0.0, DELTA_TAU)
    Y = 1.0 + tau_vals * T + rng.randn(N)

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'S_idx': S_idx,
        'Y': Y, 'tau': tau_vals, 'expert_mask': expert_mask,
    }


def run_one_mc(cfg, data, n_expert, seed):
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    K = len(cfg['subgroups'])
    subgroup_labels = cfg['subgroups']
    p_z1 = cfg['p_z1']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    # --- Global CM from expert subset ---
    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    # --- HB per-subgroup CMs ---
    counts = get_counts(z_exp, zh_exp, s_exp, subgroup_labels)
    C_hb = estimate_cm_hb_gibbs(counts, subgroup_labels, seed=seed)

    M_hb_sub = {}
    for i, s_val in enumerate(subgroup_labels):
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        if sm_exp.sum() >= 4:
            p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()])
        else:
            p_z_s = p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_hb_sub[s_val] = build_mixing_matrix(C_hb[s_val], p_z_s, p_zh_s)

    # --- Oracle per-subgroup CMs (true FPR/FNR) ---
    M_oracle_sub = {}
    for i, s_val in enumerate(subgroup_labels):
        M_oracle_sub[s_val] = build_mixing_from_rates(cfg['fpr'][i], cfg['fnr'][i], p_z1)

    rows = []
    for s_i, s_val in enumerate(subgroup_labels):
        s_mask = S == s_val
        for z_level in [0, 1]:
            oracle_cate = DELTA_TAU if z_level == 1 else 0.0

            # Observed tau vector for this subgroup
            tau_obs = np.array([
                diff_in_means(Y, T, s_mask & (Z_hat == 0))[0],
                diff_in_means(Y, T, s_mask & (Z_hat == 1))[0],
            ])
            se_obs = np.array([
                diff_in_means(Y, T, s_mask & (Z_hat == 0))[1],
                diff_in_means(Y, T, s_mask & (Z_hat == 1))[1],
            ])

            # --- Naive ---
            th_nv, se_nv = diff_in_means(Y, T, s_mask & (Z_hat == z_level))
            ci_lo_nv = th_nv - 1.96 * se_nv if not np.isnan(se_nv) else np.nan
            ci_hi_nv = th_nv + 1.96 * se_nv if not np.isnan(se_nv) else np.nan
            cov_nv = 1 if (not np.isnan(ci_lo_nv) and ci_lo_nv <= oracle_cate <= ci_hi_nv) else 0

            def make_row(method, th, se_z):
                ci_lo = th - 1.96 * se_z if not np.isnan(se_z) else np.nan
                ci_hi = th + 1.96 * se_z if not np.isnan(se_z) else np.nan
                cov = 1 if (not np.isnan(ci_lo) and ci_lo <= oracle_cate <= ci_hi) else 0
                return {
                    'seed': seed, 'n_expert': n_expert,
                    'subgroup': s_val, 'z_level': z_level, 'method': method,
                    'cate': th, 'oracle_cate': oracle_cate,
                    'bias': th - oracle_cate if not np.isnan(th) else np.nan,
                    'ci_lo': ci_lo, 'ci_hi': ci_hi, 'coverage': cov,
                }

            rows.append(make_row('naive', th_nv, se_nv))

            if np.any(np.isnan(tau_obs)):
                for m in ['global_corrected', 'hb_ec_hte', 'oracle_ec_hte']:
                    rows.append(make_row(m, np.nan, np.nan))
                continue

            # --- Global corrected ---
            tau_gc, se_gc = invert_mixing_safe(M_global, tau_obs, se_obs)
            rows.append(make_row('global_corrected', tau_gc[z_level], se_gc[z_level]))

            # --- HB EC-HTE ---
            tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], tau_obs, se_obs)
            rows.append(make_row('hb_ec_hte', tau_hb[z_level], se_hb[z_level]))

            # --- Oracle EC-HTE (true CMs) ---
            tau_or, se_or = invert_mixing_safe(M_oracle_sub[s_val], tau_obs, se_obs)
            rows.append(make_row('oracle_ec_hte', tau_or[z_level], se_or[z_level]))

    return rows


# ============================================================
# Aggregation
# ============================================================

def aggregate(df_raw):
    group_cols = ['llm', 'dataset', 'n_expert', 'subgroup', 'z_level', 'method']
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
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-mc', type=int, default=N_MC)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    n_mc = 5 if args.dry else args.n_mc

    configs = load_all_configs()
    print(f"Loaded {len(configs)} LLM-dataset configs, Δτ={DELTA_TAU}")
    for key, cfg in configs.items():
        delta_fpr = max(cfg['fpr']) - min(cfg['fpr'])
        delta_fnr = max(cfg['fnr']) - min(cfg['fnr'])
        print(f"  {key}: K={len(cfg['subgroups'])}, "
              f"P(Z=1)={cfg['p_z1']:.3f}, "
              f"ΔFPR={delta_fpr:.3f}, ΔFNR={delta_fnr:.3f}")
    print(f"\nN={N}, n_experts={N_EXPERTS}, n_mc={n_mc}\n")

    all_raw = []
    t0 = time.time()
    total = len(configs) * len(N_EXPERTS)
    count = 0

    for cfg_key, cfg in configs.items():
        llm = cfg['llm']
        dataset = cfg['dataset']
        for n_expert in N_EXPERTS:
            count += 1
            print(f"[{count}/{total}] {llm} ({dataset}), n_exp={n_expert}...",
                  end='', flush=True)
            t1 = time.time()

            for mc in range(n_mc):
                seed = SEED_BASE + mc
                data = generate_data(cfg, n_expert, seed)
                mc_rows = run_one_mc(cfg, data, n_expert, seed)
                for r in mc_rows:
                    r['llm'] = llm
                    r['dataset'] = dataset
                all_raw.extend(mc_rows)

            print(f" {time.time() - t1:.1f}s")

    total_time = time.time() - t0
    print(f"\nTotal: {total_time:.1f}s")

    df_raw = pd.DataFrame(all_raw)
    df_raw.to_csv(f'{RESULTS_DIR}/exp_real_cm_validation_raw.csv', index=False)

    df_detail = aggregate(df_raw)

    # Per-LLM-dataset-nexp summary
    summary_rows = []
    for (llm, dataset, n_expert), grp in df_detail.groupby(['llm', 'dataset', 'n_expert']):
        for method in ['naive', 'global_corrected', 'hb_ec_hte', 'oracle_ec_hte']:
            mg = grp[grp['method'] == method]
            if len(mg) == 0:
                continue
            summary_rows.append({
                'llm': llm, 'dataset': dataset, 'n_expert': int(n_expert),
                'method': method,
                'mean_abs_bias': mg['mean_abs_bias'].mean(),
                'rmse': mg['rmse'].mean(),
                'coverage': mg['coverage'].mean(),
            })

    df_final = pd.DataFrame(summary_rows)
    df_final.to_csv(f'{RESULTS_DIR}/exp_real_cm_validation.csv', index=False)

    # Print results
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 200)

    for n_expert in N_EXPERTS:
        print(f"\n{'='*80}")
        print(f"n_expert = {n_expert}")
        print(f"{'='*80}")

        sub = df_final[df_final['n_expert'] == n_expert]

        pivot_bias = sub.pivot_table(index=['llm', 'dataset'], columns='method',
                                     values='mean_abs_bias')
        pivot_rmse = sub.pivot_table(index=['llm', 'dataset'], columns='method',
                                     values='rmse')

        print("\n--- Mean |Bias| ---")
        if 'oracle_ec_hte' in pivot_bias.columns:
            print(pivot_bias[['naive', 'global_corrected', 'hb_ec_hte', 'oracle_ec_hte']].to_string())
        else:
            print(pivot_bias.to_string())
        print("\n--- RMSE ---")
        if 'oracle_ec_hte' in pivot_rmse.columns:
            print(pivot_rmse[['naive', 'global_corrected', 'hb_ec_hte', 'oracle_ec_hte']].to_string())
        else:
            print(pivot_rmse.to_string())

        # Win counts
        n_combos = 0
        n_hb_wins_bias = 0
        n_hb_wins_rmse = 0
        n_oracle_wins_bias = 0
        for (llm, dataset), grp in sub.groupby(['llm', 'dataset']):
            gc = grp[grp['method'] == 'global_corrected']
            hb = grp[grp['method'] == 'hb_ec_hte']
            orc = grp[grp['method'] == 'oracle_ec_hte']
            if len(gc) == 0 or len(hb) == 0:
                continue
            n_combos += 1
            if hb.iloc[0]['mean_abs_bias'] < gc.iloc[0]['mean_abs_bias']:
                n_hb_wins_bias += 1
            if hb.iloc[0]['rmse'] < gc.iloc[0]['rmse']:
                n_hb_wins_rmse += 1
            if len(orc) > 0 and orc.iloc[0]['mean_abs_bias'] < gc.iloc[0]['mean_abs_bias']:
                n_oracle_wins_bias += 1

        print(f"\nEC-HTE vs Global:  |bias| wins {n_hb_wins_bias}/{n_combos}, "
              f"RMSE wins {n_hb_wins_rmse}/{n_combos}")
        print(f"Oracle vs Global:  |bias| wins {n_oracle_wins_bias}/{n_combos}")

        # Overall averages
        for method in ['naive', 'global_corrected', 'hb_ec_hte', 'oracle_ec_hte']:
            ms = sub[sub['method'] == method]
            if len(ms) == 0:
                continue
            print(f"  {method:20s}: |bias|={ms['mean_abs_bias'].mean():.4f}, "
                  f"RMSE={ms['rmse'].mean():.4f}, "
                  f"cov={ms['coverage'].mean():.3f}")

    print(f"\nSaved {RESULTS_DIR}/exp_real_cm_validation_raw.csv ({len(df_raw)} rows)")
    print(f"Saved {RESULTS_DIR}/exp_real_cm_validation.csv ({len(df_final)} rows)")
