"""
exp024_realcm_bias_identity.py — Real-LLM Bias Identity Verification (exp-024)

Verifies Theorem 1 bias decomposition: bias(z,s) = Δτ · [B_s]_{z,1}
using real per-subgroup FPR/FNR from multiple LLMs across two datasets.

Inputs:
  - results/exp023_confusion_matrices.json (GPT on Civil Comments, K=4)
  - results/exp008_multi_llm_comparison.csv (GPT on TweetEval, K=4 excl. long)
  - artifacts/c1_multi_llm/{gemma,mistral,qwen}_cm_profile.md (open-source on TweetEval)

Outputs:
  - results/exp_realcm_bias_identity.csv (per-cell predicted vs observed bias)
  - results/exp_realcm_summary.csv (per-LLM R², diagnostic metrics)
"""

import numpy as np
import pandas as pd
import json
import os
import time
import sys

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

N = 50000
N_MC = 100
TAU_Z0 = 0.0
DELTA_TAUS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================
# Data loading
# ============================================================

def load_llm_configs():
    """Load per-subgroup FPR/FNR from all available data files."""
    configs = {}

    # --- Civil Comments (from exp023_confusion_matrices.json) ---
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
                'llm': model, 'dataset': 'CivilComments',
                'p_z1': total_pos / total_n,
                'subgroups': subs,
                'weights': [md[s]['n'] / total_n for s in subs],
                'fpr': [md[s]['fpr'] for s in subs],
                'fnr': [md[s]['fnr'] for s in subs],
            }

    # --- TweetEval GPT models (from exp008_multi_llm_comparison.csv) ---
    csv_path = os.path.join(RESULTS_DIR, 'exp008_multi_llm_comparison.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[~df['subgroup'].str.contains('long')]
        p_z1_tweet = 0.652
        for model in df['model'].unique():
            mdf = df[df['model'] == model].sort_values('subgroup')
            total_n = mdf['n'].sum()
            if total_n < 50:
                continue
            configs[f'{model}_tweet'] = {
                'llm': model, 'dataset': 'TweetEval',
                'p_z1': p_z1_tweet,
                'subgroups': mdf['subgroup'].tolist(),
                'weights': (mdf['n'] / total_n).tolist(),
                'fpr': mdf['FPR'].tolist(),
                'fnr': mdf['FNR'].tolist(),
            }

    # --- Open-source LLMs on TweetEval (from cm_profile.md files) ---
    configs['gemma-2-9b_tweet'] = {
        'llm': 'Gemma-2-9B', 'dataset': 'TweetEval', 'p_z1': 0.652,
        'subgroups': ['False_medium', 'False_short',
                      'True_medium', 'True_short'],
        'weights': [0.382, 0.222, 0.255, 0.141],
        'fpr': [0.110, 0.138, 0.033, 0.091],
        'fnr': [0.437, 0.300, 0.263, 0.207],
    }

    configs['mistral-7b_tweet'] = {
        'llm': 'Mistral-7B', 'dataset': 'TweetEval', 'p_z1': 0.652,
        'subgroups': ['False_medium', 'False_short',
                      'True_medium', 'True_short'],
        'weights': [0.382, 0.222, 0.255, 0.141],
        'fpr': [0.180, 0.241, 0.233, 0.091],
        'fnr': [0.299, 0.188, 0.168, 0.121],
    }

    configs['qwen-2.5-7b_tweet'] = {
        'llm': 'Qwen2.5-7B', 'dataset': 'TweetEval', 'p_z1': 0.652,
        'subgroups': ['False_medium', 'False_short',
                      'True_medium', 'True_short'],
        'weights': [0.3816, 0.2224, 0.2551, 0.1408],
        'fpr': [0.1800, 0.2414, 0.2333, 0.0000],
        'fnr': [0.2874, 0.1750, 0.2105, 0.1379],
    }

    return configs


# ============================================================
# Core: mixing matrix and analytic bias
# ============================================================

def mixing_matrix(fpr, fnr, p_z1=0.5):
    """M[zhat, z] = P(Z=z | Zhat=zhat, S=s) via Bayes' rule with prior p_z1."""
    p0, p1 = 1.0 - p_z1, p_z1
    p_zh0 = (1 - fpr) * p0 + fnr * p1
    p_zh1 = fpr * p0 + (1 - fnr) * p1
    return np.array([
        [(1 - fpr) * p0 / p_zh0, fnr * p1 / p_zh0],
        [fpr * p0 / p_zh1, (1 - fnr) * p1 / p_zh1],
    ])


def analytic_bias(fpr_list, fnr_list, weights, tau, p_z1=0.5):
    """B_s = M_global^{-1} M_s - I; bias_s = B_s @ tau = Δτ · B_s[:,1]."""
    K = len(fpr_list)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    Ms = [mixing_matrix(fpr_list[s], fnr_list[s], p_z1) for s in range(K)]
    M_global = sum(w[s] * Ms[s] for s in range(K))
    M_inv = np.linalg.inv(M_global)
    biases, Bs = {}, {}
    for s in range(K):
        B = M_inv @ Ms[s] - np.eye(2)
        Bs[s] = B
        biases[s] = B @ tau
    return biases, Bs, M_global, M_inv, Ms


# ============================================================
# MC simulation (vectorized)
# ============================================================

def mc_simulate(fpr_list, fnr_list, weights, tau_z0, tau_z1,
                p_z1, n, n_mc, seed=0):
    """Vectorized MC with asymmetric misclassification and general P(Z=1)."""
    K = len(fpr_list)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    tau = np.array([tau_z0, tau_z1])

    Ms = [mixing_matrix(fpr_list[s], fnr_list[s], p_z1) for s in range(K)]
    M_glob = sum(w[s] * Ms[s] for s in range(K))
    M_glob_inv = np.linalg.inv(M_glob)

    rng = np.random.RandomState(seed)
    Z = rng.binomial(1, p_z1, (n_mc, n))
    T = rng.binomial(1, 0.5, (n_mc, n))

    cumw = np.cumsum(w)
    S = np.searchsorted(cumw, rng.rand(n_mc, n)).clip(0, K - 1)

    tau_i = np.where(Z == 1, tau_z1, tau_z0)
    Y = tau_i * T + rng.randn(n_mc, n)

    flip_p = np.zeros((n_mc, n))
    for s in range(K):
        m = (S == s)
        flip_p += m * np.where(Z == 0, fpr_list[s], fnr_list[s])
    Z_hat = np.where(rng.rand(n_mc, n) < flip_p, 1 - Z, Z)

    ng = K * 4
    grp = (S * 4 + Z_hat * 2 + T).ravel()
    mi = np.repeat(np.arange(n_mc), n)
    comb = mi * ng + grp

    sums = np.bincount(comb, weights=Y.ravel(),
                       minlength=n_mc * ng).reshape(n_mc, ng)
    cnts = np.bincount(comb,
                       minlength=n_mc * ng).reshape(n_mc, ng)
    means = sums / np.maximum(cnts, 1)

    result = {}
    for s in range(K):
        b = s * 4
        tau_obs = np.column_stack([
            means[:, b + 1] - means[:, b + 0],   # zh=0: T=1 - T=0
            means[:, b + 3] - means[:, b + 2],   # zh=1: T=1 - T=0
        ])
        tau_corrected = (M_glob_inv @ tau_obs.T).T
        result[s] = {
            'bias_naive': tau_obs - tau,
            'bias_global': tau_corrected - tau,
        }
    return result


# ============================================================
# Metrics
# ============================================================

def compute_r2(predicted, observed):
    valid = ~(np.isnan(predicted) | np.isnan(observed))
    if valid.sum() < 2:
        return np.nan
    p, o = predicted[valid], observed[valid]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan


def diagnostic_auc(Bs, K):
    """Proposition 3 diagnostic: Lorenz-curve AUC of |B_s[z,1]| distribution."""
    abs_b = np.array([abs(Bs[s][z, 1]) for s in range(K) for z in range(2)])
    if abs_b.sum() < 1e-15:
        return 0.5
    sorted_b = np.sort(abs_b)
    cum = np.cumsum(sorted_b) / sorted_b.sum()
    x = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
    return float(np.trapezoid(cum, x))


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    dry_run = '--dry' in sys.argv
    n_mc = 10 if dry_run else N_MC
    configs = load_llm_configs()

    print("=" * 70)
    print("exp024: Real-LLM Bias Identity Verification")
    print(f"N={N}, N_MC={n_mc}, tau(Z=0)={TAU_Z0}")
    print(f"LLMs loaded: {len(configs)}")
    print("=" * 70)

    detail_rows = []
    summary_rows = []
    t0 = time.time()
    seed_counter = 0

    for cfg_key, cfg in configs.items():
        llm = cfg['llm']
        dataset = cfg['dataset']
        K = len(cfg['subgroups'])
        fpr = cfg['fpr']
        fnr = cfg['fnr']
        w = cfg['weights']
        p_z1 = cfg['p_z1']

        print(f"\n  {llm} ({dataset}, K={K}, P(Z=1)={p_z1:.3f})")

        all_pred, all_obs = [], []

        for dt in DELTA_TAUS:
            tau_z1 = TAU_Z0 + dt
            tau = np.array([TAU_Z0, tau_z1])
            seed_counter += 1
            seed = seed_counter * 7919

            biases, Bs, M_glob, M_inv, Ms = analytic_bias(
                fpr, fnr, w, tau, p_z1)

            mc = mc_simulate(fpr, fnr, w, TAU_Z0, tau_z1, p_z1,
                             N, n_mc, seed)

            for s in range(K):
                for z in range(2):
                    pred = biases[s][z]
                    mc_vals = mc[s]['bias_global'][:, z]
                    mc_mean = float(np.mean(mc_vals))
                    mc_se = float(np.std(mc_vals, ddof=1) / np.sqrt(n_mc))

                    detail_rows.append({
                        'llm': llm, 'dataset': dataset,
                        'subgroup': cfg['subgroups'][s],
                        'z': z, 'delta_tau': dt,
                        'predicted_bias': float(pred),
                        'observed_bias': mc_mean,
                        'se': mc_se,
                    })
                    all_pred.append(pred)
                    all_obs.append(mc_mean)

        pred_arr = np.array(all_pred)
        obs_arr = np.array(all_obs)
        r2 = compute_r2(pred_arr, obs_arr)

        # Diagnostic metrics (Proposition 3, asymmetric version)
        _, Bs_diag, _, _, _ = analytic_bias(
            fpr, fnr, w, np.array([0.0, 1.0]), p_z1)
        d_auc = diagnostic_auc(Bs_diag, K)
        max_b = max(abs(Bs_diag[s][z, 1]) for s in range(K) for z in range(2))

        summary_rows.append({
            'llm': llm, 'dataset': dataset, 'K': K,
            'R2': r2,
            'diagnostic_AUC': d_auc,
            'max_abs_B': max_b,
            'n_subgroups': K,
        })

        elapsed = time.time() - t0
        print(f"    R²={r2:.6f}, diag_AUC={d_auc:.4f}, "
              f"max|B|={max_b:.4f} ({elapsed:.1f}s)")

    # Save results
    df_detail = pd.DataFrame(detail_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_detail.to_csv(f'{RESULTS_DIR}/exp_realcm_bias_identity.csv',
                     index=False)
    df_summary.to_csv(f'{RESULTS_DIR}/exp_realcm_summary.csv',
                      index=False)

    # Terminal summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(df_summary.to_string(index=False))

    # Global R² (all LLMs pooled)
    pred_all = df_detail['predicted_bias'].values
    obs_all = df_detail['observed_bias'].values
    r2_global = compute_r2(pred_all, obs_all)
    print(f"\nGlobal R² (all LLMs pooled): {r2_global:.6f}")

    print(f"\nSaved {RESULTS_DIR}/exp_realcm_bias_identity.csv "
          f"({len(df_detail)} rows)")
    print(f"Saved {RESULTS_DIR}/exp_realcm_summary.csv "
          f"({len(df_summary)} rows)")
    print("Done.")
