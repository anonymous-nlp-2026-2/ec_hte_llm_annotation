"""
exp-022: PPI/DSL vs EC-HTE Baseline Comparison

Demonstrates that PPI and DSL correct ATE but not subgroup CATE under
heterogeneous misclassification — establishing EC-HTE's unique contribution.

Methods:
  oracle         — true Z (infeasible upper bound)
  naive          — Z_hat, no correction
  DSL-global     — global confusion matrix correction (≡ Global correction)
  PPI-global     — global PPI rectifier applied to subgroup CATEs
  PPI-subgroup   — per-subgroup PPI rectifier (independent mean-shift)
  DSL-subgroup   — per-subgroup confusion matrix correction (≡ Stratified MLE)
  EC-HTE         — HB Gibbs partial pooling across subgroups

Mathematical equivalences:
  DSL-global  ≡ Global correction  (global C → M → M^{-1} on each subgroup)
  DSL-subgroup ≡ Stratified MLE    (per-subgroup C_s → M_s → M_s^{-1}, no pooling)
  PPI-subgroup ≠ Stratified MLE    (PPI is additive mean-shift, not matrix inversion)

DGP: TweetEval semi-synthetic (same structure as exp-003)
"""

import numpy as np
import pandas as pd
import sys
import time
import warnings
from scipy.special import gammaln
from datasets import load_dataset

warnings.filterwarnings('ignore')

# === Config ===

N = 5000
N_MC = 50
N_EXPERTS = [100, 250, 500]
SEED_BASE = 42

CONFIGS = {
    2: {
        'misclass': {'S0': 0.08, 'S1': 0.20},
        'delta': {'S0': 0.35, 'S1': 0.15},
        'labels': ['S0', 'S1'],
    },
    4: {
        'misclass': {'S0': 0.05, 'S1': 0.10, 'S2': 0.15, 'S3': 0.25},
        'delta': {'S0': 0.4, 'S1': 0.3, 'S2': 0.2, 'S3': 0.1},
        'labels': ['S0', 'S1', 'S2', 'S3'],
    },
}

TAU_Z0 = 0.1
ALPHA = 1.0
BETA_LENGTH = 0.3
BETA_MENTION = 0.2

METHOD_ORDER = ['oracle', 'naive', 'dsl_global', 'ppi_global',
                'ppi_subgroup', 'dsl_subgroup', 'ec_hte']
METHOD_DISPLAY = {
    'oracle': 'Oracle',
    'naive': 'Naive',
    'dsl_global': 'DSL-global',
    'ppi_global': 'PPI-global',
    'ppi_subgroup': 'PPI-subgroup',
    'dsl_subgroup': 'DSL-subgroup',
    'ec_hte': 'EC-HTE',
}


# === Data Loading (from exp-003) ===

def load_tweeteval_pool():
    ds = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
    texts, labels = [], []
    for split in ['train', 'test']:
        for row in ds[split]:
            texts.append(row['text'])
            labels.append(row['label'])
    return texts, np.array(labels)


def extract_features(texts):
    has_hashtag = np.array(['#' in t for t in texts], dtype=int)
    text_lengths = np.array([len(t) for t in texts])
    has_mention = np.array(['@' in t for t in texts], dtype=int)
    return has_hashtag, text_lengths, has_mention


def sample_from_pool(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                     pool_has_mention, n, rng):
    idx_neg = np.where(pool_labels == 0)[0]
    idx_neu = np.where(pool_labels == 1)[0]
    idx_pos = np.where(pool_labels == 2)[0]
    per_class = n // 3
    remainder = n - 3 * per_class
    chosen = np.concatenate([
        rng.choice(idx_neg, per_class, replace=len(idx_neg) < per_class),
        rng.choice(idx_neu, per_class, replace=len(idx_neu) < per_class),
        rng.choice(idx_pos, per_class + remainder, replace=len(idx_pos) < per_class + remainder),
    ])
    rng.shuffle(chosen)
    return ([pool_texts[i] for i in chosen], pool_labels[chosen],
            pool_has_hashtag[chosen], pool_text_lengths[chosen], pool_has_mention[chosen])


# === DGP ===

def generate_data(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                  pool_has_mention, k_subgroups, n_expert, seed):
    cfg = CONFIGS[k_subgroups]
    misclass_config = cfg['misclass']
    delta_config = cfg['delta']
    labels = cfg['labels']
    rng = np.random.RandomState(seed)

    texts, sent_labels, has_hashtag, text_lengths, has_mention = sample_from_pool(
        pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
        pool_has_mention, N, rng)

    median_len = np.median(text_lengths)
    text_length_bin = (text_lengths > median_len).astype(int)

    if k_subgroups == 2:
        S = np.where(has_hashtag == 0, 'S0', 'S1')
    else:
        S = np.empty(N, dtype='U2')
        S[(has_hashtag == 0) & (text_length_bin == 0)] = 'S0'
        S[(has_hashtag == 0) & (text_length_bin == 1)] = 'S1'
        S[(has_hashtag == 1) & (text_length_bin == 0)] = 'S2'
        S[(has_hashtag == 1) & (text_length_bin == 1)] = 'S3'

    Z = (np.array(sent_labels) == 2).astype(int)

    mc_rates = np.array([misclass_config[s] for s in S])
    flip = rng.rand(N) < mc_rates
    Z_hat = np.where(flip, 1 - Z, Z)

    T = rng.binomial(1, 0.5, N)

    text_length_norm = (text_lengths - text_lengths.mean()) / (text_lengths.std() + 1e-8)
    delta_s = np.array([delta_config[s] for s in S])
    tau = np.where(Z == 1, TAU_Z0 + delta_s, TAU_Z0)
    eps = rng.randn(N)
    Y = ALPHA + tau * T + BETA_LENGTH * text_length_norm + BETA_MENTION * has_mention + eps

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'Y': Y, 'tau': tau,
        'expert_mask': expert_mask, 'subgroup_labels': labels,
    }


# === Core Estimators ===

def diff_in_means(Y, T, mask):
    t1, t0 = mask & (T == 1), mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


def estimate_cm_mle(z_true, z_hat):
    C = np.zeros((2, 2))
    for zt in [0, 1]:
        mask = z_true == zt
        for zh in [0, 1]:
            C[zt, zh] = ((z_hat[mask] == zh).sum() + 1) / (mask.sum() + 2)
    return C


def build_mixing_matrix(C, p_z, p_z_hat):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        for z in [0, 1]:
            M[zh, z] = C[z, zh] * p_z[z] / max(p_z_hat[zh], 1e-10)
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    cond = np.linalg.cond(M)
    if cond > cond_threshold or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs**2) @ Mi.T), 0))
    return tau_c, se_c


# === HB Gibbs Sampler (from exp-003/exp-007) ===

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
                theta[:, z, :] = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-300)

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


# === Methods ===

def method_oracle(data, s_val):
    Y, T, Z, S = data['Y'], data['T'], data['Z'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z == 1))
    return th0, th1, se0, se1


def method_naive(data, s_val):
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    return th0, th1, se0, se1


def method_dsl_global(data, s_val, C_global, p_z_global, p_zh_global):
    """DSL-global ≡ Global correction: global M^{-1} applied to subgroup observed CATEs."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan
    M = build_mixing_matrix(C_global, p_z_global, p_zh_global)
    tau_c, se_c = invert_mixing_safe(M, np.array([th0, th1]), np.array([se0, se1]))
    return tau_c[0], tau_c[1], se_c[0], se_c[1]


def method_ppi_global(data, s_val):
    """PPI-global: global rectifier applied to subgroup CATEs.

    rectifier(z) = tau_gold_all(Z=z) - tau_f_gold_all(Z_hat=z)
    tau_PPI(z,s) = tau_naive(Z_hat=z, S=s) + rectifier(z)

    The global rectifier captures average bias across all subgroups,
    so it cannot correct subgroup-specific misclassification heterogeneity.
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    s_mask = S == s_val

    results = {}
    for z_val in [0, 1]:
        tau_all, se_all = diff_in_means(Y, T, s_mask & (Z_hat == z_val))
        tau_gold, se_gold = diff_in_means(Y, T, expert_mask & (Z == z_val))
        tau_f_gold, se_f_gold = diff_in_means(Y, T, expert_mask & (Z_hat == z_val))

        if np.isnan(tau_gold) or np.isnan(tau_f_gold) or np.isnan(tau_all):
            results[z_val] = (tau_all, se_all)
            continue

        rectifier = tau_gold - tau_f_gold
        tau_ppi = tau_all + rectifier
        se_ppi = np.sqrt(se_all**2 + se_gold**2 + se_f_gold**2)
        results[z_val] = (tau_ppi, se_ppi)

    th0, se0 = results.get(0, (np.nan, np.nan))
    th1, se1 = results.get(1, (np.nan, np.nan))
    return th0, th1, se0, se1


def method_ppi_subgroup(data, s_val):
    """PPI-subgroup: per-subgroup rectifier.

    rectifier(z,s) = tau_gold(Z=z, S=s) - tau_f_gold(Z_hat=z, S=s)
    tau_PPI(z,s) = tau_naive(Z_hat=z, S=s) + rectifier(z,s)

    Additive mean-shift correction — does NOT use confusion matrix structure.
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    s_mask = S == s_val
    expert_s = expert_mask & s_mask

    if expert_s.sum() < 10:
        th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        return th0, th1, se0, se1

    results = {}
    for z_val in [0, 1]:
        tau_all, se_all = diff_in_means(Y, T, s_mask & (Z_hat == z_val))
        tau_gold, se_gold = diff_in_means(Y, T, expert_s & (Z == z_val))
        tau_f_gold, se_f_gold = diff_in_means(Y, T, expert_s & (Z_hat == z_val))

        if np.isnan(tau_gold) or np.isnan(tau_f_gold) or np.isnan(tau_all):
            results[z_val] = (tau_all, se_all)
            continue

        rectifier = tau_gold - tau_f_gold
        tau_ppi = tau_all + rectifier
        se_ppi = np.sqrt(se_all**2 + se_gold**2 + se_f_gold**2)
        results[z_val] = (tau_ppi, se_ppi)

    th0, se0 = results.get(0, (np.nan, np.nan))
    th1, se1 = results.get(1, (np.nan, np.nan))
    return th0, th1, se0, se1


def method_dsl_subgroup(data, s_val, C_s, p_z_s, p_zh_s):
    """DSL-subgroup ≡ Stratified MLE: per-subgroup M_s^{-1}, no cross-subgroup pooling."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan
    M_s = build_mixing_matrix(C_s, p_z_s, p_zh_s)
    tau_c, se_c = invert_mixing_safe(M_s, np.array([th0, th1]), np.array([se0, se1]))
    return tau_c[0], tau_c[1], se_c[0], se_c[1]


def method_ec_hte(data, s_val, C_hb_s, p_z_s, p_zh_s):
    """EC-HTE: HB Gibbs partial-pooling confusion matrix + per-subgroup M_s^{-1}."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan
    M_s = build_mixing_matrix(C_hb_s, p_z_s, p_zh_s)
    tau_c, se_c = invert_mixing_safe(M_s, np.array([th0, th1]), np.array([se0, se1]))
    return tau_c[0], tau_c[1], se_c[0], se_c[1]


# === MC Runner ===

def run_one_mc(data, k_subgroups, n_expert, seed):
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    cfg = CONFIGS[k_subgroups]
    subgroup_labels = cfg['labels']
    delta_config = cfg['delta']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    # Global estimates
    p_z_global = np.clip(np.array([(z_exp == 0).mean(), (z_exp == 1).mean()]), 0.01, 0.99)
    p_zh_global = np.clip(np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()]), 0.01, 0.99)
    C_global = estimate_cm_mle(z_exp, zh_exp)

    # Per-subgroup MLE confusion matrices
    C_mle = {}
    for s_val in subgroup_labels:
        sm_exp = s_exp == s_val
        if sm_exp.sum() >= 4:
            C_mle[s_val] = estimate_cm_mle(z_exp[sm_exp], zh_exp[sm_exp])
        else:
            C_mle[s_val] = C_global

    # HB Gibbs partial-pooling confusion matrices
    counts = get_counts(z_exp, zh_exp, s_exp, subgroup_labels)
    C_hb = estimate_cm_hb_gibbs(counts, subgroup_labels, seed=seed)

    # True CATE per subgroup
    true_tau = {}
    for s_val in subgroup_labels:
        true_tau[s_val] = {'z0': TAU_Z0, 'z1': TAU_Z0 + delta_config[s_val]}

    base = {'k_subgroups': k_subgroups, 'n_expert': n_expert, 'mc_run': seed}
    rows = []

    def make_row(method, subgroup, th0, th1, se0, se1):
        tz0, tz1 = true_tau[subgroup]['z0'], true_tau[subgroup]['z1']
        return {
            **base, 'method': method, 'subgroup': subgroup,
            'tau_hat_z0': th0, 'tau_hat_z1': th1,
            'tau_true_z0': tz0, 'tau_true_z1': tz1,
            'ci_lower_z0': th0 - 1.96 * se0 if not np.isnan(se0) else np.nan,
            'ci_upper_z0': th0 + 1.96 * se0 if not np.isnan(se0) else np.nan,
            'ci_lower_z1': th1 - 1.96 * se1 if not np.isnan(se1) else np.nan,
            'ci_upper_z1': th1 + 1.96 * se1 if not np.isnan(se1) else np.nan,
        }

    for s_val in subgroup_labels:
        s_mask = S == s_val
        expert_s = expert_mask & s_mask

        # Per-subgroup marginals
        if (s_exp == s_val).sum() >= 4:
            p_z_s = np.clip(np.array([(z_exp[s_exp == s_val] == 0).mean(),
                                       (z_exp[s_exp == s_val] == 1).mean()]), 0.01, 0.99)
        else:
            p_z_s = p_z_global
        p_zh_s = np.clip(np.array([(Z_hat[s_mask] == 0).mean(),
                                    (Z_hat[s_mask] == 1).mean()]), 0.01, 0.99)

        # 1. Oracle
        th0, th1, se0, se1 = method_oracle(data, s_val)
        rows.append(make_row('oracle', s_val, th0, th1, se0, se1))

        # 2. Naive
        th0, th1, se0, se1 = method_naive(data, s_val)
        rows.append(make_row('naive', s_val, th0, th1, se0, se1))

        # 3. DSL-global (≡ Global correction)
        th0, th1, se0, se1 = method_dsl_global(data, s_val, C_global, p_z_global, p_zh_global)
        rows.append(make_row('dsl_global', s_val, th0, th1, se0, se1))

        # 4. PPI-global
        th0, th1, se0, se1 = method_ppi_global(data, s_val)
        rows.append(make_row('ppi_global', s_val, th0, th1, se0, se1))

        # 5. PPI-subgroup
        th0, th1, se0, se1 = method_ppi_subgroup(data, s_val)
        rows.append(make_row('ppi_subgroup', s_val, th0, th1, se0, se1))

        # 6. DSL-subgroup (≡ Stratified MLE)
        th0, th1, se0, se1 = method_dsl_subgroup(data, s_val, C_mle[s_val], p_z_s, p_zh_s)
        rows.append(make_row('dsl_subgroup', s_val, th0, th1, se0, se1))

        # 7. EC-HTE
        th0, th1, se0, se1 = method_ec_hte(data, s_val, C_hb[s_val], p_z_s, p_zh_s)
        rows.append(make_row('ec_hte', s_val, th0, th1, se0, se1))

    return rows


# === Aggregation ===

def aggregate_results(df):
    rows = []
    group_cols = ['k_subgroups', 'n_expert', 'subgroup', 'method']
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'] - grp[f'tau_true_{z}']
            valid = err.dropna()
            if len(valid) == 0:
                row.update({f'bias_{z}': np.nan, f'abs_bias_{z}': np.nan,
                            f'rmse_{z}': np.nan, f'coverage_{z}': np.nan})
                continue
            row[f'bias_{z}'] = valid.mean()
            row[f'abs_bias_{z}'] = np.abs(valid.mean())
            row[f'rmse_{z}'] = np.sqrt((valid**2).mean())
            covered = ((grp[f'ci_lower_{z}'] <= grp[f'tau_true_{z}']) &
                       (grp[f'tau_true_{z}'] <= grp[f'ci_upper_{z}'])).mean()
            row[f'coverage_{z}'] = covered
        rows.append(row)
    return pd.DataFrame(rows)


def make_summary_table(df_agg):
    """Average metrics across subgroups and Z-levels per (K, n_expert, method)."""
    rows = []
    for (k, ne, method), grp in df_agg.groupby(['k_subgroups', 'n_expert', 'method']):
        avg_abs_bias = np.nanmean([grp['abs_bias_z0'].mean(), grp['abs_bias_z1'].mean()])
        avg_rmse = np.nanmean([grp['rmse_z0'].mean(), grp['rmse_z1'].mean()])
        avg_coverage = np.nanmean([grp['coverage_z0'].mean(), grp['coverage_z1'].mean()])
        max_abs_bias = np.nanmax([grp['abs_bias_z0'].max(), grp['abs_bias_z1'].max()])
        rows.append({
            'K': int(k), 'n_expert': int(ne), 'method': method,
            'avg_abs_bias': avg_abs_bias, 'max_abs_bias': max_abs_bias,
            'avg_rmse': avg_rmse, 'avg_coverage': avg_coverage,
        })
    df = pd.DataFrame(rows)
    df['method_order'] = df['method'].map({m: i for i, m in enumerate(METHOD_ORDER)})
    df = df.sort_values(['K', 'n_expert', 'method_order']).drop(columns='method_order')
    df['method'] = df['method'].map(METHOD_DISPLAY)
    return df


# === Main ===

if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--dry-run']
    n_mc = int(args[0]) if args else N_MC

    if dry_run:
        n_mc = 2
        k_list = [2]
        ne_list = [100]
    else:
        k_list = [2, 4]
        ne_list = N_EXPERTS

    print(f"exp-022: PPI/DSL vs EC-HTE Baseline Comparison")
    print(f"N={N}, N_MC={n_mc}, N_EXPERTS={ne_list}, K={k_list}")
    print(f"Methods: {', '.join(METHOD_DISPLAY[m] for m in METHOD_ORDER)}")
    if dry_run:
        print("** DRY RUN **")
    print()

    print("Loading TweetEval sentiment dataset...")
    t_load = time.time()
    pool_texts, pool_labels = load_tweeteval_pool()
    pool_has_hashtag, pool_text_lengths, pool_has_mention = extract_features(pool_texts)
    print(f"  Pool: {len(pool_texts)} tweets, loaded in {time.time()-t_load:.1f}s")
    print()

    results = []
    configs = [(k, ne) for k in k_list for ne in ne_list]
    total = len(configs) * n_mc
    count = 0
    t0 = time.time()

    for k, ne in configs:
        for mc in range(n_mc):
            seed = SEED_BASE + mc
            data = generate_data(pool_texts, pool_labels, pool_has_hashtag,
                                 pool_text_lengths, pool_has_mention, k, ne, seed)
            mc_rows = run_one_mc(data, k, ne, seed)
            results.extend(mc_rows)
            count += 1
            if count % max(1, total // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | K={k} ne={ne} mc={mc}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({count} iterations)")

    df_raw = pd.DataFrame(results)
    df_agg = aggregate_results(df_raw)
    df_summary = make_summary_table(df_agg)

    raw_path = 'results/exp022_ppi_dsl_raw.csv'
    agg_path = 'results/exp022_ppi_dsl_agg.csv'
    summary_path = 'results/exp022_ppi_dsl_summary.csv'
    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {raw_path} ({len(df_raw)} rows)")
    print(f"Saved: {agg_path} ({len(df_agg)} rows)")
    print(f"Saved: {summary_path}")

    # Print summary tables
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 200)

    for k in k_list:
        print(f"\n{'='*90}")
        cfg = CONFIGS[k]
        mc_str = ', '.join(f'{s}={cfg["misclass"][s]}' for s in cfg['labels'])
        delta_str = ', '.join(f'{s}={cfg["delta"][s]}' for s in cfg['labels'])
        print(f"K={k} subgroups | misclass: {mc_str}")
        print(f"  tau(Z=0)={TAU_Z0} | delta: {delta_str}")
        print(f"{'='*90}")
        sub = df_summary[df_summary['K'] == k]
        print(sub[['n_expert', 'method', 'avg_abs_bias', 'max_abs_bias',
                    'avg_rmse', 'avg_coverage']].to_string(index=False))

    # Per-subgroup detail for the most informative case (K=4, n_expert=250)
    print(f"\n{'='*90}")
    print("Per-subgroup detail: K=4, n_expert=250")
    print(f"{'='*90}")
    detail = df_agg[(df_agg['k_subgroups'] == 4) & (df_agg['n_expert'] == 250)].copy()
    if not detail.empty:
        detail['method_order'] = detail['method'].map({m: i for i, m in enumerate(METHOD_ORDER)})
        detail = detail.sort_values(['subgroup', 'method_order'])
        detail['method'] = detail['method'].map(METHOD_DISPLAY)
        cols = ['subgroup', 'method', 'bias_z0', 'bias_z1', 'rmse_z0', 'rmse_z1',
                'coverage_z0', 'coverage_z1']
        print(detail[cols].to_string(index=False))
