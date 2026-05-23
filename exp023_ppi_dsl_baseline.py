"""
exp-023: PPI/DSL baseline comparison for EC-HTE positioning.

Demonstrates that PPI and DSL, when applied to CATE estimation with
mismeasured effect modifiers, reduce to existing baselines:
  - DSL-global ≡ Global correction (same confusion matrix inversion)
  - DSL-subgroup ≡ Stratified MLE (per-subgroup CM inversion)
  - PPI-subgroup ≈ Stratified MLE (different estimator, similar performance)
  - EC-HTE (HB partial pooling) outperforms all of the above

Uses the same TweetEval semi-synthetic DGP as exp-003 (same seeds for reproducibility).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.special import gammaln

warnings.filterwarnings('ignore')

# === Config (same as exp-003) ===

N = 5000
N_MC = 50
N_EXPERTS = [100, 250, 500]
SEED_BASE = 42

MISCLASS_RATES = {
    'extreme': {'S0': 0.05, 'S1': 0.10, 'S2': 0.15, 'S3': 0.25},
    'moderate': {'S0': 0.08, 'S1': 0.10, 'S2': 0.12, 'S3': 0.15},
    'homogeneous': {'S0': 0.10, 'S1': 0.10, 'S2': 0.10, 'S3': 0.10},
}

DELTA_S = {'S0': 0.4, 'S1': 0.3, 'S2': 0.2, 'S3': 0.1}
TAU_Z0 = 0.1
ALPHA = 1.0
BETA_LENGTH = 0.3
BETA_MENTION = 0.2
SUBGROUP_LABELS = ['S0', 'S1', 'S2', 'S3']


# === Data Loading (same as exp-003) ===

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


def assign_subgroups(has_hashtag, text_length_bin):
    S = np.empty(len(has_hashtag), dtype='U2')
    S[(has_hashtag == 0) & (text_length_bin == 0)] = 'S0'
    S[(has_hashtag == 0) & (text_length_bin == 1)] = 'S1'
    S[(has_hashtag == 1) & (text_length_bin == 0)] = 'S2'
    S[(has_hashtag == 1) & (text_length_bin == 1)] = 'S3'
    return S


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


# === HB Gibbs Sampler (from exp-003) ===

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


def _split_rhat(chains):
    parts = []
    for c in range(chains.shape[0]):
        half = chains.shape[1] // 2
        parts.append(chains[c, :half])
        parts.append(chains[c, half:2 * half])
    parts = np.array(parts)
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


def _fast_diagnostics(alpha_arr, theta_arr, labels):
    rhats, esss = [], []
    for d in range(alpha_arr.shape[2]):
        rhats.append(_split_rhat(alpha_arr[:, :, d]))
        esss.append(_bulk_ess(alpha_arr[:, :, d]))
    k = len(labels)
    for i in range(k):
        for z in [0, 1]:
            for zh in range(theta_arr.shape[4]):
                ch = theta_arr[:, :, i, z, zh]
                rhats.append(_split_rhat(ch))
                esss.append(_bulk_ess(ch))
    return max(rhats), min(esss)


def estimate_cm_hb_gibbs(counts, labels, seed=42, n_iter=1000, n_warmup=500, n_chains=4,
                         hp_a0=3.0, hp_b0=1.0):
    k = len(labels)
    chain_inits_pool = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                        np.array([0.5, 0.5]), np.array([10.0, 10.0])]
    chain_inits = chain_inits_pool[:n_chains]
    all_alpha, all_theta = [], []

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

    alpha_arr = np.stack(all_alpha)
    theta_arr = np.stack(all_theta)

    C = {}
    for i, s in enumerate(labels):
        C[s] = theta_arr[:, :, i, :, :].mean(axis=(0, 1))

    rhat_max, ess_min = _fast_diagnostics(alpha_arr, theta_arr, labels)
    diag = {'rhat_max': rhat_max, 'ess_min': ess_min}
    return C, diag


# === DGP (same as exp-003) ===

def sample_from_pool(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                     pool_has_mention, n, rng):
    idx_neg = np.where(pool_labels == 0)[0]
    idx_neu = np.where(pool_labels == 1)[0]
    idx_pos = np.where(pool_labels == 2)[0]
    per_class = n // 3
    remainder = n - 3 * per_class
    chosen_neg = rng.choice(idx_neg, per_class, replace=len(idx_neg) < per_class)
    chosen_neu = rng.choice(idx_neu, per_class, replace=len(idx_neu) < per_class)
    chosen_pos = rng.choice(idx_pos, per_class + remainder, replace=len(idx_pos) < per_class + remainder)
    idx = np.concatenate([chosen_neg, chosen_neu, chosen_pos])
    rng.shuffle(idx)
    return (
        [pool_texts[i] for i in idx], pool_labels[idx],
        pool_has_hashtag[idx], pool_text_lengths[idx], pool_has_mention[idx],
    )


def generate_data(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                  pool_has_mention, misclass_config, n_expert, seed):
    rng = np.random.RandomState(seed)
    texts, labels, has_hashtag, text_lengths, has_mention = sample_from_pool(
        pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
        pool_has_mention, N, rng
    )
    median_len = np.median(text_lengths)
    text_length_bin = (text_lengths > median_len).astype(int)
    S = assign_subgroups(has_hashtag, text_length_bin)
    Z = (np.array(labels) == 2).astype(int)
    mc_rates = np.array([misclass_config[s] for s in S])
    flip = rng.rand(N) < mc_rates
    Z_hat = np.where(flip, 1 - Z, Z)
    T = rng.binomial(1, 0.5, N)
    text_length_norm = (text_lengths - text_lengths.mean()) / (text_lengths.std() + 1e-8)
    delta_s = np.array([DELTA_S[s] for s in S])
    tau = np.where(Z == 1, TAU_Z0 + delta_s, TAU_Z0)
    eps = rng.randn(N)
    Y = ALPHA + tau * T + BETA_LENGTH * text_length_norm + BETA_MENTION * has_mention + eps
    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True
    return {'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'Y': Y, 'tau': tau, 'expert_mask': expert_mask}


# === PPI Methods ===

def method_ppi_global(data):
    """PPI with a single global rectifier applied to all subgroups.

    rectifier(z) = tau_gold(Z=z, all) - tau_hat_gold(Z_hat=z, all)
    tau_ppi(z, s) = tau_all(Z_hat=z, s) + rectifier(z)
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']

    rect = {}
    rect_se = {}
    for z_val in [0, 1]:
        tau_gold, se_gold = diff_in_means(Y, T, expert_mask & (Z == z_val))
        tau_hat_gold, se_hat_gold = diff_in_means(Y, T, expert_mask & (Z_hat == z_val))
        if np.isnan(tau_gold) or np.isnan(tau_hat_gold):
            rect[z_val] = 0.0
            rect_se[z_val] = 0.0
        else:
            rect[z_val] = tau_gold - tau_hat_gold
            rect_se[z_val] = np.sqrt(se_gold**2 + se_hat_gold**2)

    results = {}
    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val
        th0_all, se0_all = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1_all, se1_all = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        if np.isnan(th0_all) or np.isnan(th1_all):
            results[s_val] = (np.nan, np.nan, np.nan, np.nan)
        else:
            th0_ppi = th0_all + rect[0]
            th1_ppi = th1_all + rect[1]
            se0_ppi = np.sqrt(se0_all**2 + rect_se[0]**2)
            se1_ppi = np.sqrt(se1_all**2 + rect_se[1]**2)
            results[s_val] = (th0_ppi, th1_ppi, se0_ppi, se1_ppi)

    # Marginal
    th0_all, se0_all = diff_in_means(Y, T, Z_hat == 0)
    th1_all, se1_all = diff_in_means(Y, T, Z_hat == 1)
    if np.isnan(th0_all) or np.isnan(th1_all):
        results['marginal'] = (np.nan, np.nan, np.nan, np.nan)
    else:
        results['marginal'] = (th0_all + rect[0], th1_all + rect[1],
                               np.sqrt(se0_all**2 + rect_se[0]**2),
                               np.sqrt(se1_all**2 + rect_se[1]**2))
    return results


def method_ppi_subgroup(data):
    """PPI with per-subgroup rectifier (no cross-subgroup pooling).

    For each subgroup s independently:
      rectifier_s(z) = tau_gold(Z=z, s) - tau_hat_gold(Z_hat=z, s)
      tau_ppi(z, s) = tau_all(Z_hat=z, s) + rectifier_s(z)
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']

    results = {}
    sub_tau = {}
    sub_se = {}
    sub_n = {}

    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val
        expert_s = expert_mask & s_mask
        sub_n[s_val] = s_mask.sum()

        th0_all, se0_all = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1_all, se1_all = diff_in_means(Y, T, s_mask & (Z_hat == 1))

        if expert_s.sum() < 10 or np.isnan(th0_all) or np.isnan(th1_all):
            results[s_val] = (th0_all, th1_all, se0_all, se1_all)
            sub_tau[s_val] = np.array([th0_all, th1_all])
            sub_se[s_val] = np.array([se0_all, se1_all])
            continue

        tau_z = np.zeros(2)
        se_z = np.zeros(2)
        for z_val in [0, 1]:
            tau_gold, se_gold = diff_in_means(Y, T, expert_s & (Z == z_val))
            tau_hat_gold, se_hat_gold = diff_in_means(Y, T, expert_s & (Z_hat == z_val))
            tau_all = th0_all if z_val == 0 else th1_all
            se_all = se0_all if z_val == 0 else se1_all

            if np.isnan(tau_gold) or np.isnan(tau_hat_gold):
                tau_z[z_val] = tau_all
                se_z[z_val] = se_all
            else:
                rect = tau_gold - tau_hat_gold
                tau_z[z_val] = tau_all + rect
                se_z[z_val] = np.sqrt(se_all**2 + se_gold**2 + se_hat_gold**2)

        results[s_val] = (tau_z[0], tau_z[1], se_z[0], se_z[1])
        sub_tau[s_val] = tau_z
        sub_se[s_val] = se_z

    # Marginal: weighted average of subgroup estimates
    total_n = sum(sub_n.values())
    w = {s: sub_n[s] / total_n for s in SUBGROUP_LABELS}
    tau_m = sum(w[s] * sub_tau[s] for s in SUBGROUP_LABELS)
    se_m = np.sqrt(sum((w[s] * sub_se[s])**2 for s in SUBGROUP_LABELS))
    results['marginal'] = (tau_m[0], tau_m[1], se_m[0], se_m[1])

    return results


# === MC Runner ===

def run_one_mc(data, regime_name, n_expert, seed):
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])
    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    C_sub, M_sub = {}, {}
    for s_val in SUBGROUP_LABELS:
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        if sm_exp.sum() >= 4:
            C_sub[s_val] = estimate_confusion_matrix(z_exp[sm_exp], zh_exp[sm_exp])
            p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()])
        else:
            C_sub[s_val] = C_global
            p_z_s = p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_sub[s_val] = build_mixing_matrix(C_sub[s_val], p_z_s, p_zh_s)

    counts = get_counts(z_exp, zh_exp, s_exp, SUBGROUP_LABELS)
    C_hb, _ = estimate_cm_hb_gibbs(counts, SUBGROUP_LABELS, seed=seed)

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

    true_tau = {}
    for s_val in SUBGROUP_LABELS:
        true_tau[s_val] = {'z0': TAU_Z0, 'z1': TAU_Z0 + DELTA_S[s_val]}

    sub_n = {s: (S == s).sum() for s in SUBGROUP_LABELS}
    total_n = sum(sub_n.values())
    w = {s: sub_n[s] / total_n for s in SUBGROUP_LABELS}
    true_z0_marg = TAU_Z0
    true_z1_marg = TAU_Z0 + sum(w[s] * DELTA_S[s] for s in SUBGROUP_LABELS)

    base = {'regime': regime_name, 'n_expert': n_expert, 'mc_run': seed}
    rows = []

    def make_row(method, level, th0, th1, se0, se1, true_z0, true_z1):
        ci_lo0 = th0 - 1.96 * se0 if not np.isnan(se0) else np.nan
        ci_hi0 = th0 + 1.96 * se0 if not np.isnan(se0) else np.nan
        ci_lo1 = th1 - 1.96 * se1 if not np.isnan(se1) else np.nan
        ci_hi1 = th1 + 1.96 * se1 if not np.isnan(se1) else np.nan
        return {
            **base, 'method': method, 'level': level,
            'tau_hat_z0': th0, 'tau_hat_z1': th1,
            'tau_true_z0': true_z0, 'tau_true_z1': true_z1,
            'ci_lower_z0': ci_lo0, 'ci_upper_z0': ci_hi0,
            'ci_lower_z1': ci_lo1, 'ci_upper_z1': ci_hi1,
        }

    # Compute PPI results
    ppi_global = method_ppi_global(data)
    ppi_sub = method_ppi_subgroup(data)

    sub_obs, sub_se_obs = {}, {}
    sub_corrected_strat, sub_se_strat = {}, {}
    sub_corrected_hb, sub_se_hb = {}, {}

    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val
        tz0, tz1 = true_tau[s_val]['z0'], true_tau[s_val]['z1']

        th0_obs, se0_obs = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1_obs, se1_obs = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        sub_obs[s_val] = np.array([th0_obs, th1_obs])
        sub_se_obs[s_val] = np.array([se0_obs, se1_obs])

        # Oracle
        th0_or, se0_or = diff_in_means(Y, T, s_mask & (Z == 0))
        th1_or, se1_or = diff_in_means(Y, T, s_mask & (Z == 1))
        rows.append(make_row('oracle', s_val, th0_or, th1_or, se0_or, se1_or, tz0, tz1))

        # Naive
        rows.append(make_row('naive', s_val, th0_obs, th1_obs, se0_obs, se1_obs, tz0, tz1))

        # Global-corrected (= DSL-global)
        if not np.any(np.isnan(sub_obs[s_val])):
            tau_gc, se_gc = invert_mixing_safe(M_global, sub_obs[s_val], sub_se_obs[s_val])
            rows.append(make_row('global_corrected', s_val, tau_gc[0], tau_gc[1], se_gc[0], se_gc[1], tz0, tz1))
        else:
            rows.append(make_row('global_corrected', s_val, np.nan, np.nan, np.nan, np.nan, tz0, tz1))

        # Stratified MLE (= DSL-subgroup)
        tau_strat, se_strat = invert_mixing_safe(M_sub[s_val], sub_obs[s_val], sub_se_obs[s_val])
        sub_corrected_strat[s_val] = tau_strat
        sub_se_strat[s_val] = se_strat
        rows.append(make_row('stratified_mle', s_val, tau_strat[0], tau_strat[1], se_strat[0], se_strat[1], tz0, tz1))

        # PPI-global
        pg = ppi_global[s_val]
        rows.append(make_row('ppi_global', s_val, pg[0], pg[1], pg[2], pg[3], tz0, tz1))

        # PPI-subgroup
        ps = ppi_sub[s_val]
        rows.append(make_row('ppi_subgroup', s_val, ps[0], ps[1], ps[2], ps[3], tz0, tz1))

        # EC-HTE (HB partial pooling)
        tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], sub_obs[s_val], sub_se_obs[s_val])
        sub_corrected_hb[s_val] = tau_hb
        sub_se_hb[s_val] = se_hb
        rows.append(make_row('ec_hte', s_val, tau_hb[0], tau_hb[1], se_hb[0], se_hb[1], tz0, tz1))

    # Marginal-level
    th0_or, se0_or = diff_in_means(Y, T, Z == 0)
    th1_or, se1_or = diff_in_means(Y, T, Z == 1)
    rows.append(make_row('oracle', 'marginal', th0_or, th1_or, se0_or, se1_or, true_z0_marg, true_z1_marg))

    th0_naive, se0_naive = diff_in_means(Y, T, Z_hat == 0)
    th1_naive, se1_naive = diff_in_means(Y, T, Z_hat == 1)
    rows.append(make_row('naive', 'marginal', th0_naive, th1_naive, se0_naive, se1_naive, true_z0_marg, true_z1_marg))

    tau_gc_m, se_gc_m = invert_mixing_safe(M_global, np.array([th0_naive, th1_naive]), np.array([se0_naive, se1_naive]))
    rows.append(make_row('global_corrected', 'marginal', tau_gc_m[0], tau_gc_m[1], se_gc_m[0], se_gc_m[1], true_z0_marg, true_z1_marg))

    tau_strat_m = sum(w[s] * sub_corrected_strat[s] for s in SUBGROUP_LABELS)
    se_strat_m = np.sqrt(sum((w[s] * sub_se_strat[s])**2 for s in SUBGROUP_LABELS))
    rows.append(make_row('stratified_mle', 'marginal', tau_strat_m[0], tau_strat_m[1], se_strat_m[0], se_strat_m[1], true_z0_marg, true_z1_marg))

    pg = ppi_global['marginal']
    rows.append(make_row('ppi_global', 'marginal', pg[0], pg[1], pg[2], pg[3], true_z0_marg, true_z1_marg))

    ps = ppi_sub['marginal']
    rows.append(make_row('ppi_subgroup', 'marginal', ps[0], ps[1], ps[2], ps[3], true_z0_marg, true_z1_marg))

    tau_hb_m = sum(w[s] * sub_corrected_hb[s] for s in SUBGROUP_LABELS)
    se_hb_m = np.sqrt(sum((w[s] * sub_se_hb[s])**2 for s in SUBGROUP_LABELS))
    rows.append(make_row('ec_hte', 'marginal', tau_hb_m[0], tau_hb_m[1], se_hb_m[0], se_hb_m[1], true_z0_marg, true_z1_marg))

    return rows


def aggregate_results(df):
    rows = []
    group_cols = ['regime', 'n_expert', 'method', 'level']
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for z in ['z0', 'z1']:
            err = grp[f'tau_hat_{z}'] - grp[f'tau_true_{z}']
            valid = err.dropna()
            if len(valid) == 0:
                row.update({f'bias_{z}': np.nan, f'rmse_{z}': np.nan, f'coverage_{z}': np.nan})
                continue
            row[f'bias_{z}'] = valid.mean()
            row[f'rmse_{z}'] = np.sqrt((valid**2).mean())
            covered = ((grp[f'ci_lower_{z}'] <= grp[f'tau_true_{z}']) &
                       (grp[f'tau_true_{z}'] <= grp[f'ci_upper_{z}'])).mean()
            row[f'coverage_{z}'] = covered
            row[f'bias_{z}_se'] = valid.std() / np.sqrt(len(valid))
        rows.append(row)
    return pd.DataFrame(rows)


# === Main ===

if __name__ == '__main__':
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else N_MC

    print(f"exp-023: PPI/DSL Baseline Comparison")
    print(f"N={N}, N_MC={n_mc}, N_EXPERTS={N_EXPERTS}")
    print(f"Methods: oracle, naive, global_corrected (=DSL-global), stratified_mle (=DSL-subgroup),")
    print(f"         ppi_global, ppi_subgroup, ec_hte (HB partial pooling)")
    print()

    print("Loading TweetEval sentiment dataset...")
    t_load = time.time()
    pool_texts, pool_labels = load_tweeteval_pool()
    pool_has_hashtag, pool_text_lengths, pool_has_mention = extract_features(pool_texts)
    print(f"  Pool size: {len(pool_texts)}, loaded in {time.time()-t_load:.1f}s")
    print()

    results = []
    configs = [(r, ne) for r in MISCLASS_RATES for ne in N_EXPERTS]
    total = len(configs) * n_mc
    count = 0
    t0 = time.time()

    for regime_name, ne in configs:
        misclass_config = MISCLASS_RATES[regime_name]
        for mc in range(n_mc):
            seed = SEED_BASE + mc
            data = generate_data(
                pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                pool_has_mention, misclass_config, ne, seed
            )
            mc_rows = run_one_mc(data, regime_name, ne, seed)
            results.extend(mc_rows)
            count += 1
            if count % max(1, total // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | {regime_name} ne={ne} mc={mc}")

    elapsed = time.time() - t0
    print(f"\nTotal MC time: {elapsed:.1f}s")

    df_raw = pd.DataFrame(results)
    df_agg = aggregate_results(df_raw)

    raw_path = 'results/exp023_ppi_dsl_raw.csv'
    agg_path = 'results/exp023_ppi_dsl.csv'
    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved: {raw_path} ({len(df_raw)} rows)")
    print(f"Saved: {agg_path} ({len(df_agg)} rows)")

    # === Print comparison table ===
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 260)
    pd.set_option('display.float_format', '{:.4f}'.format)

    method_order = ['oracle', 'naive', 'global_corrected', 'ppi_global',
                    'stratified_mle', 'ppi_subgroup', 'ec_hte']
    method_display = {
        'oracle': 'Oracle',
        'naive': 'Naive (no correction)',
        'global_corrected': 'Global Correction (=DSL-global)',
        'ppi_global': 'PPI-global',
        'stratified_mle': 'Stratified MLE (=DSL-subgroup)',
        'ppi_subgroup': 'PPI-subgroup',
        'ec_hte': 'EC-HTE (HB partial pooling)',
    }

    for regime in MISCLASS_RATES:
        pi = MISCLASS_RATES[regime]
        print(f"\n{'='*140}")
        print(f"Regime: {regime} | misclass: S0={pi['S0']}, S1={pi['S1']}, S2={pi['S2']}, S3={pi['S3']}")
        print(f"{'='*140}")

        for ne in N_EXPERTS:
            print(f"\n--- n_expert = {ne} ---")

            # Subgroup-averaged metrics
            sub = df_agg[(df_agg['regime'] == regime) & (df_agg['n_expert'] == ne)
                         & (df_agg['level'].isin(SUBGROUP_LABELS))]
            if sub.empty:
                continue

            print(f"{'Method':<42} {'Avg|Bias_z0|':>12} {'Avg|Bias_z1|':>12} "
                  f"{'Avg RMSE_z0':>12} {'Avg RMSE_z1':>12} "
                  f"{'Avg Cov_z0':>10} {'Avg Cov_z1':>10}")
            print("-" * 120)

            for method in method_order:
                ms = sub[sub['method'] == method]
                if ms.empty:
                    continue
                label = method_display.get(method, method)
                avg_abs_bias_z0 = ms['bias_z0'].abs().mean()
                avg_abs_bias_z1 = ms['bias_z1'].abs().mean()
                avg_rmse_z0 = ms['rmse_z0'].mean()
                avg_rmse_z1 = ms['rmse_z1'].mean()
                avg_cov_z0 = ms['coverage_z0'].mean()
                avg_cov_z1 = ms['coverage_z1'].mean()
                print(f"{label:<42} {avg_abs_bias_z0:>12.4f} {avg_abs_bias_z1:>12.4f} "
                      f"{avg_rmse_z0:>12.4f} {avg_rmse_z1:>12.4f} "
                      f"{avg_cov_z0:>10.2f} {avg_cov_z1:>10.2f}")

    # === Equivalence check: PPI-subgroup vs Stratified MLE ===
    print(f"\n\n{'='*100}")
    print("EQUIVALENCE CHECK: PPI-subgroup vs Stratified MLE (across all configs)")
    print(f"{'='*100}")

    for regime in MISCLASS_RATES:
        for ne in N_EXPERTS:
            sub_ppi = df_agg[(df_agg['regime'] == regime) & (df_agg['n_expert'] == ne)
                             & (df_agg['method'] == 'ppi_subgroup')
                             & (df_agg['level'].isin(SUBGROUP_LABELS))]
            sub_mle = df_agg[(df_agg['regime'] == regime) & (df_agg['n_expert'] == ne)
                             & (df_agg['method'] == 'stratified_mle')
                             & (df_agg['level'].isin(SUBGROUP_LABELS))]
            if sub_ppi.empty or sub_mle.empty:
                continue

            for z in ['z0', 'z1']:
                bias_diff = (sub_ppi[f'bias_{z}'].values - sub_mle[f'bias_{z}'].values)
                rmse_diff = (sub_ppi[f'rmse_{z}'].values - sub_mle[f'rmse_{z}'].values)
                print(f"  {regime} ne={ne} {z}: "
                      f"bias_diff={bias_diff.mean():.4f}±{bias_diff.std():.4f}  "
                      f"rmse_diff={rmse_diff.mean():.4f}±{rmse_diff.std():.4f}")
