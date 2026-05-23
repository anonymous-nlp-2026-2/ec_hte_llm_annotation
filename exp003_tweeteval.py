"""
exp-003 Phase A v3: Semi-synthetic benchmark on TweetEval sentiment.
Uses real text data with simulated LLM misclassification to validate EC-HTE
on realistic subgroup structures.

v3: Added HB Dirichlet (partial pooling) confusion matrix estimator from exp-001
    Gibbs sampler alongside stratified MLE to validate Claim 2.

Input: TweetEval sentiment (cardiffnlp/tweet_eval, sentiment split)
Output: results/exp003_phase_a_v3.csv, results/exp003_phase_a_v3_raw.csv, results/exp003_dgp_spec.md
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.special import gammaln

warnings.filterwarnings('ignore')

# === Config ===

N = 5000
N_MC = 100
N_EXPERTS = [50, 100, 250, 500]
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


# === Data Loading ===

def load_tweeteval_pool():
    """Load and merge TweetEval sentiment train+test into a single pool."""
    ds = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
    texts, labels = [], []
    for split in ['train', 'test']:
        for row in ds[split]:
            texts.append(row['text'])
            labels.append(row['label'])
    return texts, np.array(labels)


def extract_features(texts):
    """Extract subgroup features from tweet texts."""
    has_hashtag = np.array(['#' in t for t in texts], dtype=int)
    text_lengths = np.array([len(t) for t in texts])
    has_mention = np.array(['@' in t for t in texts], dtype=int)
    return has_hashtag, text_lengths, has_mention


def assign_subgroups(has_hashtag, text_length_bin):
    """Assign K=4 subgroups based on hashtag × text_length_bin."""
    S = np.empty(len(has_hashtag), dtype='U2')
    S[(has_hashtag == 0) & (text_length_bin == 0)] = 'S0'
    S[(has_hashtag == 0) & (text_length_bin == 1)] = 'S1'
    S[(has_hashtag == 1) & (text_length_bin == 0)] = 'S2'
    S[(has_hashtag == 1) & (text_length_bin == 1)] = 'S3'
    return S


# === Core Estimators (reused from MVP) ===

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
    """M[z_hat, z] = P(Z=z | Z_hat=z_hat) via Bayes' rule.

    Args:
        C: 2x2 confusion matrix, C[z, z_hat] = P(Z_hat=z_hat | Z=z)
        p_z: array [P(Z=0), P(Z=1)] - marginal distribution of true Z
        p_z_hat: array [P(Z_hat=0), P(Z_hat=1)] - marginal distribution of Z_hat
    """
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        for z in [0, 1]:
            M[zh, z] = C[z, zh] * p_z[z] / p_z_hat[zh] if p_z_hat[zh] > 0 else 0.5
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    """Like invert_mixing but with condition number guard."""
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


# === HB Gibbs Sampler (from exp-001) ===

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


def _log_target_alpha(alpha, theta_z, hp_a0=3.0, hp_b0=1.0):
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, labels, seed=42, n_iter=4000, n_warmup=2000, n_chains=4,
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

    rhat_max, ess_min = _fast_diagnostics(alpha_arr, theta_arr, labels)

    diag = {'rhat_max': rhat_max, 'ess_min': ess_min, 'divergent_count': 0,
            'runtime_seconds': runtime, 'hb_method': 'gibbs'}
    return C, diag


# === DGP ===

def sample_from_pool(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                     pool_has_mention, n, rng):
    """Sample n observations with approximate label balance from the pool."""
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
        [pool_texts[i] for i in idx],
        pool_labels[idx],
        pool_has_hashtag[idx],
        pool_text_lengths[idx],
        pool_has_mention[idx],
    )


def generate_data(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
                  pool_has_mention, misclass_config, n_expert, seed):
    """Generate one MC dataset: sample tweets, assign Z, simulate misclass + outcome."""
    rng = np.random.RandomState(seed)

    texts, labels, has_hashtag, text_lengths, has_mention = sample_from_pool(
        pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths,
        pool_has_mention, N, rng
    )

    median_len = np.median(text_lengths)
    text_length_bin = (text_lengths > median_len).astype(int)

    S = assign_subgroups(has_hashtag, text_length_bin)

    # Z: effect modifier from sentiment
    Z = (np.array(labels) == 2).astype(int)  # positive=1, else=0

    # Misclassification
    mc_rates = np.array([misclass_config[s] for s in S])
    flip = rng.rand(N) < mc_rates
    Z_hat = np.where(flip, 1 - Z, Z)

    # Treatment assignment (RCT)
    T = rng.binomial(1, 0.5, N)

    # Outcome
    text_length_norm = (text_lengths - text_lengths.mean()) / (text_lengths.std() + 1e-8)
    delta_s = np.array([DELTA_S[s] for s in S])
    tau = np.where(Z == 1, TAU_Z0 + delta_s, TAU_Z0)
    eps = rng.randn(N)
    Y = ALPHA + tau * T + BETA_LENGTH * text_length_norm + BETA_MENTION * has_mention + eps

    # Expert mask
    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'Y': Y, 'tau': tau,
        'expert_mask': expert_mask, 'has_hashtag': has_hashtag,
        'text_length_bin': text_length_bin, 'text_lengths': text_lengths,
        'has_mention': has_mention, 'labels': np.array(labels),
    }


# === MC Runner ===

def run_one_mc(data, regime_name, n_expert, seed):
    """Run one MC iteration: all methods × all subgroups + marginal."""
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    # Global marginals for mixing matrix
    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])

    # Global confusion matrix
    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    # Per-subgroup confusion matrices
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

    # HB Dirichlet confusion matrices (partial pooling via Gibbs sampler)
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

    # True tau per subgroup
    true_tau = {}
    for s_val in SUBGROUP_LABELS:
        true_tau[s_val] = {
            'z0': TAU_Z0,
            'z1': TAU_Z0 + DELTA_S[s_val],
        }

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

    # --- Subgroup-level estimates ---
    sub_obs, sub_se_obs = {}, {}
    sub_corrected_strat, sub_se_strat = {}, {}
    sub_corrected_hb, sub_se_hb = {}, {}
    sub_n = {}

    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val
        sub_n[s_val] = s_mask.sum()

        th0_obs, se0_obs = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1_obs, se1_obs = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        sub_obs[s_val] = np.array([th0_obs, th1_obs])
        sub_se_obs[s_val] = np.array([se0_obs, se1_obs])

        tau_strat, se_strat = invert_mixing_safe(M_sub[s_val], sub_obs[s_val], sub_se_obs[s_val])
        sub_corrected_strat[s_val] = tau_strat
        sub_se_strat[s_val] = se_strat

        tz0 = true_tau[s_val]['z0']
        tz1 = true_tau[s_val]['z1']

        # Oracle
        th0_or, se0_or = diff_in_means(Y, T, s_mask & (Z == 0))
        th1_or, se1_or = diff_in_means(Y, T, s_mask & (Z == 1))
        rows.append(make_row('oracle', s_val, th0_or, th1_or, se0_or, se1_or, tz0, tz1))

        # Naive
        rows.append(make_row('naive', s_val, th0_obs, th1_obs, se0_obs, se1_obs, tz0, tz1))

        # Global-corrected
        if not np.any(np.isnan(sub_obs[s_val])):
            tau_gc, se_gc = invert_mixing_safe(M_global, sub_obs[s_val], sub_se_obs[s_val])
            rows.append(make_row('global_corrected', s_val, tau_gc[0], tau_gc[1], se_gc[0], se_gc[1], tz0, tz1))
        else:
            rows.append(make_row('global_corrected', s_val, np.nan, np.nan, np.nan, np.nan, tz0, tz1))

        # Stratified MLE (= ec_hte placeholder)
        rows.append(make_row('stratified_mle', s_val, tau_strat[0], tau_strat[1], se_strat[0], se_strat[1], tz0, tz1))

        # HB EC-HTE (partial pooling)
        tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], sub_obs[s_val], sub_se_obs[s_val])
        sub_corrected_hb[s_val] = tau_hb
        sub_se_hb[s_val] = se_hb
        rows.append(make_row('hb_ec_hte', s_val, tau_hb[0], tau_hb[1], se_hb[0], se_hb[1], tz0, tz1))

    # --- Marginal-level estimates ---
    # True marginal tau: weighted average across subgroups
    total_n = sum(sub_n.values())
    w = {s: sub_n[s] / total_n for s in SUBGROUP_LABELS}

    true_z0_marg = TAU_Z0
    true_z1_marg = TAU_Z0 + sum(w[s] * DELTA_S[s] for s in SUBGROUP_LABELS)

    # Oracle marginal
    th0_or, se0_or = diff_in_means(Y, T, Z == 0)
    th1_or, se1_or = diff_in_means(Y, T, Z == 1)
    rows.append(make_row('oracle', 'marginal', th0_or, th1_or, se0_or, se1_or, true_z0_marg, true_z1_marg))

    # Naive marginal
    th0_naive, se0_naive = diff_in_means(Y, T, Z_hat == 0)
    th1_naive, se1_naive = diff_in_means(Y, T, Z_hat == 1)
    rows.append(make_row('naive', 'marginal', th0_naive, th1_naive, se0_naive, se1_naive, true_z0_marg, true_z1_marg))

    # Global-corrected marginal
    tau_gc_m, se_gc_m = invert_mixing_safe(M_global, np.array([th0_naive, th1_naive]), np.array([se0_naive, se1_naive]))
    rows.append(make_row('global_corrected', 'marginal', tau_gc_m[0], tau_gc_m[1], se_gc_m[0], se_gc_m[1], true_z0_marg, true_z1_marg))

    # Stratified MLE marginal: weighted average of subgroup-corrected
    tau_strat_m = sum(w[s] * sub_corrected_strat[s] for s in SUBGROUP_LABELS)
    se_strat_m = np.sqrt(sum((w[s] * sub_se_strat[s]) ** 2 for s in SUBGROUP_LABELS))
    rows.append(make_row('stratified_mle', 'marginal', tau_strat_m[0], tau_strat_m[1], se_strat_m[0], se_strat_m[1], true_z0_marg, true_z1_marg))

    # HB EC-HTE marginal: weighted average of subgroup-corrected (HB)
    tau_hb_m = sum(w[s] * sub_corrected_hb[s] for s in SUBGROUP_LABELS)
    se_hb_m = np.sqrt(sum((w[s] * sub_se_hb[s]) ** 2 for s in SUBGROUP_LABELS))
    rows.append(make_row('hb_ec_hte', 'marginal', tau_hb_m[0], tau_hb_m[1], se_hb_m[0], se_hb_m[1], true_z0_marg, true_z1_marg))

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
            row[f'rmse_{z}'] = np.sqrt((valid ** 2).mean())
            covered = ((grp[f'ci_lower_{z}'] <= grp[f'tau_true_{z}']) &
                       (grp[f'tau_true_{z}'] <= grp[f'ci_upper_{z}'])).mean()
            row[f'coverage_{z}'] = covered
            row[f'bias_{z}_se'] = valid.std() / np.sqrt(len(valid))
        rows.append(row)
    return pd.DataFrame(rows)


def write_dgp_spec(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths):
    """Write DGP specification document."""
    n_total = len(pool_labels)
    n_neg = (pool_labels == 0).sum()
    n_neu = (pool_labels == 1).sum()
    n_pos = (pool_labels == 2).sum()
    pct_hashtag = pool_has_hashtag.mean() * 100
    median_len = np.median(pool_text_lengths)

    spec = f"""# exp-003 Phase A: DGP Specification

## Data Source
- **Dataset**: TweetEval sentiment (cardiffnlp/tweet_eval, sentiment split)
- **Reference**: Barbieri et al. (2020)
- **Pool size**: {n_total} tweets (train + test merged)
- **Label distribution**: negative={n_neg} ({n_neg/n_total*100:.1f}%), neutral={n_neu} ({n_neu/n_total*100:.1f}%), positive={n_pos} ({n_pos/n_total*100:.1f}%)

## Sampling Strategy
- Each MC run samples N={N} tweets with approximate label balance (~{N//3} per class)
- Sampling with replacement from pool (pool size >> N, minimal collision)

## Subgroup Definition (K=4)

Two binary features define 4 subgroups:

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `has_hashtag` | Text contains '#' | Hashtags shift LLM attention; informal/emphatic language may increase misclassification |
| `text_length_bin` | Above/below median length ({median_len:.0f} chars) | Longer texts provide more context but also more noise for sentiment classification |

| Subgroup | has_hashtag | text_length_bin | Approx. proportion |
|----------|-------------|-----------------|---------------------|
| S0 | 0 (no) | 0 (short) | ~{(1-pool_has_hashtag.mean())*(pool_text_lengths<=median_len).mean()*100:.0f}% |
| S1 | 0 (no) | 1 (long) | ~{(1-pool_has_hashtag.mean())*(pool_text_lengths>median_len).mean()*100:.0f}% |
| S2 | 1 (yes) | 0 (short) | ~{pool_has_hashtag.mean()*(pool_text_lengths<=median_len).mean()*100:.0f}% |
| S3 | 1 (yes) | 1 (long) | ~{pool_has_hashtag.mean()*(pool_text_lengths>median_len).mean()*100:.0f}% |

Prevalence of hashtag: {pct_hashtag:.1f}%

## Effect Modifier Z
- Z=1 if sentiment == positive (label 2)
- Z=0 if sentiment != positive (label 0 or 1)
- P(Z=1) ≈ {n_pos/n_total*100:.1f}%

## Misclassification Rates (Symmetric, A2 Assumption)

P(Z_hat ≠ Z | S=s) = π_s

| Regime | S0 | S1 | S2 | S3 | Rationale |
|--------|----|----|----|----|-----------|
| extreme | 0.05 | 0.10 | 0.15 | 0.25 | Large heterogeneity; hashtag texts are harder |
| moderate | 0.08 | 0.10 | 0.12 | 0.15 | Modest gradient |
| homogeneous | 0.10 | 0.10 | 0.10 | 0.10 | Null hypothesis (no heterogeneity) |

Rates are informed by typical LLM sentiment accuracy on tweets (85-95% in literature),
with the gradient reflecting empirical findings that informal/hashtag-heavy text
degrades classifier performance.

## Outcome Model

Y = α + τ(Z, S) · T + β₁ · text_length_normalized + β₂ · has_mention + ε

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| α | {ALPHA} | Intercept |
| τ(Z=0, all S) | {TAU_Z0} | Baseline treatment effect |
| δ_S0 | {DELTA_S['S0']} | Additional effect for Z=1 in S0 |
| δ_S1 | {DELTA_S['S1']} | Additional effect for Z=1 in S1 |
| δ_S2 | {DELTA_S['S2']} | Additional effect for Z=1 in S2 |
| δ_S3 | {DELTA_S['S3']} | Additional effect for Z=1 in S3 |
| β₁ | {BETA_LENGTH} | Text length effect (normalized) |
| β₂ | {BETA_MENTION} | Mention (@) effect |
| ε | N(0, 1) | Noise |
| T | Bernoulli(0.5) | RCT |

τ(Z=1, S=s) = {TAU_Z0} + δ_s, so treatment effects range from {TAU_Z0 + DELTA_S['S3']} (S3) to {TAU_Z0 + DELTA_S['S0']} (S0).

## Expert Annotation
- n_expert ∈ {{{', '.join(str(x) for x in N_EXPERTS)}}}
- Randomly selected observations with both Z and Z_hat observed

## Difference from Pure Synthetic DGP (MVP/exp-001/exp-002)
1. **Real text features**: Subgroups come from actual tweet characteristics, not random assignment
2. **Correlated subgroup sizes**: Subgroup proportions are data-driven and unbalanced
3. **Real covariates in outcome model**: text_length and has_mention are real text features
4. **Non-uniform Z prevalence**: P(Z=1) ≈ {n_pos/n_total*100:.1f}%, not 50%
5. **Subgroup-covariate correlation**: Text length correlates with hashtag presence and sentiment
"""
    with open('results/exp003_dgp_spec.md', 'w') as f:
        f.write(spec)
    print("Wrote results/exp003_dgp_spec.md")


# === Main ===

if __name__ == '__main__':
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else N_MC

    print(f"exp-003 Phase A: Semi-synthetic benchmark on TweetEval sentiment")
    print(f"N={N}, N_MC={n_mc}, N_EXPERTS={N_EXPERTS}")
    print(f"Regimes: {list(MISCLASS_RATES.keys())}")
    print(f"Methods: oracle, naive, global_corrected, stratified_mle, hb_ec_hte")
    print()

    # Load pool once
    print("Loading TweetEval sentiment dataset...")
    t_load = time.time()
    pool_texts, pool_labels = load_tweeteval_pool()
    pool_has_hashtag, pool_text_lengths, pool_has_mention = extract_features(pool_texts)
    print(f"  Pool size: {len(pool_texts)}, loaded in {time.time()-t_load:.1f}s")
    print(f"  Labels: neg={sum(pool_labels==0)}, neu={sum(pool_labels==1)}, pos={sum(pool_labels==2)}")
    print(f"  Hashtag prevalence: {pool_has_hashtag.mean()*100:.1f}%")
    print(f"  Median text length: {np.median(pool_text_lengths):.0f}")
    print()

    # Write DGP spec
    write_dgp_spec(pool_texts, pool_labels, pool_has_hashtag, pool_text_lengths)

    # Run MC simulation
    results = []
    configs = []
    for regime_name in MISCLASS_RATES:
        for ne in N_EXPERTS:
            configs.append((regime_name, ne))

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
                print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | {regime_name} ne={ne}")

    elapsed = time.time() - t0
    print(f"\nTotal MC time: {elapsed:.1f}s")

    df_raw = pd.DataFrame(results)
    df_agg = aggregate_results(df_raw)

    # Save
    raw_path = 'results/exp003_phase_a_v3_raw.csv'
    agg_path = 'results/exp003_phase_a_v3.csv'
    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved: {raw_path} ({len(df_raw)} rows)")
    print(f"Saved: {agg_path} ({len(df_agg)} rows)")

    # Print summary
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 260)
    pd.set_option('display.float_format', '{:.4f}'.format)

    for regime in MISCLASS_RATES:
        print(f"\n{'='*180}")
        pi = MISCLASS_RATES[regime]
        print(f"Regime: {regime} | misclass: S0={pi['S0']}, S1={pi['S1']}, S2={pi['S2']}, S3={pi['S3']}")
        print(f"{'='*180}")
        for level in ['marginal'] + SUBGROUP_LABELS:
            sub = df_agg[(df_agg['regime'] == regime) & (df_agg['level'] == level)]
            if sub.empty:
                continue
            sub = sub.sort_values(['n_expert', 'method'])
            cols = ['n_expert', 'method', 'level', 'bias_z0', 'rmse_z0', 'coverage_z0', 'bias_z1', 'rmse_z1', 'coverage_z1']
            print(sub[cols].to_string(index=False))
            print()
