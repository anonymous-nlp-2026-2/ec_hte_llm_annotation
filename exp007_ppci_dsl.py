"""
exp-007: PPCI + Stratified MLE Baseline Comparison for EC-HTE
Compares measurement-model-level correction (EC-HTE) vs estimand-level correction (Stratified MLE)
vs standard PPCI approach. DGP uses pure covariate-error: only Z has LLM misclassification.
Outcome Y can optionally have measurement error (sigma_outcome > 0).

Fix (2026-05-20): method_ec_hte_ppci double-correction bug.
  Rectifier was using Z_hat grouping on gold subset, capturing both covariate and outcome error.
  Since M^{-1} already corrects covariate error, this caused double correction under DGP A/C.
  Fixed: rectifier now uses true Z grouping on gold subset so it only captures outcome error.

HB backfill (2026-05-21): replaced MLE placeholder in ec_hte/ec_hte_ppci with real HB Gibbs
  sampler (Dirichlet-Multinomial, 4 chains, 1000 iter, 500 warmup). Ported from exp-003.
  Added invert_mixing_safe with cond(M) > 100 singularity guard.
"""

import numpy as np
import pandas as pd
import sys
import time
import warnings
from scipy.special import gammaln
warnings.filterwarnings('ignore')

# === DGP ===

REGIMES = {
    'extreme':     {'tau_z1': 0.5, 'tau_z0': 0.1},
    'moderate':    {'tau_z1': 0.3, 'tau_z0': 0.15},
    'homogeneous': {'tau_z1': 0.25, 'tau_z0': 0.20},
}

MISCLASS_K2 = {
    'extreme':     [0.25, 0.05],
    'moderate':    [0.15, 0.10],
    'homogeneous': [0.12, 0.12],
}

MISCLASS_K4 = {
    'extreme':     [0.30, 0.20, 0.10, 0.03],
    'moderate':    [0.20, 0.15, 0.10, 0.05],
    'homogeneous': [0.12, 0.12, 0.12, 0.12],
}

OUTCOME_BIAS = {
    'extreme':     [0.3, 0.1, 0.5, 0.2],
    'moderate':    [0.15, 0.05, 0.25, 0.10],
    'homogeneous': [0.1, 0.1, 0.1, 0.1],
}

DGP_CONFIGS = {
    'A_covariate_only':  {'has_covariate_error': True,  'has_outcome_error': False},
    'B_outcome_only':    {'has_covariate_error': False, 'has_outcome_error': True},
    'C_both_errors':     {'has_covariate_error': True,  'has_outcome_error': True},
    'D_no_error':        {'has_covariate_error': False, 'has_outcome_error': False},
}

N = 5000
N_EXPERTS = [50, 100, 250, 500]
N_MC = 100
OUTCOME_NOISE_LEVELS = [0.0, 0.5, 1.0]


def generate_data(regime_params, misclass_rates, n_expert, k_subgroups, seed,
                  sigma_outcome=0.0, outcome_noise_heterogeneous=True,
                  has_covariate_error=True, has_outcome_error=False,
                  outcome_bias_per_subgroup=None):
    """Generate synthetic data with configurable covariate and outcome error.

    DGP A (covariate only): Z_hat has misclassification, Y = Y_true
    DGP B (outcome only): Z_hat = Z, Y has treatment-differential bias
    DGP C (both): Z_hat has misclassification AND Y has bias
    DGP D (no error): Z_hat = Z, Y = Y_true
    """
    rng = np.random.RandomState(seed)
    k = k_subgroups

    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(k, N, p=[1/k]*k)
    X = rng.randn(N, 5)

    tau = np.where(Z == 1, regime_params['tau_z1'], regime_params['tau_z0'])
    eps = rng.randn(N)
    Y_true = 1 + tau * T + 0.5 * X[:, 0] + eps
    Y = Y_true.copy()

    if has_outcome_error:
        if outcome_bias_per_subgroup is not None:
            for i in range(k):
                s_mask = (S == i)
                bias_i = outcome_bias_per_subgroup[i] if i < len(outcome_bias_per_subgroup) else outcome_bias_per_subgroup[-1]
                Y[s_mask] += bias_i * T[s_mask]
        if sigma_outcome > 0:
            for i in range(k):
                s_mask = (S == i)
                noise_scale = sigma_outcome * (1 + 0.5 * i / max(k - 1, 1))
                Y[s_mask] += rng.normal(0, noise_scale, s_mask.sum())

    if has_covariate_error:
        mc_rates = np.array([misclass_rates[s] for s in S])
        flip = rng.rand(N) < mc_rates
        Z_hat = np.where(flip, 1 - Z, Z)
    else:
        Z_hat = Z.copy()

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'X': X, 'Y': Y,
        'Y_true': Y_true, 'tau': tau, 'expert_mask': expert_mask,
    }


# === Confusion Matrix Estimators ===

def estimate_cm_mle(z_true, z_hat):
    """MLE confusion matrix with Laplace smoothing. C[z, z_hat] = P(Z_hat=z_hat | Z=z)."""
    C = np.zeros((2, 2))
    for zt in [0, 1]:
        mask = z_true == zt
        for zh in [0, 1]:
            C[zt, zh] = ((z_hat[mask] == zh).sum() + 1) / (mask.sum() + 2)
    return C


def get_counts(z_true, z_hat, subgroups, labels):
    """Build count matrix counts[subgroup, true_z, predicted_z_hat]."""
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
    """Log posterior for alpha_0[z] in log-alpha space (includes Jacobian)."""
    if alpha < 1e-10:
        return -1e10
    a = alpha / 2.0
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - 2.0 * gammaln(a))
    lp += (a - 1.0) * np.sum(np.log(np.clip(theta_z, 1e-300, None)))
    return lp


def _split_rhat(chains):
    """Split R-hat for a scalar parameter. chains: (n_chains, n_draws)."""
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
    """Approximate bulk ESS via initial positive sequence. chains: (n_chains, n_draws)."""
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
    """Compute R-hat and ESS over all parameters without arviz."""
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
    """Gibbs sampler for HB Dirichlet-Multinomial (conjugate + adaptive MH for alpha_0).

    Partial pooling across subgroups via shared Dirichlet concentration parameter.
    hp_a0, hp_b0: Gamma hyperprior shape/rate on alpha_0 (default Gamma(3,1)).
    """
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


# === Correction Methods ===

def diff_in_means(Y, T, mask):
    """Difference-in-means estimator for subpopulation defined by mask."""
    t1 = mask & (T == 1)
    t0 = mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    tau_hat = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
    return tau_hat, se


def build_mixing_matrix(C, p_z, p_z_hat):
    """M[z_hat, z] = P(Z=z | Z_hat=z_hat) via Bayes' rule."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        for z in [0, 1]:
            M[zh, z] = C[z, zh] * p_z[z] / p_z_hat[zh]
    return M


def invert_mixing(M, tau_obs, se_obs):
    """Apply M^{-1} correction. Returns (tau_corrected, se_corrected)."""
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    M_inv = np.linalg.inv(M)
    tau_corrected = M_inv @ tau_obs
    sigma_obs = np.diag(se_obs ** 2)
    sigma_corrected = M_inv @ sigma_obs @ M_inv.T
    se_corrected = np.sqrt(np.diag(sigma_corrected))
    return tau_corrected, se_corrected


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    """Like invert_mixing but with condition number guard. cond(M) > threshold → fallback to uncorrected."""
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


def method_oracle(data, s_val):
    """Oracle: use true Z within subgroup s."""
    Y, T, Z, S = data['Y'], data['T'], data['Z'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z == 1))
    return th0, th1, se0, se1


def method_naive(data, s_val):
    """Naive: use Z_hat within subgroup s, no correction."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val
    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    return th0, th1, se0, se1


def method_global_corrected(data, s_val, C_global, p_z_global, p_zh_global):
    """Global-corrected: use global M_global^{-1} on subgroup-level observed CATEs."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val

    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan

    M = build_mixing_matrix(C_global, p_z_global, p_zh_global)
    tau_corr, se_corr = invert_mixing(M, np.array([th0, th1]), np.array([se0, se1]))
    return tau_corr[0], tau_corr[1], se_corr[0], se_corr[1]


def method_stratified_mle(data, s_val, C_s, p_z_s, p_zh_s):
    """Stratified MLE: per-subgroup independent correction.

    Each subgroup estimates its own confusion matrix independently (no pooling).
    This is estimand-level correction -- correct the CATE estimate directly
    using the subgroup's own confusion matrix.

    Key difference from EC-HTE: no cross-subgroup shrinkage.
    When n_expert is small per subgroup, Stratified MLE has high variance.
    """
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val

    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan

    M_s = build_mixing_matrix(C_s, p_z_s, p_zh_s)
    tau_corr, se_corr = invert_mixing(M_s, np.array([th0, th1]), np.array([se0, se1]))
    return tau_corr[0], tau_corr[1], se_corr[0], se_corr[1]


def method_ec_hte(data, s_val, C_s_hb, p_z_s, p_zh_s):
    """EC-HTE: HB Gibbs sampler confusion matrix + subgroup correction with singularity guard."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    s_mask = S == s_val

    th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0) or np.isnan(th1):
        return np.nan, np.nan, np.nan, np.nan

    M_s = build_mixing_matrix(C_s_hb, p_z_s, p_zh_s)
    tau_corr, se_corr = invert_mixing_safe(M_s, np.array([th0, th1]), np.array([se0, se1]))
    return tau_corr[0], tau_corr[1], se_corr[0], se_corr[1]


def method_mi_rubin(data, s_val, M=20, seed=0):
    """Multiple Imputation with Rubin's rules (formerly called PPCI).

    For each non-expert observation, impute Z from P(Z|Z_hat, S) estimated
    on expert data. Repeat M times, compute CATE on each imputed dataset,
    then combine estimates using Rubin's rules.
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    rng = np.random.RandomState(seed)

    s_mask = S == s_val
    expert_s = expert_mask & s_mask

    if expert_s.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan

    p_z1_given_zhat = {}
    for z_hat_val in [0, 1]:
        expert_zhat = expert_s & (Z_hat == z_hat_val)
        if expert_zhat.sum() > 0:
            p_z1_given_zhat[z_hat_val] = np.clip(Z[expert_zhat].mean(), 0.01, 0.99)
        else:
            p_z1_given_zhat[z_hat_val] = float(z_hat_val)

    Y_s = Y[s_mask]
    T_s = T[s_mask]
    Z_hat_s = Z_hat[s_mask]
    is_expert_s = expert_mask[s_mask]
    Z_true_s = Z[s_mask]
    non_expert_s = ~is_expert_s

    tau_z0_list, tau_z1_list = [], []
    var_z0_list, var_z1_list = [], []

    for m_iter in range(M):
        Z_imp = Z_hat_s.copy().astype(float)
        Z_imp[is_expert_s] = Z_true_s[is_expert_s]
        if non_expert_s.any():
            probs = np.array([p_z1_given_zhat[zh] for zh in Z_hat_s[non_expert_s]])
            Z_imp[non_expert_s] = rng.binomial(1, probs).astype(float)

        for z_val, tau_list, var_list in [(0, tau_z0_list, var_z0_list),
                                          (1, tau_z1_list, var_z1_list)]:
            z_mask = Z_imp == z_val
            t1 = z_mask & (T_s == 1)
            t0 = z_mask & (T_s == 0)
            n1, n0 = t1.sum(), t0.sum()
            if n1 < 2 or n0 < 2:
                tau_list.append(np.nan)
                var_list.append(np.nan)
            else:
                y1, y0 = Y_s[t1], Y_s[t0]
                tau_hat = y1.mean() - y0.mean()
                v = y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0
                tau_list.append(tau_hat)
                var_list.append(v)

    def rubins_combine(tau_list, var_list):
        valid = [(t, v) for t, v in zip(tau_list, var_list)
                 if not np.isnan(t) and not np.isnan(v)]
        if len(valid) < 2:
            return np.nan, np.nan
        taus = np.array([t for t, v in valid])
        vars_ = np.array([v for t, v in valid])
        m_eff = len(valid)
        tau_bar = taus.mean()
        W = vars_.mean()
        B = taus.var(ddof=1)
        V_total = W + (1 + 1.0 / m_eff) * B
        return tau_bar, np.sqrt(max(V_total, 0))

    th0, se0 = rubins_combine(tau_z0_list, var_z0_list)
    th1, se1 = rubins_combine(tau_z1_list, var_z1_list)
    return th0, th1, se0, se1


def method_ppci(data, s_val):
    """PPI rectifier for CATE estimation.

    General-purpose correction using a gold-standard expert subset.
    The rectifier captures ALL systematic error (covariate + outcome) between
    the gold estimate (true Z, true Y) and the noisy estimate (Z_hat, observed Y).
    This is complementary to EC-HTE's structured covariate-only correction via M^{-1}.

    For each subgroup s and Z-level z:
      1. tau_hat_all(z,s) = CATE from ALL data using Z_hat and observed Y
      2. tau_gold(z,s) = CATE from GOLD subset using true Z and true Y
      3. tau_hat_gold(z,s) = CATE from GOLD subset using Z_hat and observed Y
      4. rectifier = tau_gold - tau_hat_gold  (captures covariate + outcome error)
      5. tau_ppci = tau_hat_all + rectifier
    """
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    Y_true = data['Y_true']
    expert_mask = data['expert_mask']

    s_mask = S == s_val
    expert_s = expert_mask & s_mask
    n_expert_s = expert_s.sum()

    if n_expert_s < 10:
        th0, se0 = diff_in_means(Y, T, s_mask & (Z_hat == 0))
        th1, se1 = diff_in_means(Y, T, s_mask & (Z_hat == 1))
        return th0, th1, se0, se1

    expert_s_indices = np.where(expert_s)[0]
    Z_true_s = Z[expert_s_indices]

    results_z = {}
    for z_val in [0, 1]:
        all_mask = s_mask & (Z_hat == z_val)
        tau_hat_all, se_hat_all = diff_in_means(Y, T, all_mask)

        gold_mask_z = (Z_true_s == z_val)
        gold_indices = expert_s_indices[gold_mask_z]

        if len(gold_indices) < 4:
            results_z[z_val] = (tau_hat_all, se_hat_all)
            continue

        # tau_gold: use true Z and true Y on gold subset
        gold_bool = np.zeros(len(Y), dtype=bool)
        gold_bool[gold_indices] = True
        tau_gold, se_gold = diff_in_means(Y_true, T, gold_bool)

        # tau_hat_gold: use Z_hat and observed Y on gold subset (same people)
        gold_zhat_mask = (Z_hat[expert_s_indices] == z_val)
        gold_zhat_indices = expert_s_indices[gold_zhat_mask]
        gold_zhat_bool = np.zeros(len(Y), dtype=bool)
        gold_zhat_bool[gold_zhat_indices] = True
        tau_hat_gold, se_hat_gold = diff_in_means(Y, T, gold_zhat_bool)

        if np.isnan(tau_gold) or np.isnan(tau_hat_gold) or np.isnan(tau_hat_all):
            results_z[z_val] = (tau_hat_all, se_hat_all)
            continue

        rectifier = tau_gold - tau_hat_gold
        tau_ppci = tau_hat_all + rectifier
        se_ppci = np.sqrt(se_hat_all**2 + se_gold**2 + se_hat_gold**2)
        results_z[z_val] = (tau_ppci, se_ppci)

    th0, se0 = results_z.get(0, (np.nan, np.nan))
    th1, se1 = results_z.get(1, (np.nan, np.nan))
    return th0, th1, se0, se1


def method_ec_hte_ppci(data, s_val, C_s, p_z_s, p_zh_s):
    """EC-HTE + PPCI: HB Gibbs M^{-1} correction (with singularity guard) + PPI rectifier for outcome error."""
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    Y_true = data['Y_true']
    expert_mask = data['expert_mask']

    s_mask = S == s_val
    expert_s = expert_mask & s_mask

    th0_obs, se0_obs = diff_in_means(Y, T, s_mask & (Z_hat == 0))
    th1_obs, se1_obs = diff_in_means(Y, T, s_mask & (Z_hat == 1))
    if np.isnan(th0_obs) or np.isnan(th1_obs):
        return np.nan, np.nan, np.nan, np.nan

    M_s = build_mixing_matrix(C_s, p_z_s, p_zh_s)
    tau_ec, se_ec = invert_mixing_safe(M_s, np.array([th0_obs, th1_obs]),
                                       np.array([se0_obs, se1_obs]))

    n_expert_s = expert_s.sum()
    if n_expert_s < 10:
        return tau_ec[0], tau_ec[1], se_ec[0], se_ec[1]

    expert_s_indices = np.where(expert_s)[0]
    Z_true_s = Z[expert_s_indices]

    results_z = {}
    for idx, z_val in enumerate([0, 1]):
        gold_z_mask = (Z_true_s == z_val)
        gold_indices = expert_s_indices[gold_z_mask]

        if len(gold_indices) < 4:
            results_z[z_val] = (tau_ec[idx], se_ec[idx])
            continue

        gold_bool = np.zeros(len(Y), dtype=bool)
        gold_bool[gold_indices] = True

        # tau_gold: true Z + true Y → no error
        tau_gold, se_gold = diff_in_means(Y_true, T, gold_bool)

        # tau_hat_gold_true_z: true Z + observed Y → only outcome error remains
        tau_hat_gold_true_z, se_hat_gold = diff_in_means(Y, T, gold_bool)

        if np.isnan(tau_gold) or np.isnan(tau_hat_gold_true_z):
            results_z[z_val] = (tau_ec[idx], se_ec[idx])
            continue

        # rectifier captures outcome error only (covariate error already handled by M^{-1})
        rectifier = tau_gold - tau_hat_gold_true_z
        tau_combined = tau_ec[idx] + rectifier
        se_combined = np.sqrt(se_ec[idx]**2 + se_gold**2 + se_hat_gold**2)
        results_z[z_val] = (tau_combined, se_combined)

    th0, se0 = results_z.get(0, (np.nan, np.nan))
    th1, se1 = results_z.get(1, (np.nan, np.nan))
    return th0, th1, se0, se1


# === Main Simulation Loop ===

def run_one_mc(regime_name, regime_params, misclass_rates, n_expert, k_subgroups, seed,
               sigma_outcome=0.0, dgp_name='A_covariate_only', dgp_cfg=None,
               outcome_bias_per_subgroup=None):
    """Run one MC iteration for all methods."""
    if dgp_cfg is None:
        dgp_cfg = DGP_CONFIGS[dgp_name]

    data = generate_data(regime_params, misclass_rates, n_expert, k_subgroups, seed,
                         sigma_outcome=sigma_outcome,
                         has_covariate_error=dgp_cfg['has_covariate_error'],
                         has_outcome_error=dgp_cfg['has_outcome_error'],
                         outcome_bias_per_subgroup=outcome_bias_per_subgroup)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    tau_true_z0, tau_true_z1 = regime_params['tau_z0'], regime_params['tau_z1']

    C_global = estimate_cm_mle(Z[expert_mask], Z_hat[expert_mask])
    p_z_global = np.clip(np.array([(Z[expert_mask]==0).mean(), (Z[expert_mask]==1).mean()]), 0.01, 0.99)
    p_zh_global = np.clip(np.array([(Z_hat==0).mean(), (Z_hat==1).mean()]), 0.01, 0.99)

    C_mle = {}
    C_hb = {}

    for s in range(k_subgroups):
        s_mask = S == s
        expert_s = expert_mask & s_mask
        if expert_s.sum() >= 4:
            C_mle[s] = estimate_cm_mle(Z[expert_s], Z_hat[expert_s])
        else:
            C_mle[s] = C_global

    # HB Gibbs sampler: Dirichlet-Multinomial partial pooling across subgroups
    subgroup_labels = list(range(k_subgroups))
    counts = get_counts(Z[expert_mask], Z_hat[expert_mask], S[expert_mask], subgroup_labels)
    C_hb_all, _hb_diag = estimate_cm_hb_gibbs(counts, subgroup_labels, seed=seed)
    for s in range(k_subgroups):
        s_mask = S == s
        expert_s = expert_mask & s_mask
        if expert_s.sum() >= 4:
            C_hb[s] = C_hb_all[s]
        else:
            C_hb[s] = C_global

    rows = []
    base = {
        'regime': regime_name, 'n_expert': n_expert, 'k_subgroups': k_subgroups,
        'sigma_outcome': sigma_outcome, 'dgp': dgp_name,
        'mc_run': seed, 'tau_true_z0': tau_true_z0, 'tau_true_z1': tau_true_z1,
    }

    def make_row(method, subgroup, th0, th1, se0, se1):
        return {**base, 'method': method, 'subgroup': subgroup,
                'tau_hat_z0': th0, 'tau_hat_z1': th1,
                'ci_lower_z0': th0 - 1.96*se0 if not np.isnan(se0) else np.nan,
                'ci_upper_z0': th0 + 1.96*se0 if not np.isnan(se0) else np.nan,
                'ci_lower_z1': th1 - 1.96*se1 if not np.isnan(se1) else np.nan,
                'ci_upper_z1': th1 + 1.96*se1 if not np.isnan(se1) else np.nan}

    for s in range(k_subgroups):
        s_mask = S == s
        expert_s = expert_mask & s_mask
        if expert_s.sum() >= 4:
            p_z_s = np.clip(np.array([(Z[expert_s]==0).mean(), (Z[expert_s]==1).mean()]), 0.01, 0.99)
        else:
            p_z_s = p_z_global
        p_zh_s = np.clip(np.array([(Z_hat[s_mask]==0).mean(), (Z_hat[s_mask]==1).mean()]), 0.01, 0.99)

        th0, th1, se0, se1 = method_oracle(data, s)
        rows.append(make_row('oracle', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_naive(data, s)
        rows.append(make_row('naive', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_global_corrected(data, s, C_global, p_z_global, p_zh_global)
        rows.append(make_row('global_corrected', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_stratified_mle(data, s, C_mle[s], p_z_s, p_zh_s)
        rows.append(make_row('stratified_mle', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_ec_hte(data, s, C_hb[s], p_z_s, p_zh_s)
        rows.append(make_row('ec_hte', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_mi_rubin(data, s, M=20, seed=seed * 1000 + s)
        rows.append(make_row('mi_rubin', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_ppci(data, s)
        rows.append(make_row('ppci', s, th0, th1, se0, se1))

        th0, th1, se0, se1 = method_ec_hte_ppci(data, s, C_hb[s], p_z_s, p_zh_s)
        rows.append(make_row('ec_hte_ppci', s, th0, th1, se0, se1))

    return rows


def aggregate_results(df):
    """Aggregate per-run results into summary statistics."""
    rows = []
    group_cols = ['regime', 'n_expert', 'k_subgroups', 'sigma_outcome', 'dgp', 'subgroup', 'method']
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


def run_simulation(n_mc, k_list=[2, 4], sigma_list=None, dgp_list=None, verbose=True):
    """Run full simulation sweep over DGP configs."""
    if sigma_list is None:
        sigma_list = OUTCOME_NOISE_LEVELS
    if dgp_list is None:
        dgp_list = list(DGP_CONFIGS.keys())
    results = []
    configs = []
    for regime_name in REGIMES:
        for k in k_list:
            misclass = MISCLASS_K2[regime_name] if k == 2 else MISCLASS_K4[regime_name]
            for ne in N_EXPERTS:
                for sigma in sigma_list:
                    for dgp_name in dgp_list:
                        configs.append((regime_name, k, misclass, ne, sigma, dgp_name))

    total = len(configs) * n_mc
    count = 0
    t0 = time.time()

    for regime_name, k, misclass, ne, sigma, dgp_name in configs:
        dgp_cfg = DGP_CONFIGS[dgp_name]
        outcome_bias = OUTCOME_BIAS[regime_name]
        if k == 2:
            outcome_bias_k = outcome_bias[:2]
        else:
            outcome_bias_k = outcome_bias[:k]

        for mc in range(n_mc):
            rows = run_one_mc(regime_name, REGIMES[regime_name], misclass, ne, k, seed=mc,
                              sigma_outcome=sigma, dgp_name=dgp_name, dgp_cfg=dgp_cfg,
                              outcome_bias_per_subgroup=outcome_bias_k)
            results.extend(rows)
            count += 1
            if verbose and count % max(1, total // 20) == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | {dgp_name} k={k} {regime_name} ne={ne} sigma={sigma} mc={mc}")

    return pd.DataFrame(results), time.time() - t0


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--dry-run']
    n_mc = int(args[0]) if args else N_MC

    if dry_run:
        n_mc = 1
        sigma_list = [0.5]
        k_list = [2]
        dgp_list = list(DGP_CONFIGS.keys())
    else:
        sigma_list = None
        k_list = [2, 4]
        dgp_list = None

    methods_str = 'oracle, naive, global_corrected, stratified_mle, ec_hte, mi_rubin, ppci, ec_hte_ppci'
    print(f"exp-007: PPCI + EC-HTE 4-DGP Matrix")
    print(f"N={N}, N_MC={n_mc}")
    print(f"Methods: {methods_str}")
    print(f"DGPs: {dgp_list if dgp_list else list(DGP_CONFIGS.keys())}")
    print(f"Outcome noise levels: {sigma_list if sigma_list else OUTCOME_NOISE_LEVELS}")
    print(f"HB: Gibbs sampler (4 chains, 1000 iter, 500 warmup, Dirichlet-Multinomial)")
    if dry_run:
        print(f"** DRY RUN: 1 MC, k=[2], sigma=[0.5], all 4 DGPs **")
    print()

    df_raw, elapsed = run_simulation(n_mc, k_list=k_list, sigma_list=sigma_list,
                                     dgp_list=dgp_list)
    df_agg = aggregate_results(df_raw)

    raw_path = 'results/exp007_ppci_dsl_hb_raw.csv'
    agg_path = 'results/exp007_ppci_dsl_hb.csv'
    df_raw.to_csv(raw_path, index=False)
    df_agg.to_csv(agg_path, index=False)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Saved: {raw_path}, {agg_path}")

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', '{:.4f}'.format)

    for dgp in (dgp_list or list(DGP_CONFIGS.keys())):
        for regime in REGIMES:
            print(f"\n{'='*140}")
            print(f"DGP: {dgp} | Regime: {regime}")
            print(f"{'='*140}")
            sub = df_agg[(df_agg['regime'] == regime) & (df_agg['dgp'] == dgp)]
            if sub.empty:
                continue
            cols = ['n_expert', 'k_subgroups', 'sigma_outcome', 'subgroup', 'method',
                    'bias_z0', 'rmse_z0', 'coverage_z0', 'bias_z1', 'rmse_z1', 'coverage_z1']
            print(sub[cols].to_string(index=False))
