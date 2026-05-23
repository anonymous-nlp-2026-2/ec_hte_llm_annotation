#!/usr/bin/env python3
"""exp032_confounded_sweep.py — Confounded DGP sweep over beta and propensity type

Extends exp031 (confounded DGP) with two new dimensions:
  1. beta sweep: beta in {0.25, 0.5, 1.0, 2.0} controlling confounding strength
  2. Oracle vs Estimated propensity: oracle uses true propensity e(X)=sigmoid(beta*X);
     estimated uses logistic regression fit on (X, T) training data

Produces a 2x4 = 8 configuration grid, each with N_MC=100 replications.

DGP:
  - N=5000, K=2, N_EXPERT=250, PI=[0.05, 0.25], TAU_Z={0: 0.0, 1: 0.3}
  - X ~ N(0,1) confounds T and Y
  - T ~ Bernoulli(sigmoid(beta * X))
  - Y = 1 + 0.5*Z + 0.3*X + tau(Z)*T + eps,  eps ~ N(0,1)
  - Z_hat has subgroup-specific misclassification at rates PI[s]
  - Expert labels: 250 randomly sampled

Methods: Oracle (true Z + IPW), Naive (Z_hat + IPW), Global, Strat MLE, HB EC-HTE
  All methods use IPW-DIM with propensity source determined by propensity_type.

Output:
  results/exp_confounded_sweep.csv         (raw per-MC rows)
  results/exp_confounded_sweep_summary.csv (grouped by beta, propensity_type, method)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ---------- Constants ----------
N = 5000
K = 2
N_EXPERT = 250
PI = [0.05, 0.25]
TAU_Z = {0: 0.0, 1: 0.3}
SUB_WEIGHTS = [0.5, 0.5]
N_MC = 100

BETAS = [0.25, 0.5, 1.0, 2.0]
PROPENSITY_TYPES = ['oracle', 'estimated']


# ---------- DGP ----------
def generate_data(seed, beta):
    """Generate confounded DGP data with given confounding strength beta.

    Returns:
        Y, T, Z_true, Z_hat, S, expert_mask, X_conf, true_propensity
    """
    rng = np.random.RandomState(seed)
    S = rng.choice(K, N, p=SUB_WEIGHTS)
    Z_true = rng.binomial(1, 0.5, N)

    # Covariate that confounds T and Y
    X_conf = rng.normal(0, 1, N)

    # Confounded treatment: depends on X (not Z, to avoid collider)
    true_propensity = expit(beta * X_conf)
    T = rng.binomial(1, true_propensity)

    # Subgroup-specific misclassification
    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, PI[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    tau = np.array([TAU_Z[z] for z in Z_true])
    baseline = 1.0 + 0.5 * Z_true + 0.3 * X_conf  # X affects Y
    Y = baseline + tau * T + rng.normal(0, 1, N)

    expert_idx = rng.choice(N, N_EXPERT, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask, X_conf, true_propensity


def estimate_propensity(X_conf, T):
    """Fit logistic regression on (X, T) to estimate propensity scores.

    Returns clipped estimated propensity P(T=1|X).
    """
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(X_conf.reshape(-1, 1), T)
    p_hat = lr.predict_proba(X_conf.reshape(-1, 1))[:, 1]
    return p_hat


# ---------- Confusion matrix utilities ----------
def compute_confusion_matrix(z_true, z_hat):
    """Compute 2x2 confusion matrix with Laplace smoothing (+1)."""
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mz = z_true == z
        C[z, 0] = (z_hat[mz] == 0).sum() + 1
        C[z, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


def hb_confusion_matrices(z_true_list, z_hat_list, alpha_prior=3.0):
    """Hierarchical Bayes shrinkage of subgroup confusion matrices toward global."""
    raw_cms = []
    ns = []
    for i in range(len(z_true_list)):
        C = compute_confusion_matrix(z_true_list[i], z_hat_list[i])
        raw_cms.append(C)
        ns.append(len(z_true_list[i]))

    global_z_true = np.concatenate(z_true_list)
    global_z_hat = np.concatenate(z_hat_list)
    C_global = compute_confusion_matrix(global_z_true, global_z_hat)

    shrunk = []
    for i in range(len(z_true_list)):
        w = ns[i] / (ns[i] + alpha_prior)
        C_s = w * raw_cms[i] + (1 - w) * C_global
        C_s = C_s / C_s.sum(axis=1, keepdims=True)
        shrunk.append(C_s)
    return shrunk


def invert_cm(C, tau_obs_vec, se_obs_vec, cond_threshold=100):
    """Invert confusion matrix to correct observed estimates for misclassification."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        col_sum = C[0, zh] + C[1, zh]
        if col_sum > 0:
            M[zh, 0] = C[0, zh] / col_sum
            M[zh, 1] = C[1, zh] / col_sum
        else:
            M[zh, :] = 0.5
    cond = np.linalg.cond(M)
    if cond > cond_threshold or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs_vec.copy(), se_obs_vec.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs_vec
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs_vec ** 2) @ Mi.T), 0))
    return tau_c, se_c


# ---------- IPW-DIM estimator ----------
def ipw_dim_estimate(Y, T, Z_labels, S, s, propensity):
    """IPW-adjusted DIM estimator with given propensity scores.

    Args:
        Y: outcome vector
        T: treatment vector
        Z_labels: subtype labels (true or hat)
        S: subgroup assignments
        s: subgroup index to estimate for
        propensity: propensity scores P(T=1|X), already computed (oracle or estimated)

    Returns:
        dict mapping z_level -> (tau_hat, se)
    """
    mask_s = S == s
    results = {}
    for z in [0, 1]:
        mask_sz = mask_s & (Z_labels == z)
        idx_t1 = np.where(mask_sz & (T == 1))[0]
        idx_t0 = np.where(mask_sz & (T == 0))[0]
        n1 = len(idx_t1)
        n0 = len(idx_t0)
        if n1 < 2 or n0 < 2:
            results[z] = (np.nan, np.nan)
            continue

        # Horvitz-Thompson estimator
        w1 = 1.0 / np.clip(propensity[idx_t1], 0.05, 0.95)
        w0 = 1.0 / np.clip(1.0 - propensity[idx_t0], 0.05, 0.95)

        y1_ipw = np.sum(Y[idx_t1] * w1) / np.sum(w1)
        y0_ipw = np.sum(Y[idx_t0] * w0) / np.sum(w0)
        tau_hat = y1_ipw - y0_ipw

        # Linearized variance
        resid1 = w1 * (Y[idx_t1] - y1_ipw)
        resid0 = w0 * (Y[idx_t0] - y0_ipw)
        var1 = np.sum(resid1**2) / (np.sum(w1))**2
        var0 = np.sum(resid0**2) / (np.sum(w0))**2
        se = np.sqrt(var1 + var0)

        results[z] = (tau_hat, se)
    return results


# ---------- Single MC iteration ----------
def run_one_mc(seed, beta, propensity_type):
    """Run one Monte Carlo replication for a given (beta, propensity_type) config.

    Args:
        seed: random seed for reproducibility
        beta: confounding strength
        propensity_type: 'oracle' (true propensity) or 'estimated' (logistic regression)

    Returns:
        list of result dicts
    """
    Y, T, Z_true, Z_hat, S, expert_mask, X_conf, true_propensity = generate_data(seed, beta)

    # Determine propensity scores to use
    if propensity_type == 'oracle':
        propensity = true_propensity
    else:
        propensity = estimate_propensity(X_conf, T)

    # Expert-based confusion matrices
    z_true_expert = [Z_true[expert_mask & (S == s)] for s in range(K)]
    z_hat_expert = [Z_hat[expert_mask & (S == s)] for s in range(K)]

    cm_mle = [compute_confusion_matrix(z_true_expert[s], z_hat_expert[s]) for s in range(K)]
    cm_hb = hb_confusion_matrices(z_true_expert, z_hat_expert)
    cm_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])

    rows = []
    for s in range(K):
        # Oracle: IPW with true Z
        oracle = ipw_dim_estimate(Y, T, Z_true, S, s, propensity)

        # IPW with Z_hat (naive baseline for EC-HTE)
        dim_zhat = ipw_dim_estimate(Y, T, Z_hat, S, s, propensity)

        tau_obs = np.array([dim_zhat[0][0], dim_zhat[1][0]])
        se_obs = np.array([dim_zhat[0][1], dim_zhat[1][1]])

        if np.any(np.isnan(tau_obs)) or np.any(np.isnan(se_obs)):
            continue

        o0, o1 = oracle[0], oracle[1]
        if np.isnan(o0[0]) or np.isnan(o1[0]):
            continue
        tau_oracle = np.array([o0[0], o1[0]])
        se_oracle = np.array([o0[1], o1[1]])

        # EC-HTE corrections
        tau_mle, se_mle = invert_cm(cm_mle[s], tau_obs, se_obs)
        tau_hb, se_hb = invert_cm(cm_hb[s], tau_obs, se_obs)
        tau_gl, se_gl = invert_cm(cm_global, tau_obs, se_obs)

        methods = [
            ('Oracle (true Z)', tau_oracle, se_oracle),
            ('Naive', tau_obs, se_obs),
            ('Global', tau_gl, se_gl),
            ('Strat MLE', tau_mle, se_mle),
            ('HB EC-HTE', tau_hb, se_hb),
        ]

        for method_name, tau_hat_vec, se_hat_vec in methods:
            for zi, z in enumerate([0, 1]):
                th = tau_hat_vec[zi]
                se = se_hat_vec[zi]
                ci_lo = th - 1.96 * se
                ci_hi = th + 1.96 * se
                true_tau = TAU_Z[z]
                covers = int(ci_lo <= true_tau <= ci_hi)
                rows.append({
                    'beta': beta,
                    'propensity_type': propensity_type,
                    'mc_seed': seed,
                    'subgroup': s,
                    'z_level': z,
                    'method': method_name,
                    'tau_hat': th,
                    'tau_true': true_tau,
                    'bias': th - true_tau,
                    'se': se,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'covers': covers,
                })
    return rows


# ---------- Main ----------
def main():
    t0 = time.time()
    os.makedirs('results', exist_ok=True)

    print(f"Confounded DGP Sweep: K={K}, N={N}, n_expert={N_EXPERT}")
    print(f"Betas={BETAS}, Propensity types={PROPENSITY_TYPES}")
    print(f"PI={PI}, TAU_Z={TAU_Z}, N_MC={N_MC}")
    print(f"Grid: {len(BETAS)} x {len(PROPENSITY_TYPES)} = {len(BETAS)*len(PROPENSITY_TYPES)} configs")

    # Show propensity distributions for each beta
    rng = np.random.RandomState(42)
    X_test = rng.normal(0, 1, N)
    for beta in BETAS:
        prop_test = expit(beta * X_test)
        print(f"  beta={beta:.2f}: propensity mean={prop_test.mean():.3f}, "
              f"std={prop_test.std():.3f}, range=[{prop_test.min():.3f}, {prop_test.max():.3f}]")

    all_rows = []
    total_configs = len(BETAS) * len(PROPENSITY_TYPES)
    config_idx = 0

    for beta in BETAS:
        for prop_type in PROPENSITY_TYPES:
            config_idx += 1
            print(f"\n--- Config {config_idx}/{total_configs}: "
                  f"beta={beta}, propensity={prop_type} ---")

            for mc in range(N_MC):
                if mc % 20 == 0:
                    print(f"  MC {mc}/{N_MC} ...", flush=True)
                rows = run_one_mc(42 + mc, beta, prop_type)
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/exp_confounded_sweep.csv', index=False)
    print(f"\nWrote {len(df)} rows to results/exp_confounded_sweep.csv")

    # Summary grouped by (beta, propensity_type, method)
    summary = df.groupby(['beta', 'propensity_type', 'method']).agg(
        mean_abs_bias=('bias', lambda x: x.abs().mean()),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        coverage=('covers', 'mean'),
        mean_se=('se', 'mean'),
    ).reset_index()

    summary.to_csv('results/exp_confounded_sweep_summary.csv', index=False)
    print(f"Wrote summary to results/exp_confounded_sweep_summary.csv")

    # Print summary table
    print("\n" + "=" * 80)
    print("CONFOUNDED DGP SWEEP — IPW-DIM")
    print("=" * 80)
    print(f"{'Beta':>6} {'Propensity':>12} {'Method':>18} "
          f"{'|Bias|':>8} {'RMSE':>8} {'Cov':>6}")
    print("-" * 64)

    for beta in BETAS:
        for prop_type in PROPENSITY_TYPES:
            mask = (summary['beta'] == beta) & (summary['propensity_type'] == prop_type)
            sub = summary[mask].sort_values('method')
            for _, row in sub.iterrows():
                print(f"{row['beta']:6.2f} {row['propensity_type']:>12} "
                      f"{row['method']:>18} {row['mean_abs_bias']:8.4f} "
                      f"{row['rmse']:8.4f} {row['coverage']:6.3f}")
            print("-" * 64)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
