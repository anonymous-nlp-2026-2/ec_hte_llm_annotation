#!/usr/bin/env python3
"""exp011_causal_forest.py - CausalForestDML EC-HTE integration (exp-011)

Validates EC-HTE correction with CausalForestDML as CATE estimator.
Three methods: naive_forest (Z_hat, no correction), global_forest (global M^-1),
ec_hte_forest (per-subgroup M_s^-1).

Output: results/exp011_causal_forest.csv
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

N = 5000
TAU_TRUE = {0: 0.5, 1: 1.5}

MISCLASS_RATES = {
    2: {'extreme': [0.05, 0.20], 'moderate': [0.10, 0.15]},
    4: {'extreme': [0.05, 0.15, 0.25, 0.40], 'moderate': [0.08, 0.12, 0.16, 0.20]},
}

CONFIGS = [
    {'K': 2, 'regime': 'extreme', 'n_expert': 100},
    {'K': 2, 'regime': 'extreme', 'n_expert': 500},
    {'K': 2, 'regime': 'moderate', 'n_expert': 100},
    {'K': 2, 'regime': 'moderate', 'n_expert': 500},
    {'K': 4, 'regime': 'extreme', 'n_expert': 100},
    {'K': 4, 'regime': 'extreme', 'n_expert': 500},
    {'K': 4, 'regime': 'moderate', 'n_expert': 100},
    {'K': 4, 'regime': 'moderate', 'n_expert': 500},
]


# -- DGP ------------------------------------------------------------------

def generate_data(N, K, regime, n_expert, seed):
    """Generate synthetic data with subgroup-specific misclassification.

    Returns Y, T, Z_true, Z_hat, S, expert_mask.
    """
    rng = np.random.RandomState(seed)
    S = rng.randint(0, K, N)
    T = rng.binomial(1, 0.5, N)
    Z_true = rng.binomial(1, 0.5, N)

    pi_s = MISCLASS_RATES[K][regime]
    Z_hat = Z_true.copy()
    for s in range(K):
        mask_s = S == s
        flip = rng.binomial(1, pi_s[s], mask_s.sum())
        Z_hat[mask_s] = np.where(flip, 1 - Z_true[mask_s], Z_true[mask_s])

    tau = np.where(Z_true == 1, TAU_TRUE[1], TAU_TRUE[0])
    baseline = 1.0 + 0.5 * Z_true
    Y = baseline + tau * T + rng.normal(0, 1, N)

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return Y, T, Z_true, Z_hat, S, expert_mask


# -- Helpers ---------------------------------------------------------------

def compute_confusion_matrix(z_true, z_hat):
    """2x2 confusion matrix C[z, z_hat] = P(Z_hat|Z) with Laplace smoothing."""
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mz = z_true == z
        C[z, 0] = (z_hat[mz] == 0).sum() + 1
        C[z, 1] = (z_hat[mz] == 1).sum() + 1
    return C / C.sum(axis=1, keepdims=True)


def build_mixing_matrix(C):
    """C[z, z_hat] -> M[z_hat, z] via Bayes rule (P(Z=0)=P(Z=1)=0.5)."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        if cs > 0:
            M[zh, 0] = C[0, zh] / cs
            M[zh, 1] = C[1, zh] / cs
        else:
            M[zh, :] = 0.5
    return M


def invert_mixing_safe(M, tau_obs, se_obs, cond_threshold=100):
    """M^-1 correction with condition number guard."""
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


# -- Single MC Run --------------------------------------------------------

def run_one_mc(config, seed):
    """Run one Monte Carlo iteration: fit forest, apply 3 correction methods."""
    K = config['K']
    Y, T, Z_true, Z_hat, S, expert_mask = generate_data(
        N, K, config['regime'], config['n_expert'], seed)

    X_hat = np.column_stack([Z_hat, S]).astype(float)
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200, min_samples_leaf=20, random_state=42)
    cf.fit(Y, T, X=X_hat, W=None)

    # Predict CATE at each unique (Z_hat, S) point
    X_eval = np.array([[z, s] for s in range(K) for z in [0, 1]], dtype=float)
    cate_vals = cf.effect(X_eval).flatten()

    try:
        inf = cf.effect_inference(X_eval)
        se_raw = inf.stderr
        if hasattr(se_raw, 'values'):
            se_raw = se_raw.values
        se_vals = np.asarray(se_raw).flatten()
    except Exception:
        se_vals = np.full(len(X_eval), np.nan)

    cate, cate_se = {}, {}
    idx = 0
    for s in range(K):
        for z in [0, 1]:
            cate[(z, s)] = float(cate_vals[idx])
            cate_se[(z, s)] = float(se_vals[idx])
            idx += 1

    # Confusion matrices from expert subset
    C_global = compute_confusion_matrix(Z_true[expert_mask], Z_hat[expert_mask])
    M_global = build_mixing_matrix(C_global)

    M_sub = {}
    for s in range(K):
        mask_s = expert_mask & (S == s)
        if mask_s.sum() >= 4:
            C_s = compute_confusion_matrix(Z_true[mask_s], Z_hat[mask_s])
        else:
            C_s = np.array([[0.5, 0.5], [0.5, 0.5]])
        M_sub[s] = build_mixing_matrix(C_s)

    rows = []
    for s in range(K):
        tau_obs = np.array([cate[(0, s)], cate[(1, s)]])
        se_obs = np.array([cate_se[(0, s)], cate_se[(1, s)]])

        tau_naive, se_naive = tau_obs.copy(), se_obs.copy()
        tau_global, se_global = invert_mixing_safe(M_global, tau_obs, se_obs)
        tau_ec, se_ec = invert_mixing_safe(M_sub[s], tau_obs, se_obs)

        base = {'K': K, 'regime': config['regime'], 'n_expert': config['n_expert'],
                'subgroup': s, 'mc_run': seed}

        for z in [0, 1]:
            for method, th, se in [
                ('naive_forest', tau_naive[z], se_naive[z]),
                ('global_forest', tau_global[z], se_global[z]),
                ('ec_hte_forest', tau_ec[z], se_ec[z]),
            ]:
                if np.isnan(se):
                    ci_lo = ci_hi = covers = np.nan
                else:
                    ci_lo = th - 1.96 * se
                    ci_hi = th + 1.96 * se
                    covers = int(ci_lo <= TAU_TRUE[z] <= ci_hi)
                rows.append({**base, 'z_level': z, 'method': method,
                             'tau_hat': th, 'se': se, 'tau_true': TAU_TRUE[z],
                             'ci_lo': ci_lo, 'ci_hi': ci_hi, 'covers': covers})

    return rows


# -- Aggregation -----------------------------------------------------------

def aggregate_results(df):
    """Compute bias, RMSE, coverage per config x method x subgroup x z_level."""
    grp_cols = ['K', 'regime', 'n_expert', 'subgroup', 'z_level', 'method']
    rows = []
    for keys, g in df.groupby(grp_cols):
        K, regime, n_expert, subgroup, z_level, method = keys
        err = g['tau_hat'].values - g['tau_true'].values
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        cov = g['covers'].mean()
        n_mc = len(g)
        rows.append({
            'K': int(K), 'regime': regime, 'n_expert': int(n_expert),
            'subgroup': int(subgroup), 'z_level': int(z_level), 'method': method,
            'bias': bias, 'rmse': rmse, 'coverage': cov,
            'bias_se': err.std() / np.sqrt(n_mc),
            'rmse_se': (err ** 2).std() / (2 * rmse * np.sqrt(n_mc)) if rmse > 1e-10 else np.nan,
            'n_mc': n_mc,
        })
    return pd.DataFrame(rows)


# -- Output ----------------------------------------------------------------

def print_comparison_table(df_agg):
    """Print detailed per-subgroup comparison table."""
    print("\n" + "=" * 90)
    print("exp-011 CausalForest EC-HTE Results")
    print("=" * 90)

    for K in sorted(df_agg['K'].unique()):
        for regime in ['extreme', 'moderate']:
            sub = df_agg[(df_agg['K'] == K) & (df_agg['regime'] == regime)]
            if sub.empty:
                continue
            pi_s = MISCLASS_RATES[K][regime]
            pi_label = ','.join(f'{p:.2f}' for p in pi_s)
            print(f"\n-- K={K}, {regime} (pi=[{pi_label}]) " + "-" * 50)
            for ne in sorted(sub['n_expert'].unique()):
                sub2 = sub[sub['n_expert'] == ne]
                print(f"\n  n_expert = {ne}")
                print(f"  {'method':>15s}  {'sub':>3s}  {'z':>1s}  "
                      f"{'bias':>8s}  {'rmse':>8s}  {'cov':>6s}")
                print(f"  {'-'*15}  {'-'*3}  {'-'*1}  "
                      f"{'-'*8}  {'-'*8}  {'-'*6}")
                for method in ['naive_forest', 'global_forest', 'ec_hte_forest']:
                    for s_val in sorted(sub2['subgroup'].unique()):
                        for z in [0, 1]:
                            r = sub2[(sub2['method'] == method)
                                     & (sub2['subgroup'] == s_val)
                                     & (sub2['z_level'] == z)]
                            if r.empty:
                                continue
                            r = r.iloc[0]
                            print(f"  {method:>15s}  {s_val:>3d}  {z:>1d}  "
                                  f"{r['bias']:8.4f}  {r['rmse']:8.4f}  "
                                  f"{r['coverage']:6.2f}")


def print_verification(df_agg):
    """Check key patterns: ec_hte bias ~0, global hurts low-pi subgroups."""
    print("\n" + "=" * 90)
    print("Verification Checks")
    print("=" * 90)

    for K in sorted(df_agg['K'].unique()):
        for regime in ['extreme', 'moderate']:
            sub = df_agg[(df_agg['K'] == K) & (df_agg['regime'] == regime)]
            if sub.empty:
                continue
            pi_s = MISCLASS_RATES[K][regime]
            print(f"\n-- K={K}, {regime} (pi={pi_s}) --")

            for ne in sorted(sub['n_expert'].unique()):
                sub2 = sub[sub['n_expert'] == ne]
                print(f"\n  n_expert = {ne}")

                ec = sub2[sub2['method'] == 'ec_hte_forest']
                gl = sub2[sub2['method'] == 'global_forest']
                na = sub2[sub2['method'] == 'naive_forest']

                ec_bias = ec['bias'].abs().mean()
                gl_bias = gl['bias'].abs().mean()
                na_bias = na['bias'].abs().mean()
                ec_rmse = ec['rmse'].mean()
                gl_rmse = gl['rmse'].mean()
                na_rmse = na['rmse'].mean()

                print(f"    Mean |bias|:  naive={na_bias:.4f}  "
                      f"global={gl_bias:.4f}  ec_hte={ec_bias:.4f}")
                print(f"    Mean RMSE:   naive={na_rmse:.4f}  "
                      f"global={gl_rmse:.4f}  ec_hte={ec_rmse:.4f}")

                for s_val in sorted(sub2['subgroup'].unique()):
                    na_s = na[na['subgroup'] == s_val]['bias'].abs().mean()
                    gl_s = gl[gl['subgroup'] == s_val]['bias'].abs().mean()
                    ec_s = ec[ec['subgroup'] == s_val]['bias'].abs().mean()
                    tag = ""
                    if gl_s > na_s * 1.1:
                        tag = f"  << global HURTS (pi_s={pi_s[s_val]:.2f})"
                    print(f"    subgroup {s_val} (pi={pi_s[s_val]:.2f}): "
                          f"|bias| naive={na_s:.4f} global={gl_s:.4f} "
                          f"ec_hte={ec_s:.4f}{tag}")

    # Overall pattern consistency check
    print("\n-- Pattern Consistency with exp-001 --")
    methods = ['naive_forest', 'global_forest', 'ec_hte_forest']
    overall = {}
    for m in methods:
        ms = df_agg[df_agg['method'] == m]
        overall[m] = {'bias': ms['bias'].abs().mean(), 'rmse': ms['rmse'].mean()}
    print(f"  Overall mean |bias|:")
    for m in methods:
        print(f"    {m:>15s}: {overall[m]['bias']:.4f}")
    print(f"  Overall mean RMSE:")
    for m in methods:
        print(f"    {m:>15s}: {overall[m]['rmse']:.4f}")

    ec_wins = (overall['ec_hte_forest']['bias'] < overall['naive_forest']['bias']
               and overall['ec_hte_forest']['bias'] < overall['global_forest']['bias'])
    print(f"\n  ec_hte best overall bias? {'YES' if ec_wins else 'NO'}")

    # Check global hurts in at least one subgroup in extreme regime
    extreme = df_agg[df_agg['regime'] == 'extreme']
    if not extreme.empty:
        n_hurts = 0
        for K in extreme['K'].unique():
            for ne in extreme['n_expert'].unique():
                s = extreme[(extreme['K'] == K) & (extreme['n_expert'] == ne)]
                for sv in s['subgroup'].unique():
                    na_b = s[(s['method'] == 'naive_forest') & (s['subgroup'] == sv)]['bias'].abs().mean()
                    gl_b = s[(s['method'] == 'global_forest') & (s['subgroup'] == sv)]['bias'].abs().mean()
                    if gl_b > na_b * 1.1:
                        n_hurts += 1
        print(f"  Global hurts low-pi subgroups (extreme)? "
              f"{'YES' if n_hurts > 0 else 'NO'} ({n_hurts} cases)")


# -- Main ------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exp-011 CausalForest EC-HTE')
    parser.add_argument('--n-mc', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else args.n_mc
    configs = CONFIGS[:2] if args.dry_run else CONFIGS

    print(f"exp-011 CausalForest EC-HTE | N={N} n_mc={n_mc}")
    print(f"Configs: {len(configs)} | True CATE: tau(Z=0)={TAU_TRUE[0]}, tau(Z=1)={TAU_TRUE[1]}")

    # Timing estimate
    print("\nTiming single forest fit...")
    t_test = time.time()
    td = generate_data(N, 2, 'moderate', 100, seed=9999)
    X_test = np.column_stack([td[3], td[4]]).astype(float)
    cf_test = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200, min_samples_leaf=20, random_state=42)
    cf_test.fit(td[0], td[1], X=X_test, W=None)
    cf_test.effect(X_test[:4])
    t_per = time.time() - t_test
    total = len(configs) * n_mc
    print(f"  {t_per:.2f}s/run -> estimated total: "
          f"{t_per * total:.0f}s ({t_per * total / 60:.1f} min)\n")

    all_rows = []
    count = 0
    t0 = time.time()

    for cfg in configs:
        cfg_label = f"K={cfg['K']} {cfg['regime']} ne={cfg['n_expert']}"
        for mc in range(n_mc):
            try:
                rows = run_one_mc(cfg, seed=mc)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  WARN: {cfg_label} mc={mc} failed: {e}")
            count += 1
            if count % max(1, total // 20) == 0:
                el = time.time() - t0
                eta = el / count * (total - count) if count > 0 else 0
                print(f"  [{count}/{total}] {el:.0f}s elapsed, "
                      f"ETA {eta:.0f}s | {cfg_label}")

    elapsed = time.time() - t0
    print(f"\nSweep done: {elapsed:.1f}s ({count} runs, {elapsed/count:.1f}s/run)")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate_results(df_raw)

    os.makedirs('results', exist_ok=True)
    df_agg.to_csv('results/exp011_causal_forest.csv', index=False)
    print(f"Saved: results/exp011_causal_forest.csv ({len(df_agg)} rows)")

    print_comparison_table(df_agg)
    print_verification(df_agg)
