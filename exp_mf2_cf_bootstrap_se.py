#!/usr/bin/env python3
"""
exp_mf2_cf_bootstrap_se.py — CausalForestDML: Bootstrap SE vs Plug-in SE Coverage (MF2)

Validates whether CausalForestDML's plug-in SE captures confusion matrix
estimation uncertainty. Compares:
  1. Plug-in SE: forest effect_inference SE propagated through M⁻¹ (delta method)
  2. CM-Bootstrap SE: resample experts → re-estimate CM → re-apply M⁻¹ → SD of corrected CATEs
  3. Combined SE: sqrt(plugin² + bootstrap²)

DGP: K=2, π=[0.05, 0.25], τ(z=0)=0.10, τ(z=1)=0.40
N=5000, n_expert ∈ {100, 250, 500}, 50 MC runs, 100 CM-bootstrap resamples
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
MISCLASS = [0.05, 0.25]
TAU_Z0 = 0.10
TAU_Z1 = 0.40
TAU_TRUE = np.array([TAU_Z0, TAU_Z1])
LABELS = ['A', 'B']


def generate_data(n_expert, seed):
    rng = np.random.RandomState(seed)
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(LABELS, N, p=[0.5, 0.5])
    X = rng.randn(N, 5)
    tau = np.where(Z == 1, TAU_Z1, TAU_Z0)
    Y = 1.0 + tau * T + 0.5 * X[:, 0] + rng.randn(N)
    mc_rates = np.where(S == 'A', MISCLASS[0], MISCLASS[1])
    Z_tilde = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)
    idx = rng.choice(N, min(n_expert, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True
    return dict(T=T, Z=Z, Z_tilde=Z_tilde, S=S, X=X, Y=Y, expert_mask=expert_mask)


def estimate_cm(z_true, z_hat):
    C = np.zeros((2, 2))
    for z in [0, 1]:
        mask = z_true == z
        n = mask.sum()
        for zh in [0, 1]:
            C[z, zh] = ((z_hat[mask] == zh).sum() + 1) / (n + 2) if n > 0 else 0.5
    return C


def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        M[zh, 0] = C[0, zh] / cs if cs > 0 else 0.5
        M[zh, 1] = C[1, zh] / cs if cs > 0 else 0.5
    return M


def invert_mixing(M, tau_obs, se_obs):
    if np.linalg.cond(M) > 100 or abs(np.linalg.det(M)) < 1e-10:
        return tau_obs.copy(), se_obs.copy(), True
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c, False


def run_one_mc(n_expert, seed, n_bootstrap):
    data = generate_data(n_expert, seed)
    Y, T, Z, Z_tilde, S, X = (
        data['Y'], data['T'], data['Z'], data['Z_tilde'], data['S'], data['X']
    )
    em = data['expert_mask']

    z_exp = Z[em]
    zt_exp = Z_tilde[em]
    s_exp = S[em]
    n_exp = em.sum()

    # Per-subgroup CM from expert sample
    M_sub = {}
    for s in LABELS:
        sm = s_exp == s
        if sm.sum() >= 4:
            C_s = estimate_cm(z_exp[sm], zt_exp[sm])
        else:
            C_s = estimate_cm(z_exp, zt_exp)
        M_sub[s] = build_mixing_matrix(C_s)

    # ── Fit CausalForestDML ──
    X_features = np.column_stack([Z_tilde, X]).astype(float)
    cf = CausalForestDML(
        model_y=LinearRegression(),
        model_t=LogisticRegression(solver='lbfgs'),
        discrete_treatment=True,
        n_estimators=200,
        min_samples_leaf=20,
        random_state=seed,
    )
    cf.fit(Y, T, X=X_features)

    # Individual CATEs
    cate_all = cf.effect(X_features).flatten()

    # Per-subgroup observed CATEs: mean(τ̂ᵢ | Z_tilde=z, S=s)
    sub_obs = {}
    for s in LABELS:
        sm = S == s
        vals = []
        for z in [0, 1]:
            mask = sm & (Z_tilde == z)
            if mask.sum() == 0:
                return None
            vals.append(cate_all[mask].mean())
        sub_obs[s] = np.array(vals)

    # Plug-in SE from effect_inference at representative points
    X_mean = X.mean(axis=0)
    X_eval = np.array([
        np.concatenate([[0], X_mean]),
        np.concatenate([[1], X_mean]),
    ])
    try:
        inf = cf.effect_inference(X_eval)
        se_raw = np.asarray(inf.stderr).flatten()
        if np.any(np.isnan(se_raw)) or np.any(se_raw <= 0):
            se_raw = np.full(2, np.nan)
    except Exception:
        se_raw = np.full(2, np.nan)

    sub_se_obs = {s: se_raw.copy() for s in LABELS}

    # ── Plug-in: M⁻¹ corrected ──
    plugin_tau_sub, plugin_se_sub = {}, {}
    for s in LABELS:
        tau_c, se_c, _ = invert_mixing(M_sub[s], sub_obs[s], sub_se_obs[s])
        plugin_tau_sub[s] = tau_c
        plugin_se_sub[s] = se_c

    sub_w = {s: (S == s).sum() / N for s in LABELS}
    plugin_tau = sum(sub_w[s] * plugin_tau_sub[s] for s in LABELS)
    plugin_se = np.sqrt(sum((sub_w[s] * plugin_se_sub[s]) ** 2 for s in LABELS))

    # ── CM Bootstrap ──
    boot_taus = np.zeros((n_bootstrap, 2))
    rng_boot = np.random.RandomState(seed * 1000 + 7)

    for b in range(n_bootstrap):
        boot_idx = rng_boot.choice(n_exp, n_exp, replace=True)
        z_b, zt_b, s_b = z_exp[boot_idx], zt_exp[boot_idx], s_exp[boot_idx]

        b_tau = np.zeros(2)
        for s in LABELS:
            sm_b = s_b == s
            if sm_b.sum() >= 4:
                C_b = estimate_cm(z_b[sm_b], zt_b[sm_b])
            else:
                C_b = estimate_cm(z_b, zt_b)
            M_b = build_mixing_matrix(C_b)
            tau_c_b, _, _ = invert_mixing(M_b, sub_obs[s], sub_se_obs[s])
            b_tau += sub_w[s] * tau_c_b
        boot_taus[b] = b_tau

    boot_se = np.nanstd(boot_taus, axis=0)

    # ── Results ──
    result = {'n_expert': n_expert, 'seed': seed}
    for z in [0, 1]:
        zl = f'z{z}'
        # Plug-in CI
        pi_lo = plugin_tau[z] - 1.96 * plugin_se[z]
        pi_hi = plugin_tau[z] + 1.96 * plugin_se[z]
        result[f'plugin_covers_{zl}'] = int(pi_lo <= TAU_TRUE[z] <= pi_hi)
        result[f'plugin_se_{zl}'] = plugin_se[z]
        result[f'plugin_width_{zl}'] = pi_hi - pi_lo
        result[f'tau_hat_{zl}'] = plugin_tau[z]

        # CM-Bootstrap CI (uses same point estimate, different SE)
        bs_lo = plugin_tau[z] - 1.96 * boot_se[z]
        bs_hi = plugin_tau[z] + 1.96 * boot_se[z]
        result[f'bootstrap_covers_{zl}'] = int(bs_lo <= TAU_TRUE[z] <= bs_hi)
        result[f'bootstrap_se_{zl}'] = boot_se[z]
        result[f'bootstrap_width_{zl}'] = bs_hi - bs_lo

        # Combined CI
        comb_se = np.sqrt(plugin_se[z] ** 2 + boot_se[z] ** 2)
        cb_lo = plugin_tau[z] - 1.96 * comb_se
        cb_hi = plugin_tau[z] + 1.96 * comb_se
        result[f'combined_covers_{zl}'] = int(cb_lo <= TAU_TRUE[z] <= cb_hi)
        result[f'combined_se_{zl}'] = comb_se
        result[f'combined_width_{zl}'] = cb_hi - cb_lo

    return result


def aggregate(df_raw):
    rows = []
    for ne, grp in df_raw.groupby('n_expert'):
        for method, col_prefix in [
            ('plugin', 'plugin'),
            ('cm_bootstrap', 'bootstrap'),
            ('combined', 'combined'),
        ]:
            row = {'n_expert': int(ne), 'se_method': method}
            for z in [0, 1]:
                zl = f'z{z}'
                row[f'coverage_{zl}'] = grp[f'{col_prefix}_covers_{zl}'].mean()
                row[f'mean_se_{zl}'] = grp[f'{col_prefix}_se_{zl}'].mean()
                row[f'mean_width_{zl}'] = grp[f'{col_prefix}_width_{zl}'].mean()
            row['coverage_avg'] = (row['coverage_z0'] + row['coverage_z1']) / 2
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n_mc = 2 if args.dry_run else 50
    n_bootstrap = 5 if args.dry_run else 100
    n_experts = [100, 250, 500]
    seeds = list(range(42, 42 + n_mc))

    print(f"MF2: CausalForestDML Bootstrap SE vs Plug-in SE Coverage")
    print(f"N={N}, n_expert={n_experts}, N_MC={n_mc}, N_boot={n_bootstrap}")
    print(f"DGP: K=2, pi={MISCLASS}, tau(z=0)={TAU_Z0}, tau(z=1)={TAU_Z1}")
    if args.dry_run:
        print("** DRY RUN **")
    print()

    all_rows = []
    total = len(n_experts) * n_mc
    count = 0
    t0 = time.time()

    for ne in n_experts:
        for seed in seeds:
            result = run_one_mc(ne, seed, n_bootstrap)
            if result is not None:
                all_rows.append(result)
            count += 1
            if count % max(1, total // 10) == 0:
                el = time.time() - t0
                eta = el / count * (total - count)
                print(f"  [{count}/{total}] {el:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(all_rows)} successful MC runs)\n")

    df_raw = pd.DataFrame(all_rows)
    df_agg = aggregate(df_raw)

    # ── Print summary ──
    print("=" * 115)
    print("SUMMARY: CausalForestDML SE Coverage")
    print("=" * 115)
    header = (f"{'n_exp':>5} {'method':>14} | {'cov_z0':>7} {'cov_z1':>7} "
              f"{'cov_avg':>7} | {'SE_z0':>8} {'SE_z1':>8} | "
              f"{'width_z0':>9} {'width_z1':>9}")
    print(header)
    print("-" * len(header))
    for _, r in df_agg.iterrows():
        print(f"{int(r['n_expert']):5d} {r['se_method']:>14s} | "
              f"{r['coverage_z0']:7.3f} {r['coverage_z1']:7.3f} "
              f"{r['coverage_avg']:7.3f} | "
              f"{r['mean_se_z0']:8.4f} {r['mean_se_z1']:8.4f} | "
              f"{r['mean_width_z0']:9.4f} {r['mean_width_z1']:9.4f}")

    # ── Save ──
    proj = '.'
    os.makedirs(f'{proj}/results', exist_ok=True)
    csv_path = f'{proj}/results/exp_mf2_cf_bootstrap_se.csv'
    raw_path = f'{proj}/results/exp_mf2_cf_bootstrap_se_raw.csv'
    df_agg.to_csv(csv_path, index=False)
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df_agg)} rows)")
    print(f"Saved: {raw_path} ({len(df_raw)} rows)")

    artifacts_dir = os.path.expanduser(
        '~/.agent-ml-research-idea_gen_0520/projects/'
        'ec_hte_llm_annotation/artifacts/mf2/data/')
    os.makedirs(artifacts_dir, exist_ok=True)
    art_path = os.path.join(artifacts_dir, 'exp_mf2_cf_bootstrap_se.csv')
    df_agg.to_csv(art_path, index=False)
    print(f"Saved: {art_path}")

    # ── Interpretation ──
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    for ne in n_experts:
        sub = df_agg[df_agg['n_expert'] == ne]
        pi_cov = sub[sub['se_method'] == 'plugin']['coverage_avg'].values[0]
        bs_cov = sub[sub['se_method'] == 'cm_bootstrap']['coverage_avg'].values[0]
        cb_cov = sub[sub['se_method'] == 'combined']['coverage_avg'].values[0]
        if pi_cov >= 0.90:
            verdict = "plug-in SE sufficient"
        elif bs_cov >= 0.90:
            verdict = "plug-in insufficient, CM-bootstrap recommended"
        else:
            verdict = "both insufficient — possible CF SE calibration issue"
        print(f"  n_expert={ne:3d}: plugin={pi_cov:.2f}, bootstrap={bs_cov:.2f}, "
              f"combined={cb_cov:.2f} -> {verdict}")


if __name__ == '__main__':
    main()
