"""
exp-023 Phase 2 v2: Semi-synthetic DGP on Civil Comments.

Two regimes:
  A) Real LLM confusion matrices (from annotation) — extreme regime
  B) Controlled moderate misclassification (mimicking TweetEval setup) — moderate regime

Both use Civil Comments subgroup structure (text_length × has_question, K=4).

Methods: Oracle, Naive, Oracle-corrected (true CM), Global, Stratified MLE, HB EC-HTE
n_expert sweep: [100, 250, 500], MC=50

Output:
  results/exp023_semisynthetic_v2_raw.csv
  results/exp023_semisynthetic_v2_summary.csv
  results/exp023_semisynthetic_v2_analysis.md
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
DELTA_TAU = 0.30
TAU_Z1 = TAU_Z0 + DELTA_TAU
P_T = 0.5

CONTROLLED_CMS = {
    'moderate': {
        'S0': {'fpr': 0.08, 'fnr': 0.10},
        'S1': {'fpr': 0.12, 'fnr': 0.18},
        'S2': {'fpr': 0.06, 'fnr': 0.08},
        'S3': {'fpr': 0.18, 'fnr': 0.25},
    },
    'extreme': {
        'S0': {'fpr': 0.05, 'fnr': 0.10},
        'S1': {'fpr': 0.10, 'fnr': 0.15},
        'S2': {'fpr': 0.15, 'fnr': 0.20},
        'S3': {'fpr': 0.25, 'fnr': 0.25},
    },
}


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


def estimate_cm_per_subgroup(z_true, z_hat, subgroups, labels):
    C = {}
    for s in labels:
        ms = subgroups == s
        C_s = np.zeros((2, 2))
        for zt in [0, 1]:
            mask = ms & (z_true == zt)
            n = mask.sum()
            for zh in [0, 1]:
                C_s[zt, zh] = ((z_hat[mask] == zh).sum() + 1) / (n + 2) if n > 0 else 0.5
        C[s] = C_s
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


def generate_data(subgroup_cms, subgroup_weights, p_z1, n_expert, seed):
    rng = np.random.RandomState(seed)

    S_idx = rng.choice(4, size=N, p=subgroup_weights)
    S = np.array([SUBGROUP_LABELS[i] for i in S_idx])

    Z = rng.binomial(1, p_z1, N)
    T = rng.binomial(1, P_T, N)

    Z_hat = np.empty(N, dtype=int)
    true_cm_per_sub = {}
    for i in range(4):
        s_mask = S_idx == i
        s_name = SUBGROUP_LABELS[i]
        cm = get_cm_matrix(subgroup_cms[s_name]['fpr'], subgroup_cms[s_name]['fnr'])
        true_cm_per_sub[s_name] = cm
        z_s = Z[s_mask]
        p_obs1 = np.where(z_s == 0, cm[0, 1], cm[1, 1])
        Z_hat[s_mask] = rng.binomial(1, p_obs1)

    tau_vals = np.where(Z == 1, TAU_Z1, TAU_Z0)
    eps = rng.randn(N)
    Y = 1.0 + tau_vals * T + eps

    expert_idx = rng.choice(N, n_expert, replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[expert_idx] = True

    return {
        'T': T, 'Z': Z, 'Z_hat': Z_hat, 'S': S, 'S_idx': S_idx,
        'Y': Y, 'tau': tau_vals, 'expert_mask': expert_mask,
        'true_cm_per_sub': true_cm_per_sub,
    }


def run_one_mc(data, seed):
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    expert_mask = data['expert_mask']
    true_cm_per_sub = data['true_cm_per_sub']

    z_exp = Z[expert_mask]
    zh_exp = Z_hat[expert_mask]
    s_exp = S[expert_mask]

    C_global = estimate_confusion_matrix(z_exp, zh_exp)
    p_z_global = np.array([(z_exp == 0).mean(), (z_exp == 1).mean()])
    p_zh_global = np.array([(Z_hat == 0).mean(), (Z_hat == 1).mean()])
    M_global = build_mixing_matrix(C_global, p_z_global, p_zh_global)

    C_mle = estimate_cm_per_subgroup(z_exp, zh_exp, s_exp, SUBGROUP_LABELS)

    counts = get_counts(z_exp, zh_exp, s_exp, SUBGROUP_LABELS)
    C_hb = estimate_cm_hb_gibbs(counts, SUBGROUP_LABELS, seed=seed)

    M_mle_sub, M_hb_sub, M_oracle_sub = {}, {}, {}
    for s_val in SUBGROUP_LABELS:
        sm_exp = (s_exp == s_val)
        sm_all = (S == s_val)
        p_z_s = np.array([(z_exp[sm_exp] == 0).mean(), (z_exp[sm_exp] == 1).mean()]) \
            if sm_exp.sum() >= 4 else p_z_global
        p_zh_s = np.array([(Z_hat[sm_all] == 0).mean(), (Z_hat[sm_all] == 1).mean()])
        M_mle_sub[s_val] = build_mixing_matrix(C_mle[s_val], p_z_s, p_zh_s)
        M_hb_sub[s_val] = build_mixing_matrix(C_hb[s_val], p_z_s, p_zh_s)
        # Oracle CM: use true confusion matrix
        p_z_true = np.array([(Z[sm_all] == 0).mean(), (Z[sm_all] == 1).mean()])
        M_oracle_sub[s_val] = build_mixing_matrix(true_cm_per_sub[s_val], p_z_true, p_zh_s)

    rows = []
    for s_val in SUBGROUP_LABELS:
        s_mask = S == s_val

        tau_obs = np.array([
            diff_in_means(Y, T, s_mask & (Z_hat == 0))[0],
            diff_in_means(Y, T, s_mask & (Z_hat == 1))[0],
        ])
        se_obs = np.array([
            diff_in_means(Y, T, s_mask & (Z_hat == 0))[1],
            diff_in_means(Y, T, s_mask & (Z_hat == 1))[1],
        ])

        for z_level in [0, 1]:
            true_cate = TAU_Z1 if z_level == 1 else TAU_Z0

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
                # Oracle-corrected (uses true CM)
                tau_oc, se_oc = invert_mixing_safe(M_oracle_sub[s_val], tau_obs, se_obs)
                rows.append(make_row('oracle_corrected', tau_oc[z_level], se_oc[z_level]))

                tau_gc, se_gc = invert_mixing_safe(M_global, tau_obs, se_obs)
                rows.append(make_row('global_corrected', tau_gc[z_level], se_gc[z_level]))

                tau_ml, se_ml = invert_mixing_safe(M_mle_sub[s_val], tau_obs, se_obs)
                rows.append(make_row('stratified_mle', tau_ml[z_level], se_ml[z_level]))

                tau_hb, se_hb = invert_mixing_safe(M_hb_sub[s_val], tau_obs, se_obs)
                rows.append(make_row('hb_ec_hte', tau_hb[z_level], se_hb[z_level]))
            else:
                for m in ['oracle_corrected', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
                    rows.append(make_row(m, np.nan, np.nan))

    return rows


def run_experiment(regime_name, subgroup_cms, subgroup_weights, p_z1):
    all_rows = []
    for n_expert in N_EXPERT_LIST:
        t0 = time.time()
        for mc in range(N_MC):
            seed = SEED_BASE + mc
            data = generate_data(subgroup_cms, subgroup_weights, p_z1, n_expert, seed)
            mc_rows = run_one_mc(data, seed)
            for r in mc_rows:
                r['regime'] = regime_name
                r['n_expert'] = n_expert
            all_rows.extend(mc_rows)
        elapsed = time.time() - t0
        print(f"  {regime_name} n_expert={n_expert}: {N_MC} MC done in {elapsed:.1f}s", flush=True)
    return all_rows


def aggregate(df_raw):
    group_cols = ['regime', 'n_expert', 'subgroup', 'z_level', 'method']
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


def write_analysis(df_summary):
    lines = []
    lines.append("# exp-023: Civil Comments Semi-Synthetic EC-HTE Validation (v2)\n")
    lines.append("## Design\n")
    lines.append("- Dataset: Civil Comments (toxicity detection), subgroups: text_length × has_question (K=4)")
    lines.append(f"- DGP: N={N}, P(Z=1)=0.30, tau(z=0)={TAU_Z0}, tau(z=1)={TAU_Z1}, MC={N_MC}")
    lines.append("- Two regimes: (A) controlled moderate misclassification, (B) controlled extreme misclassification")
    lines.append("- Methods: Oracle, Naive, Oracle-corrected (true CM), Global, Stratified MLE, HB EC-HTE\n")

    for regime in df_summary['regime'].unique():
        rsub = df_summary[df_summary['regime'] == regime]
        lines.append(f"\n## Regime: {regime}\n")

        for n_exp in N_EXPERT_LIST:
            esub = rsub[rsub['n_expert'] == n_exp]
            lines.append(f"\n### n_expert = {n_exp}\n")
            lines.append("| Method | |Bias| | RMSE | Coverage |")
            lines.append("|--------|-------|------|----------|")
            for m in ['oracle', 'naive', 'oracle_corrected', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
                sub = esub[esub['method'] == m]
                if len(sub) > 0:
                    lines.append(f"| {m} | {sub['mean_abs_bias'].mean():.4f} | "
                                 f"{sub['rmse'].mean():.4f} | {sub['coverage'].mean():.3f} |")

        # EC-HTE vs Global comparison at n_expert=250
        e250 = rsub[rsub['n_expert'] == 250]
        if len(e250) > 0:
            lines.append(f"\n### Bias reduction: EC-HTE vs Global (n_expert=250)\n")
            lines.append("| Subgroup | Z | Naive |Bias| | Global |Bias| | EC-HTE |Bias| |")
            lines.append("|----------|---|-----------|-------------|-------------|")
            for s in SUBGROUP_LABELS:
                for z in [0, 1]:
                    nv = e250[(e250['subgroup'] == s) & (e250['z_level'] == z) & (e250['method'] == 'naive')]
                    gc = e250[(e250['subgroup'] == s) & (e250['z_level'] == z) & (e250['method'] == 'global_corrected')]
                    hb = e250[(e250['subgroup'] == s) & (e250['z_level'] == z) & (e250['method'] == 'hb_ec_hte')]
                    if len(nv) > 0 and len(gc) > 0 and len(hb) > 0:
                        lines.append(f"| {s} | {z} | {nv.iloc[0]['mean_abs_bias']:.4f} | "
                                     f"{gc.iloc[0]['mean_abs_bias']:.4f} | {hb.iloc[0]['mean_abs_bias']:.4f} |")

    # Final summary
    lines.append("\n## Summary\n")
    for regime in df_summary['regime'].unique():
        rsub = df_summary[(df_summary['regime'] == regime) & (df_summary['n_expert'] == 250)]
        if len(rsub) == 0:
            continue
        lines.append(f"\n**{regime}** (n_expert=250):")
        for m in ['oracle', 'naive', 'oracle_corrected', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
            sub = rsub[rsub['method'] == m]
            if len(sub) > 0:
                lines.append(f"- {m}: RMSE={sub['rmse'].mean():.4f}, |Bias|={sub['mean_abs_bias'].mean():.4f}, Cov={sub['coverage'].mean():.3f}")

    return "\n".join(lines)


def main():
    df_ann = pd.read_parquet(RESULTS / "exp023_civil_comments_pool.parquet")
    sg_counts = df_ann['subgroup'].value_counts()
    subgroup_weights = np.array([sg_counts.get(s, 0) for s in SUBGROUP_LABELS], dtype=float)
    subgroup_weights /= subgroup_weights.sum()
    p_z1 = df_ann['y_true'].mean()

    print(f"P(Z=1) = {p_z1:.3f}")
    print(f"Subgroup weights: {dict(zip(SUBGROUP_LABELS, [f'{w:.3f}' for w in subgroup_weights]))}")

    all_rows = []

    # Regime A: Controlled moderate misclassification
    for regime_name, regime_cms in CONTROLLED_CMS.items():
        print(f"\n=== Regime: {regime_name} ===")
        for s in SUBGROUP_LABELS:
            print(f"  {s}: FPR={regime_cms[s]['fpr']:.3f}, FNR={regime_cms[s]['fnr']:.3f}")
        rows = run_experiment(regime_name, regime_cms, subgroup_weights, p_z1)
        all_rows.extend(rows)

    # Regime B: Real LLM confusion matrices (if available)
    cm_path = RESULTS / "exp023_confusion_matrices.json"
    if cm_path.exists():
        cms = json.loads(cm_path.read_text())
        for model in cms:
            model_cms = cms[model]
            if '_global' in model_cms:
                model_cms.pop('_global')
            subgroup_cm_dict = {}
            has_all = True
            for s in SUBGROUP_LABELS:
                if s in model_cms:
                    subgroup_cm_dict[s] = {
                        'fpr': model_cms[s]['fpr'],
                        'fnr': model_cms[s]['fnr'],
                    }
                else:
                    has_all = False
            if has_all:
                print(f"\n=== Regime: real_{model} ===")
                for s in SUBGROUP_LABELS:
                    print(f"  {s}: FPR={subgroup_cm_dict[s]['fpr']:.3f}, FNR={subgroup_cm_dict[s]['fnr']:.3f}")
                rows = run_experiment(f"real_{model}", subgroup_cm_dict, subgroup_weights, p_z1)
                all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(RESULTS / "exp023_semisynthetic_v2_raw.csv", index=False)

    df_summary = aggregate(df_raw)
    df_summary.to_csv(RESULTS / "exp023_semisynthetic_v2_summary.csv", index=False)

    report = write_analysis(df_summary)
    (RESULTS / "exp023_semisynthetic_v2_analysis.md").write_text(report)

    print("\n" + report)
    print(f"\nSaved to results/exp023_semisynthetic_v2_*.csv and .md")


if __name__ == "__main__":
    main()
