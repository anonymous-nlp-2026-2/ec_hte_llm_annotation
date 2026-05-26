#!/usr/bin/env python3
"""UCB Hate Speech annotator demographic subgroup analysis for EC-HTE.

Computes per-subgroup confusion matrices (LLM vs annotator label) grouped by
annotator demographics (race, ideology, education, gender, age, income).
If max Δπ > 0.15, runs semi-synthetic EC-HTE validation.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.special import gammaln
from pathlib import Path

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

SEED_BASE = 42
N_SIM = 5000
N_MC = 100
N_EXPERT_LIST = [100, 250, 500]
TAU_Z0 = 0.10
TAU_Z1 = 0.30
P_T = 0.5


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Data loading and exploration
# ═══════════════════════════════════════════════════════════════════

def load_data():
    raw = pd.read_parquet('data/exp004_raw.parquet')
    llm = pd.read_csv('results/exp004_llm_annotations.csv')

    llm_comments = set(llm['comment_id'].values)
    annotator_llm = raw[raw['comment_id'].isin(llm_comments)].copy()

    annotator_llm = annotator_llm.merge(
        llm[['comment_id', 'Z_llm']], on='comment_id', how='left')
    annotator_llm['Z_human'] = (annotator_llm['dehumanize'] >= 2).astype(int)

    return raw, llm, annotator_llm


def build_demographic_columns(df):
    """Decode one-hot demographics into single categorical columns."""
    # Race: pick from one-hot columns (annotators can be multi-race; use primary)
    race_cols = [c for c in df.columns if c.startswith('annotator_race_')]
    race_labels = [c.replace('annotator_race_', '') for c in race_cols]

    def get_primary_race(row):
        vals = [row[c] for c in race_cols]
        if sum(vals) == 0:
            return 'unknown'
        idx = np.argmax(vals)
        return race_labels[idx]

    df['race'] = df.apply(get_primary_race, axis=1)

    # Ideology: already categorical
    df['ideology'] = df['annotator_ideology'].fillna('unknown')

    # Education: already categorical
    df['education'] = df['annotator_educ'].fillna('unknown')

    # Gender: already categorical
    df['gender'] = df['annotator_gender'].fillna('unknown')

    # Age: bin into groups
    df['age_group'] = pd.cut(
        df['annotator_age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56+'],
        include_lowest=True
    ).astype(str).replace('nan', 'unknown')

    # Income: already categorical
    df['income'] = df['annotator_income'].fillna('unknown')

    # Ideology grouped: liberal vs conservative vs neutral
    ideo_map = {
        'extremely_liberal': 'liberal',
        'liberal': 'liberal',
        'slightly_liberal': 'liberal',
        'neutral': 'neutral',
        'no_opinion': 'neutral',
        'slightly_conservative': 'conservative',
        'conservative': 'conservative',
        'extremely_conservative': 'conservative',
    }
    df['ideology_grouped'] = df['ideology'].map(ideo_map).fillna('unknown')

    return df


def explore_and_save(raw, annotator_llm):
    lines = []
    lines.append("# UCB Hate Speech Annotator Demographic Exploration\n")
    lines.append(f"## Dataset\n")
    lines.append(f"- Raw data: {len(raw)} rows, {raw.comment_id.nunique()} comments, "
                 f"{raw.annotator_id.nunique()} annotators")
    lines.append(f"- LLM-annotated subset: {len(annotator_llm)} annotator-level rows, "
                 f"{annotator_llm.comment_id.nunique()} comments, "
                 f"{annotator_llm.annotator_id.nunique()} annotators\n")

    lines.append("## Annotator Demographic Features\n")

    demo_features = {
        'race': 'race',
        'gender': 'gender',
        'ideology': 'ideology',
        'ideology_grouped': 'ideology_grouped',
        'education': 'education',
        'age_group': 'age_group',
        'income': 'income',
    }

    for name, col in demo_features.items():
        vc = annotator_llm[col].value_counts()
        lines.append(f"### {name}")
        for val, cnt in vc.items():
            lines.append(f"  - {val}: {cnt} ({100*cnt/len(annotator_llm):.1f}%)")
        lines.append("")

    lines.append("## Label distributions (LLM-annotated subset)\n")
    lines.append(f"- Z_llm=1: {(annotator_llm.Z_llm==1).sum()} "
                 f"({100*(annotator_llm.Z_llm==1).mean():.1f}%)")
    lines.append(f"- Z_human=1: {(annotator_llm.Z_human==1).sum()} "
                 f"({100*(annotator_llm.Z_human==1).mean():.1f}%)")
    lines.append(f"- Overall agreement: {(annotator_llm.Z_llm == annotator_llm.Z_human).mean():.4f}")

    text = '\n'.join(lines)
    (RESULTS / 'ucb_demographic_exploration.md').write_text(text)
    print(text)
    return demo_features


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Per-subgroup confusion matrices
# ═══════════════════════════════════════════════════════════════════

def compute_cm_stats(z_true, z_hat):
    tp = ((z_true == 1) & (z_hat == 1)).sum()
    tn = ((z_true == 0) & (z_hat == 0)).sum()
    fp = ((z_true == 0) & (z_hat == 1)).sum()
    fn = ((z_true == 1) & (z_hat == 0)).sum()
    n_pos = (z_true == 1).sum()
    n_neg = (z_true == 0).sum()
    n = len(z_true)

    fpr = fp / n_neg if n_neg > 0 else np.nan
    fnr = fn / n_pos if n_pos > 0 else np.nan
    acc = (tp + tn) / n if n > 0 else np.nan
    pi = (fpr + fnr) / 2 if not (np.isnan(fpr) or np.isnan(fnr)) else np.nan

    return {
        'n': n, 'n_pos': n_pos, 'n_neg': n_neg,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'fpr': fpr, 'fnr': fnr, 'accuracy': acc, 'pi': pi,
    }


def compute_per_subgroup_cm(df, feature_col, min_n=50):
    results = []
    groups = df[feature_col].value_counts()

    for val, cnt in groups.items():
        if cnt < min_n:
            continue
        mask = df[feature_col] == val
        sub = df[mask]
        stats = compute_cm_stats(sub['Z_human'].values, sub['Z_llm'].values)
        stats['feature'] = feature_col
        stats['subgroup'] = val
        results.append(stats)

    return results


def compute_all_delta_pi(df, demo_features, min_n=50):
    all_results = []
    delta_pi_summary = []

    for name, col in demo_features.items():
        results = compute_per_subgroup_cm(df, col, min_n=min_n)
        if len(results) < 2:
            continue

        all_results.extend(results)

        pis = [r['pi'] for r in results if not np.isnan(r['pi'])]
        if len(pis) >= 2:
            delta_pi = max(pis) - min(pis)
            best_sg = results[np.argmax([r['pi'] for r in results])]['subgroup']
            worst_sg = results[np.argmin([r['pi'] for r in results])]['subgroup']
            delta_pi_summary.append({
                'feature': name,
                'n_subgroups': len(results),
                'min_pi': min(pis),
                'max_pi': max(pis),
                'delta_pi': delta_pi,
                'highest_pi_subgroup': best_sg,
                'lowest_pi_subgroup': worst_sg,
            })

    return pd.DataFrame(all_results), pd.DataFrame(delta_pi_summary)


def save_phase2_results(cm_df, summary_df, annotator_llm):
    cm_df.to_csv(RESULTS / 'ucb_demographic_cm_detail.csv', index=False)
    summary_df.to_csv(RESULTS / 'ucb_demographic_delta_pi.csv', index=False)

    lines = []
    lines.append("# UCB Hate Speech — Annotator Demographic Δπ Summary\n")
    lines.append("## Approach\n")
    lines.append("For each (comment, annotator) pair with LLM annotation:")
    lines.append("- Z_true = annotator's dehumanize label (≥2 → 1, else 0)")
    lines.append("- Z_hat = GPT-4.1 LLM label")
    lines.append("- Subgroup = annotator demographic attribute")
    lines.append(f"- N = {len(annotator_llm)} annotator-level observations\n")

    lines.append("## Δπ by demographic feature\n")
    lines.append("| Feature | # Groups | min π | max π | Δπ | Highest π group | Lowest π group |")
    lines.append("|---------|----------|-------|-------|-----|-----------------|----------------|")
    for _, row in summary_df.sort_values('delta_pi', ascending=False).iterrows():
        lines.append(f"| {row['feature']} | {row['n_subgroups']} | "
                     f"{row['min_pi']:.4f} | {row['max_pi']:.4f} | "
                     f"**{row['delta_pi']:.4f}** | {row['highest_pi_subgroup']} | "
                     f"{row['lowest_pi_subgroup']} |")

    max_dp = summary_df['delta_pi'].max()
    best_feat = summary_df.loc[summary_df['delta_pi'].idxmax(), 'feature']
    lines.append(f"\n## Key finding\n")
    lines.append(f"Best feature: **{best_feat}** with Δπ = **{max_dp:.4f}**")
    if max_dp > 0.15:
        lines.append(f"\n✓ Δπ > 0.15 threshold — proceeding with semi-synthetic EC-HTE validation")
    else:
        lines.append(f"\n✗ Max Δπ = {max_dp:.4f} < 0.15 threshold — EC-HTE unlikely to show improvement")

    lines.append("\n## Per-subgroup detail\n")
    for feat in summary_df.sort_values('delta_pi', ascending=False)['feature'].values:
        feat_data = cm_df[cm_df['feature'] == feat].sort_values('pi', ascending=False)
        lines.append(f"### {feat}")
        lines.append("| Subgroup | N | FPR | FNR | π | Accuracy |")
        lines.append("|----------|---|-----|-----|---|----------|")
        for _, row in feat_data.iterrows():
            lines.append(f"| {row['subgroup']} | {row['n']} | "
                         f"{row['fpr']:.4f} | {row['fnr']:.4f} | "
                         f"{row['pi']:.4f} | {row['accuracy']:.4f} |")
        lines.append("")

    text = '\n'.join(lines)
    (RESULTS / 'ucb_demographic_delta_pi_summary.md').write_text(text)
    print(text)
    return max_dp, best_feat


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Semi-synthetic EC-HTE
# ═══════════════════════════════════════════════════════════════════

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


def estimate_cm_mle(z_true, z_hat, subgroups, labels):
    cms = {}
    for s in labels:
        ms = subgroups == s
        zt, zh = z_true[ms], z_hat[ms]
        n = ms.sum()
        if n == 0:
            cms[s] = np.array([[0.5, 0.5], [0.5, 0.5]])
            continue
        C = np.zeros((2, 2))
        for i in [0, 1]:
            mi = zt == i
            ni = mi.sum()
            for j in [0, 1]:
                C[i, j] = ((zh[mi] == j).sum() + 1) / (ni + 2) if ni > 0 else 0.5
        cms[s] = C
    return cms


def log_beta_binom(k, n, a, b):
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) +
            gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b) -
            (gammaln(a) + gammaln(b) - gammaln(a + b)))


def hb_estimate_cm(z_hat_expert, z_true_expert, subgroups_expert, labels,
                   a0=2.0, b0=2.0, grid_size=200):
    pi_grid = np.linspace(0.001, 0.499, grid_size)
    cms = {}

    n_err_global = (z_true_expert != z_hat_expert).sum()
    n_global = len(z_true_expert)

    for s in labels:
        ms = subgroups_expert == s
        zt, zh = z_true_expert[ms], z_hat_expert[ms]
        n_s = ms.sum()
        n_err_s = (zt != zh).sum()

        log_lik = np.zeros(grid_size)
        for i, pi_val in enumerate(pi_grid):
            log_lik[i] = (n_err_s * np.log(pi_val + 1e-15) +
                          (n_s - n_err_s) * np.log(1 - pi_val + 1e-15))

        log_post = log_lik
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        pi_hat = (pi_grid * post).sum()
        C = np.array([[1 - pi_hat, pi_hat], [pi_hat, 1 - pi_hat]])
        cms[s] = C
    return cms


def run_semisynthetic(real_cm_dict, subgroup_labels, subgroup_weights):
    rng = np.random.default_rng(SEED_BASE)
    results = []

    fpr_per_sg = {s: real_cm_dict[s]['fpr'] for s in subgroup_labels}
    fnr_per_sg = {s: real_cm_dict[s]['fnr'] for s in subgroup_labels}

    print(f"\nReal CMs used for simulation:")
    for s in subgroup_labels:
        print(f"  {s}: FPR={fpr_per_sg[s]:.4f}, FNR={fnr_per_sg[s]:.4f}, "
              f"π={real_cm_dict[s]['pi']:.4f}, weight={subgroup_weights[s]:.3f}")

    for mc in range(N_MC):
        if mc % 20 == 0:
            print(f"  MC iteration {mc}/{N_MC}")
        seed = SEED_BASE + mc
        rng_mc = np.random.default_rng(seed)

        sg_keys = list(subgroup_labels)
        sg_probs = np.array([subgroup_weights[s] for s in sg_keys])
        subgroups = np.array(rng_mc.choice(sg_keys, size=N_SIM, p=sg_probs))

        T = rng_mc.binomial(1, P_T, N_SIM)
        Z_true = rng_mc.binomial(1, 0.3, N_SIM)
        tau_true = np.where(Z_true == 1, TAU_Z1, TAU_Z0)
        Y0 = rng_mc.normal(0, 1, N_SIM)
        Y1 = Y0 + tau_true
        Y = np.where(T == 1, Y1, Y0)

        Z_hat = Z_true.copy()
        for s in sg_keys:
            ms = subgroups == s
            for i in np.where(ms & (Z_true == 0))[0]:
                if rng_mc.random() < fpr_per_sg[s]:
                    Z_hat[i] = 1
            for i in np.where(ms & (Z_true == 1))[0]:
                if rng_mc.random() < fnr_per_sg[s]:
                    Z_hat[i] = 0

        for n_expert in N_EXPERT_LIST:
            idx_expert = rng_mc.choice(N_SIM, n_expert, replace=False)
            z_true_exp = Z_true[idx_expert]
            z_hat_exp = Z_hat[idx_expert]
            sg_exp = subgroups[idx_expert]

            # Oracle
            tau_oracle, se_oracle = diff_in_means(Y, T, Z_true == 1)

            # Naive
            tau_naive, se_naive = diff_in_means(Y, T, Z_hat == 1)

            # Global correction
            fpr_g = ((z_true_exp != z_hat_exp) & (z_true_exp == 0)).sum()
            n_neg_exp = (z_true_exp == 0).sum()
            fpr_g = fpr_g / n_neg_exp if n_neg_exp > 0 else 0.1

            fnr_g = ((z_true_exp != z_hat_exp) & (z_true_exp == 1)).sum()
            n_pos_exp = (z_true_exp == 1).sum()
            fnr_g = fnr_g / n_pos_exp if n_pos_exp > 0 else 0.1

            denom_g = 1 - fpr_g - fnr_g
            if abs(denom_g) < 0.05:
                denom_g = 0.05 * np.sign(denom_g) if denom_g != 0 else 0.05

            tau1_dim, _ = diff_in_means(Y, T, Z_hat == 1)
            tau0_dim, _ = diff_in_means(Y, T, Z_hat == 0)
            if not np.isnan(tau1_dim) and not np.isnan(tau0_dim):
                tau_global = (tau1_dim - fpr_g * tau0_dim) / denom_g
            else:
                tau_global = np.nan

            # EC-HTE: MLE per-subgroup
            cms_mle = estimate_cm_mle(z_true_exp, z_hat_exp, sg_exp, sg_keys)
            tau_mle = _echte_estimate(Y, T, Z_hat, subgroups, sg_keys, cms_mle)

            # EC-HTE: HB per-subgroup
            cms_hb = hb_estimate_cm(z_hat_exp, z_true_exp, sg_exp, sg_keys)
            tau_hb = _echte_estimate(Y, T, Z_hat, subgroups, sg_keys, cms_hb)

            tau_true_z1 = TAU_Z1

            for method, est in [('oracle', tau_oracle), ('naive', tau_naive),
                                ('global_corrected', tau_global),
                                ('stratified_mle', tau_mle),
                                ('hb_ec_hte', tau_hb)]:
                if not np.isnan(est):
                    bias = est - tau_true_z1
                    ci_lo = est - 1.96 * (se_oracle if method == 'oracle' else se_naive if method == 'naive' else se_naive)
                    ci_hi = est + 1.96 * (se_oracle if method == 'oracle' else se_naive if method == 'naive' else se_naive)
                    covers = int(ci_lo <= tau_true_z1 <= ci_hi)
                    results.append({
                        'mc': mc, 'n_expert': n_expert,
                        'method': method, 'estimate': est,
                        'bias': bias, 'abs_bias': abs(bias),
                        'covers': covers,
                    })

    return pd.DataFrame(results)


def _echte_estimate(Y, T, Z_hat, subgroups, sg_keys, cms):
    tau_parts = []
    w_parts = []
    for s in sg_keys:
        ms = subgroups == s
        C = cms[s]
        fpr_s, fnr_s = C[0, 1], C[1, 0]
        denom_s = 1 - fpr_s - fnr_s
        if abs(denom_s) < 0.05:
            denom_s = 0.05
        tau1_s, _ = diff_in_means(Y, T, ms & (Z_hat == 1))
        tau0_s, _ = diff_in_means(Y, T, ms & (Z_hat == 0))
        if not np.isnan(tau1_s) and not np.isnan(tau0_s):
            tau_s_corr = (tau1_s - fpr_s * tau0_s) / denom_s
            tau_parts.append(tau_s_corr)
            w_parts.append(ms.sum())
    if len(tau_parts) > 0:
        w_arr = np.array(w_parts, dtype=float)
        w_arr /= w_arr.sum()
        return np.dot(w_arr, tau_parts)
    return np.nan


def save_phase3_results(results_df, best_feature, real_cm_dict, subgroup_labels):
    results_df.to_csv(RESULTS / 'ucb_demographic_echte_results.csv', index=False)

    summary = results_df.groupby(['method', 'n_expert']).agg(
        mean_bias=('bias', 'mean'),
        median_bias=('bias', 'median'),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        mean_abs_bias=('abs_bias', 'mean'),
        coverage=('covers', 'mean'),
        n_mc=('mc', 'nunique'),
    ).reset_index()

    summary.to_csv(RESULTS / 'ucb_demographic_echte_summary.csv', index=False)

    lines = []
    lines.append("# UCB Hate Speech — Semi-synthetic EC-HTE Results\n")
    lines.append(f"## Setup")
    lines.append(f"- Subgroup feature: **{best_feature}**")
    lines.append(f"- N_sim = {N_SIM}, N_MC = {N_MC}")
    lines.append(f"- τ(Z=0) = {TAU_Z0}, τ(Z=1) = {TAU_Z1}")
    lines.append(f"- n_expert ∈ {N_EXPERT_LIST}\n")

    lines.append("## Real confusion matrices used\n")
    lines.append("| Subgroup | FPR | FNR | π |")
    lines.append("|----------|-----|-----|---|")
    for s in subgroup_labels:
        d = real_cm_dict[s]
        lines.append(f"| {s} | {d['fpr']:.4f} | {d['fnr']:.4f} | {d['pi']:.4f} |")

    lines.append("\n## Results\n")
    for n_exp in N_EXPERT_LIST:
        sub = summary[summary['n_expert'] == n_exp]
        lines.append(f"### n_expert = {n_exp}\n")
        lines.append("| Method | Mean Bias | RMSE | Coverage |")
        lines.append("|--------|-----------|------|----------|")
        for _, row in sub.sort_values('rmse').iterrows():
            lines.append(f"| {row['method']} | {row['mean_bias']:.4f} | "
                         f"{row['rmse']:.4f} | {row['coverage']:.3f} |")
        lines.append("")

    text = '\n'.join(lines)
    (RESULTS / 'ucb_demographic_echte_analysis.md').write_text(text)
    print(text)
    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 1: Data exploration")
    print("=" * 60)

    raw, llm, annotator_llm = load_data()
    annotator_llm = build_demographic_columns(annotator_llm)
    demo_features = explore_and_save(raw, annotator_llm)

    print("\n" + "=" * 60)
    print("Phase 2: Per-subgroup confusion matrices")
    print("=" * 60)

    cm_df, summary_df = compute_all_delta_pi(annotator_llm, demo_features, min_n=50)
    max_dp, best_feat = save_phase2_results(cm_df, summary_df, annotator_llm)

    print(f"\n>>> Max Δπ = {max_dp:.4f} (feature: {best_feat})")

    if max_dp > 0.15:
        print("\n" + "=" * 60)
        print("Phase 3: Semi-synthetic EC-HTE validation")
        print("=" * 60)

        best_col = demo_features[best_feat]
        sg_vals = annotator_llm[best_col].value_counts()
        sg_vals = sg_vals[sg_vals >= 50]
        subgroup_labels = sorted(sg_vals.index.tolist())

        subgroup_weights = {}
        total = sg_vals.sum()
        for s in subgroup_labels:
            subgroup_weights[s] = sg_vals[s] / total

        real_cm_dict = {}
        for s in subgroup_labels:
            mask = annotator_llm[best_col] == s
            sub = annotator_llm[mask]
            real_cm_dict[s] = compute_cm_stats(
                sub['Z_human'].values, sub['Z_llm'].values)

        results_df = run_semisynthetic(real_cm_dict, subgroup_labels, subgroup_weights)
        summary = save_phase3_results(results_df, best_feat, real_cm_dict, subgroup_labels)

        # Check if EC-HTE improves over naive/global
        for n_exp in N_EXPERT_LIST:
            sub = summary[summary['n_expert'] == n_exp]
            naive_rmse = sub[sub['method'] == 'naive']['rmse'].values[0]
            global_rmse = sub[sub['method'] == 'global_corrected']['rmse'].values[0]
            echte_methods = sub[sub['method'].isin(['stratified_mle', 'hb_ec_hte'])]
            best_echte_rmse = echte_methods['rmse'].min()
            print(f"\nn_expert={n_exp}: naive RMSE={naive_rmse:.4f}, "
                  f"global RMSE={global_rmse:.4f}, best EC-HTE RMSE={best_echte_rmse:.4f}")
            if best_echte_rmse < global_rmse:
                print("  → EC-HTE IMPROVES over global correction!")
            else:
                print("  → EC-HTE does NOT improve over global correction")
    else:
        print(f"\nMax Δπ = {max_dp:.4f} < 0.15 — skipping semi-synthetic EC-HTE")
        print("Consider using finer-grained ideology or race×ideology interaction")

        # Try additional features: interactions
        print("\n" + "=" * 60)
        print("Bonus: Testing interaction features")
        print("=" * 60)

        annotator_llm['race_x_ideology'] = (
            annotator_llm['race'] + '_' + annotator_llm['ideology_grouped'])
        annotator_llm['gender_x_ideology'] = (
            annotator_llm['gender'] + '_' + annotator_llm['ideology_grouped'])

        interaction_features = {
            'race_x_ideology': 'race_x_ideology',
            'gender_x_ideology': 'gender_x_ideology',
        }

        cm_int, sum_int = compute_all_delta_pi(annotator_llm, interaction_features, min_n=30)
        if len(sum_int) > 0:
            max_dp_int = sum_int['delta_pi'].max()
            best_int = sum_int.loc[sum_int['delta_pi'].idxmax(), 'feature']
            print(f"\nInteraction features — max Δπ = {max_dp_int:.4f} ({best_int})")

            all_cm = pd.concat([cm_df, cm_int])
            all_summary = pd.concat([summary_df, sum_int])
            all_cm.to_csv(RESULTS / 'ucb_demographic_cm_detail.csv', index=False)
            all_summary.to_csv(RESULTS / 'ucb_demographic_delta_pi.csv', index=False)

            # Re-save summary with interactions
            save_phase2_results(all_cm, all_summary, annotator_llm)

            if max_dp_int > 0.15:
                print(f"\nInteraction Δπ > 0.15 — running semi-synthetic EC-HTE")
                best_col = interaction_features[best_int]
                sg_vals = annotator_llm[best_col].value_counts()
                sg_vals = sg_vals[sg_vals >= 30]
                subgroup_labels = sorted(sg_vals.index.tolist())

                subgroup_weights = {}
                total = sg_vals.sum()
                for s in subgroup_labels:
                    subgroup_weights[s] = sg_vals[s] / total

                real_cm_dict = {}
                for s in subgroup_labels:
                    mask = annotator_llm[best_col] == s
                    sub = annotator_llm[mask]
                    real_cm_dict[s] = compute_cm_stats(
                        sub['Z_human'].values, sub['Z_llm'].values)

                results_df = run_semisynthetic(real_cm_dict, subgroup_labels, subgroup_weights)
                save_phase3_results(results_df, best_int, real_cm_dict, subgroup_labels)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
