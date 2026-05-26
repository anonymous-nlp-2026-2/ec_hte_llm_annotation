"""
Civil Comments demographic subgroup analysis for EC-HTE validation.

Uses existing LLM annotations (3 models × 3000 comments) with keyword-based
demographic subgroups to compute per-subgroup confusion matrices and run
semi-synthetic causal inference (naive / global corrected / EC-HTE).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

warnings.filterwarnings('ignore')

PROJ = Path(__file__).parent
RESULTS = PROJ / "results"
ARTIFACTS = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

LLM_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

N_SIM = 5000
N_MC = 50
N_EXPERT_LIST = [100, 250, 500]
SEED_BASE = 42
TAU_Z0 = 0.10
TAU_Z1 = 0.30
P_T = 0.5


# ── Step 1: Load data and assign demographic subgroups ──

def load_and_assign_subgroups():
    df = pd.read_csv(RESULTS / "exp023_llm_annotations.csv")

    text_lower = df['text'].str.lower()

    gender_pat = r'man\b|men\b|male|masculin|boy|father|husband|son\b|brother|guy|gentleman|woman|women|female|girl|mother|wife|daughter|sister|feminin|lady|ladies'
    race_pat = r'\bwhites?\b|\bcaucasian|\beuropean|\bblacks?\b|\bafrican|\basian|\bhispanic|\blatino|\blatina|\bgay\b|\blesbian|\bhomosexual|\bbisexual|\btransgender|\btrans\b|\bqueer\b|\blgbt'
    religion_pat = r'\bmuslim|\bislam|\bmosque|\bquran|\ballah|\bchristian|\bchurch\b|\bbible|\bjesus|\bchrist\b|\bjews?\b|\bjewish|\bjudaism|\bhindu|\bbuddhist'

    df['dm_gender'] = text_lower.str.contains(gender_pat, regex=True, na=False).astype(int)
    df['dm_race'] = text_lower.str.contains(race_pat, regex=True, na=False).astype(int)
    df['dm_religion'] = text_lower.str.contains(religion_pat, regex=True, na=False).astype(int)

    def assign(row):
        if row['dm_race'] == 1:
            return 'race_lgbtq'
        if row['dm_religion'] == 1:
            return 'religion'
        if row['dm_gender'] == 1:
            return 'gender'
        return 'no_identity'

    df['demo_subgroup'] = df.apply(assign, axis=1)
    return df


# ── Step 2: Per-subgroup confusion matrices ──

def compute_subgroup_cm(df, pred_col, subgroup_col='demo_subgroup'):
    results = {}
    for sg in sorted(df[subgroup_col].unique()):
        mask = df[subgroup_col] == sg
        sub = df[mask]
        y = sub['y_true'].values
        yhat = sub[pred_col].values

        valid = ~np.isnan(yhat)
        y, yhat = y[valid], yhat[valid].astype(int)

        tp = ((y == 1) & (yhat == 1)).sum()
        tn = ((y == 0) & (yhat == 0)).sum()
        fp = ((y == 0) & (yhat == 1)).sum()
        fn = ((y == 1) & (yhat == 0)).sum()
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()

        fpr = fp / n_neg if n_neg > 0 else 0
        fnr = fn / n_pos if n_pos > 0 else 0
        acc = (tp + tn) / len(y) if len(y) > 0 else 0

        results[sg] = {
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'n': len(y), 'n_pos': int(n_pos), 'n_neg': int(n_neg),
            'fpr': round(fpr, 4), 'fnr': round(fnr, 4),
            'accuracy': round(acc, 4),
            'kappa': round(1 / (1 - fpr - fnr), 4) if (1 - fpr - fnr) != 0 else float('inf'),
            'pi': round((fpr + fnr) / 2, 4),
        }
    return results


# ── Step 3: Semi-synthetic DGP (diff-in-means with misclassification) ──

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


def estimate_cm_from_expert(z_true, z_hat, subgroups, labels):
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
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + \
           gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b) - \
           (gammaln(a) + gammaln(b) - gammaln(a + b))


def hb_estimate_cm(z_hat_expert, z_true_expert, subgroups_expert, labels,
                   a0=2.0, b0=2.0, grid_size=200):
    pi_grid = np.linspace(0.001, 0.499, grid_size)
    cms = {}

    all_z = z_true_expert
    all_zh = z_hat_expert
    n_err_global = (all_z != all_zh).sum()
    n_global = len(all_z)

    for s in labels:
        ms = subgroups_expert == s
        zt, zh = z_true_expert[ms], z_hat_expert[ms]
        n_s = ms.sum()
        n_err_s = (zt != zh).sum()

        log_prior = np.zeros(grid_size)
        for i, pi_val in enumerate(pi_grid):
            log_prior[i] = log_beta_binom(n_err_global, n_global, a0, b0) + \
                           np.log(pi_val) * 0

        log_lik = np.zeros(grid_size)
        for i, pi_val in enumerate(pi_grid):
            log_lik[i] = n_err_s * np.log(pi_val + 1e-15) + \
                         (n_s - n_err_s) * np.log(1 - pi_val + 1e-15)

        log_post = log_lik
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        pi_hat = (pi_grid * post).sum()
        C = np.array([[1 - pi_hat, pi_hat], [pi_hat, 1 - pi_hat]])
        cms[s] = C
    return cms


def run_semisynthetic(real_cms_per_model, subgroup_labels, subgroup_weights):
    rng = np.random.default_rng(SEED_BASE)
    results = []

    for model_name, real_cm_dict in real_cms_per_model.items():
        fpr_per_sg = {s: real_cm_dict[s]['fpr'] for s in subgroup_labels}
        fnr_per_sg = {s: real_cm_dict[s]['fnr'] for s in subgroup_labels}

        for mc in range(N_MC):
            seed = SEED_BASE + mc
            rng_mc = np.random.default_rng(seed)

            subgroups = rng_mc.choice(subgroup_labels, size=N_SIM, p=subgroup_weights)
            T = rng_mc.binomial(1, P_T, N_SIM)
            Z_true = rng_mc.binomial(1, 0.3, N_SIM)

            tau_true = np.where(Z_true == 1, TAU_Z1, TAU_Z0)
            Y0 = rng_mc.normal(0, 1, N_SIM)
            Y1 = Y0 + tau_true
            Y = np.where(T == 1, Y1, Y0)

            Z_hat = Z_true.copy()
            for s in subgroup_labels:
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
                tau0_oracle, _ = diff_in_means(Y, T, Z_true == 0)

                # Naive
                tau_naive, se_naive = diff_in_means(Y, T, Z_hat == 1)
                tau0_naive, _ = diff_in_means(Y, T, Z_hat == 0)

                # Global correction
                fpr_g = (z_true_exp != z_hat_exp)[z_true_exp == 0].mean() if (z_true_exp == 0).sum() > 0 else 0.1
                fnr_g = (z_true_exp != z_hat_exp)[z_true_exp == 1].mean() if (z_true_exp == 1).sum() > 0 else 0.1
                denom_g = 1 - fpr_g - fnr_g
                if abs(denom_g) < 0.05:
                    denom_g = 0.05 * np.sign(denom_g) if denom_g != 0 else 0.05
                p_z1_hat = (Z_hat == 1).mean()
                p_z1_corrected = (p_z1_hat - fpr_g) / denom_g
                p_z1_corrected = np.clip(p_z1_corrected, 0.05, 0.95)

                tau1_dim, se1 = diff_in_means(Y, T, Z_hat == 1)
                tau0_dim, se0 = diff_in_means(Y, T, Z_hat == 0)
                if not np.isnan(tau1_dim) and not np.isnan(tau0_dim):
                    tau_global = (tau1_dim - fpr_g * tau0_dim) / denom_g
                else:
                    tau_global = np.nan

                # EC-HTE (per-subgroup correction)
                cms_strat = estimate_cm_from_expert(z_true_exp, z_hat_exp, sg_exp, subgroup_labels)
                tau_echte_parts = []
                w_parts = []
                for s in subgroup_labels:
                    ms = subgroups == s
                    C = cms_strat[s]
                    fpr_s, fnr_s = C[0, 1], C[1, 0]
                    denom_s = 1 - fpr_s - fnr_s
                    if abs(denom_s) < 0.05:
                        denom_s = 0.05
                    tau1_s, _ = diff_in_means(Y, T, ms & (Z_hat == 1))
                    tau0_s, _ = diff_in_means(Y, T, ms & (Z_hat == 0))
                    if not np.isnan(tau1_s) and not np.isnan(tau0_s):
                        tau_s_corr = (tau1_s - fpr_s * tau0_s) / denom_s
                        tau_echte_parts.append(tau_s_corr)
                        w_parts.append(ms.sum())
                if len(tau_echte_parts) > 0:
                    w_arr = np.array(w_parts, dtype=float)
                    w_arr /= w_arr.sum()
                    tau_echte = np.dot(w_arr, tau_echte_parts)
                else:
                    tau_echte = np.nan

                # HB EC-HTE
                cms_hb = hb_estimate_cm(z_hat_exp, z_true_exp, sg_exp, subgroup_labels)
                tau_hb_parts = []
                w_hb_parts = []
                for s in subgroup_labels:
                    ms = subgroups == s
                    C = cms_hb[s]
                    fpr_s, fnr_s = C[0, 1], C[1, 0]
                    denom_s = 1 - fpr_s - fnr_s
                    if abs(denom_s) < 0.05:
                        denom_s = 0.05
                    tau1_s, _ = diff_in_means(Y, T, ms & (Z_hat == 1))
                    tau0_s, _ = diff_in_means(Y, T, ms & (Z_hat == 0))
                    if not np.isnan(tau1_s) and not np.isnan(tau0_s):
                        tau_s_corr = (tau1_s - fpr_s * tau0_s) / denom_s
                        tau_hb_parts.append(tau_s_corr)
                        w_hb_parts.append(ms.sum())
                if len(tau_hb_parts) > 0:
                    w_arr = np.array(w_hb_parts, dtype=float)
                    w_arr /= w_arr.sum()
                    tau_hb = np.dot(w_arr, tau_hb_parts)
                else:
                    tau_hb = np.nan

                tau_true_z1 = TAU_Z1

                for method, est in [('oracle', tau_oracle), ('naive', tau_naive),
                                     ('global_corrected', tau_global),
                                     ('stratified_mle', tau_echte),
                                     ('hb_ec_hte', tau_hb)]:
                    if not np.isnan(est):
                        bias = est - tau_true_z1
                        results.append({
                            'model': model_name, 'mc': mc, 'n_expert': n_expert,
                            'method': method, 'estimate': est,
                            'bias': bias, 'abs_bias': abs(bias),
                            'sign_correct': int(np.sign(est) == np.sign(tau_true_z1)),
                        })

    return pd.DataFrame(results)


# ── Step 4: Main ──

def main():
    print("=== Civil Comments Demographic Subgroup Analysis ===\n")

    # Load data
    df = load_and_assign_subgroups()

    subgroup_labels = sorted(df['demo_subgroup'].unique())
    print(f"Subgroups: {subgroup_labels}")
    print(f"Subgroup sizes:")
    sg_counts = df['demo_subgroup'].value_counts()
    for sg in subgroup_labels:
        n = sg_counts[sg]
        toxic = df.loc[df['demo_subgroup'] == sg, 'y_true'].mean()
        print(f"  {sg}: n={n}, toxic_rate={toxic:.3f}")

    # Compute confusion matrices per model per subgroup
    all_cms = {}
    cm_rows = []
    for model in LLM_MODELS:
        pred_col = f'pred_{model}'
        cms = compute_subgroup_cm(df, pred_col)
        all_cms[model] = cms

        for sg, cm in cms.items():
            cm['model'] = model
            cm['subgroup'] = sg
            cm_rows.append(cm)

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(ARTIFACTS / "civilcomments_analysis.csv", index=False)
    print("\n--- Per-subgroup confusion matrices saved ---")

    # Print CM table
    print("\n=== Per-Subgroup Error Rates ===")
    print(f"{'Model':<16} {'Subgroup':<14} {'N':>5} {'FPR':>7} {'FNR':>7} {'Acc':>7} {'κ':>8} {'π':>7}")
    print("-" * 75)
    for model in LLM_MODELS:
        for sg in subgroup_labels:
            cm = all_cms[model][sg]
            print(f"{model:<16} {sg:<14} {cm['n']:>5} {cm['fpr']:>7.4f} {cm['fnr']:>7.4f} "
                  f"{cm['accuracy']:>7.4f} {cm['kappa']:>8.3f} {cm['pi']:>7.4f}")
        print()

    # Compute kappa range and bias amplification potential
    print("\n=== Kappa Analysis ===")
    for model in LLM_MODELS:
        kappas = [all_cms[model][sg]['kappa'] for sg in subgroup_labels if all_cms[model][sg]['kappa'] != float('inf')]
        fpr_range = max(all_cms[model][sg]['fpr'] for sg in subgroup_labels) - min(all_cms[model][sg]['fpr'] for sg in subgroup_labels)
        fnr_range = max(all_cms[model][sg]['fnr'] for sg in subgroup_labels) - min(all_cms[model][sg]['fnr'] for sg in subgroup_labels)
        if kappas:
            print(f"{model}: κ range [{min(kappas):.3f}, {max(kappas):.3f}], "
                  f"ΔFPR={fpr_range:.4f}, ΔFNR={fnr_range:.4f}")

    # Subgroup weights for simulation
    subgroup_weights = np.array([sg_counts[sg] for sg in subgroup_labels], dtype=float)
    subgroup_weights /= subgroup_weights.sum()

    # Run semi-synthetic DGP
    print("\n=== Running Semi-Synthetic DGP ===")
    print(f"N={N_SIM}, MC={N_MC}, τ(Z=0)={TAU_Z0}, τ(Z=1)={TAU_Z1}")
    sim_df = run_semisynthetic(all_cms, subgroup_labels, subgroup_weights)

    # Summarize
    summary = sim_df.groupby(['model', 'n_expert', 'method']).agg(
        mean_bias=('bias', 'mean'),
        mean_abs_bias=('abs_bias', 'mean'),
        rmse=('bias', lambda x: np.sqrt((x**2).mean())),
        sign_reversal_rate=('sign_correct', lambda x: 1 - x.mean()),
    ).reset_index()

    print("\n=== Semi-Synthetic Results ===")
    print(f"{'Model':<16} {'n_exp':>5} {'Method':<20} {'|Bias|':>8} {'RMSE':>8} {'SignRev':>8}")
    print("-" * 70)
    for _, row in summary.sort_values(['model', 'n_expert', 'method']).iterrows():
        print(f"{row['model']:<16} {row['n_expert']:>5} {row['method']:<20} "
              f"{row['mean_abs_bias']:>8.4f} {row['rmse']:>8.4f} {row['sign_reversal_rate']:>8.3f}")

    # Check for sign reversal
    print("\n=== Sign Reversal Analysis ===")
    for model in LLM_MODELS:
        for n_exp in N_EXPERT_LIST:
            for method in ['naive', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
                mask = (sim_df['model'] == model) & (sim_df['n_expert'] == n_exp) & (sim_df['method'] == method)
                sub = sim_df[mask]
                if len(sub) > 0:
                    sr = 1 - sub['sign_correct'].mean()
                    if sr > 0.01:
                        print(f"  {model} / n={n_exp} / {method}: sign reversal rate = {sr:.3f}")

    # Check bias amplification
    print("\n=== Bias Amplification ===")
    for model in LLM_MODELS:
        for n_exp in N_EXPERT_LIST:
            naive_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'naive')]['mean_abs_bias'].values
            global_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'global_corrected')]['mean_abs_bias'].values
            echte_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'stratified_mle')]['mean_abs_bias'].values
            if len(naive_bias) > 0 and len(echte_bias) > 0:
                amp = echte_bias[0] / naive_bias[0] if naive_bias[0] > 0 else float('inf')
                if amp > 1.5 or amp < 0.7:
                    print(f"  {model} / n={n_exp}: naive |bias|={naive_bias[0]:.4f}, "
                          f"EC-HTE |bias|={echte_bias[0]:.4f}, amplification={amp:.2f}x")

    # Generate report
    generate_report(df, all_cms, summary, sim_df, subgroup_labels, subgroup_weights)
    print(f"\nArtifacts saved to {ARTIFACTS}")


def generate_report(df, all_cms, summary, sim_df, subgroup_labels, subgroup_weights):
    lines = []
    lines.append("# Civil Comments Demographic Subgroup Analysis — EC-HTE Validation\n")
    lines.append("## 1. Dataset & Subgroup Design\n")
    lines.append(f"**Dataset**: Civil Comments, {len(df)} annotated comments")
    lines.append(f"**Subgroups** (K={len(subgroup_labels)}): keyword-based demographic assignment")
    lines.append(f"**Subgroup labels**: {', '.join(subgroup_labels)}")
    lines.append(f"**Subgroup weights**: {', '.join(f'{s}={w:.3f}' for s, w in zip(subgroup_labels, subgroup_weights))}")
    sg_counts = df['demo_subgroup'].value_counts()
    for sg in subgroup_labels:
        n = sg_counts[sg]
        toxic = df.loc[df['demo_subgroup'] == sg, 'y_true'].mean()
        lines.append(f"  - {sg}: n={n}, P(toxic)={toxic:.3f}")

    lines.append("\n## 2. Per-Subgroup Confusion Matrices\n")
    lines.append("| Model | Subgroup | N | FPR | FNR | Accuracy | κ | π |")
    lines.append("|-------|----------|---|-----|-----|----------|---|---|")
    for model in LLM_MODELS:
        for sg in subgroup_labels:
            cm = all_cms[model][sg]
            kappa_str = f"{cm['kappa']:.3f}" if cm['kappa'] != float('inf') else "∞"
            lines.append(f"| {model} | {sg} | {cm['n']} | {cm['fpr']:.4f} | {cm['fnr']:.4f} | {cm['accuracy']:.4f} | {kappa_str} | {cm['pi']:.4f} |")

    lines.append("\n## 3. Key Findings: Misclassification Heterogeneity\n")
    for model in LLM_MODELS:
        fprs = [all_cms[model][sg]['fpr'] for sg in subgroup_labels]
        fnrs = [all_cms[model][sg]['fnr'] for sg in subgroup_labels]
        kappas = [all_cms[model][sg]['kappa'] for sg in subgroup_labels if all_cms[model][sg]['kappa'] != float('inf')]
        lines.append(f"**{model}**: ΔFPR={max(fprs)-min(fprs):.4f}, ΔFNR={max(fnrs)-min(fnrs):.4f}")
        if kappas:
            lines.append(f"  κ range: [{min(kappas):.3f}, {max(kappas):.3f}]")

    lines.append("\n## 4. Semi-Synthetic DGP Results\n")
    lines.append(f"**DGP**: N={N_SIM}, τ(Z=0)={TAU_Z0}, τ(Z=1)={TAU_Z1}, MC={N_MC}")
    lines.append(f"**Real asymmetric CMs** from LLM annotations on demographic subgroups\n")
    lines.append("| Model | n_expert | Method | |Bias| | RMSE | Sign Reversal Rate |")
    lines.append("|-------|----------|--------|-------|------|--------------------|")
    for _, row in summary.sort_values(['model', 'n_expert', 'method']).iterrows():
        lines.append(f"| {row['model']} | {row['n_expert']:.0f} | {row['method']} | "
                     f"{row['mean_abs_bias']:.4f} | {row['rmse']:.4f} | {row['sign_reversal_rate']:.3f} |")

    # Sign reversal summary
    lines.append("\n## 5. Sign Reversal Analysis\n")
    any_reversal = False
    for model in LLM_MODELS:
        for n_exp in N_EXPERT_LIST:
            for method in ['naive', 'global_corrected', 'stratified_mle', 'hb_ec_hte']:
                mask = (sim_df['model'] == model) & (sim_df['n_expert'] == n_exp) & (sim_df['method'] == method)
                sub = sim_df[mask]
                if len(sub) > 0:
                    sr = 1 - sub['sign_correct'].mean()
                    if sr > 0.01:
                        lines.append(f"- **{model}** / n_expert={n_exp} / {method}: sign reversal rate = {sr:.1%}")
                        any_reversal = True
    if not any_reversal:
        lines.append("No sign reversals detected (τ(Z=1)=0.30 is far from zero).")

    # Bias amplification
    lines.append("\n## 6. Bias Amplification\n")
    for model in LLM_MODELS:
        for n_exp in N_EXPERT_LIST:
            naive_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'naive')]['mean_abs_bias'].values
            echte_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'stratified_mle')]['mean_abs_bias'].values
            hb_bias = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'hb_ec_hte')]['mean_abs_bias'].values
            if len(naive_bias) > 0 and len(echte_bias) > 0 and len(hb_bias) > 0 and naive_bias[0] > 0:
                amp_mle = echte_bias[0] / naive_bias[0]
                amp_hb = hb_bias[0] / naive_bias[0]
                lines.append(f"- **{model}** / n={n_exp}: naive→MLE amplification={amp_mle:.2f}x, naive→HB={amp_hb:.2f}x")

    lines.append("\n## 7. Summary\n")
    lines.append("### Key numbers:")
    # (a) sign reversal
    lines.append(f"- **(a) Sign reversal**: {'Yes' if any_reversal else 'No'} (with τ(Z=1)={TAU_Z1})")
    # (b) bias amplification range
    amps = []
    for model in LLM_MODELS:
        for n_exp in N_EXPERT_LIST:
            nb = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'naive')]['mean_abs_bias'].values
            eb = summary[(summary['model'] == model) & (summary['n_expert'] == n_exp) & (summary['method'] == 'stratified_mle')]['mean_abs_bias'].values
            if len(nb) > 0 and len(eb) > 0 and nb[0] > 0:
                amps.append(eb[0] / nb[0])
    if amps:
        lines.append(f"- **(b) Bias amplification**: {min(amps):.2f}x – {max(amps):.2f}x (stratified MLE vs naive)")
    # (c) kappa ranges
    for model in LLM_MODELS:
        kappas = [all_cms[model][sg]['kappa'] for sg in subgroup_labels if all_cms[model][sg]['kappa'] != float('inf')]
        if kappas:
            lines.append(f"- **(c) κ range ({model})**: [{min(kappas):.3f}, {max(kappas):.3f}]")

    report = "\n".join(lines)
    with open(ARTIFACTS / "civilcomments_report.md", "w") as f:
        f.write(report)
    print("Report written to civilcomments_report.md")


if __name__ == "__main__":
    main()
