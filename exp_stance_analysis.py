"""
Experiment B: Stance Detection — EC-HTE validation on TweetEval stance data.

Uses GPT-4o real annotations (from checkpoint) + simulated error profiles for
GPT-4o-mini and GPT-3.5-Turbo (based on real confusion matrix patterns from
the project's CivilComments multi-LLM study).

Output:
  artifacts/stance_analysis.csv
  artifacts/stance_analysis_report.md
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset

warnings.filterwarnings('ignore')

ARTIFACTS = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts")
RESULTS = Path("/home/ubuntu/ec_hte_llm_annotation/results")
CHECKPOINT = RESULTS / "stance_llm_annotations.json"

TARGETS = ['stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary', 'stance_abortion']
TARGET_NAMES = {
    'stance_atheism': 'Atheism',
    'stance_climate': 'Climate Change',
    'stance_feminist': 'Feminist Movement',
    'stance_hillary': 'Hillary Clinton',
    'stance_abortion': 'Abortion',
}
LABEL_MAP = {0: 'none', 1: 'against', 2: 'favor'}
LABEL_TO_INT = {'none': 0, 'against': 1, 'favor': 2}

SEED = 42
N_MC = 200
SAMPLE_PER_TARGET = 300

TAU_FAVOR = 0.20
TAU_AGAINST = 0.05

# ── Data Loading ────────────────────────────────────────────────────────────

def load_stance_data():
    rows = []
    for target in TARGETS:
        ds = load_dataset('cardiffnlp/tweet_eval', target)
        for split in ds:
            for row in ds[split]:
                rows.append({
                    'text': row['text'],
                    'label': row['label'],
                    'label_str': LABEL_MAP[row['label']],
                    'target': target,
                    'target_name': TARGET_NAMES[target],
                })
    return pd.DataFrame(rows)


def sample_data(df, per_target=SAMPLE_PER_TARGET, seed=SEED):
    rng = np.random.RandomState(seed)
    sampled = []
    for target in TARGETS:
        sub = df[df['target'] == target]
        n = min(len(sub), per_target)
        idx = rng.choice(len(sub), n, replace=False)
        sampled.append(sub.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


# ── Confusion Matrix from GPT-4o annotations ───────────────────────────────

def compute_cm_from_annotations(df, preds):
    pred_ints = np.array([LABEL_TO_INT.get(p, 0) for p in preds])
    true_ints = df['label'].values

    result = {}
    for target in TARGETS:
        tname = TARGET_NAMES[target]
        mask = (df['target'] == target).values
        y_true = true_ints[mask]
        y_pred = pred_ints[mask]
        n = mask.sum()

        z_true_bin = (y_true == 2).astype(int)
        z_pred_bin = (y_pred == 2).astype(int)

        cm2 = np.zeros((2, 2), dtype=int)
        for zt, zp in zip(z_true_bin, z_pred_bin):
            cm2[zt, zp] += 1

        n0 = cm2[0].sum()
        n1 = cm2[1].sum()
        fpr = (cm2[0, 1] + 1) / (n0 + 2) if n0 > 0 else 0.5
        fnr = (cm2[1, 0] + 1) / (n1 + 2) if n1 > 0 else 0.5
        acc = (cm2[0, 0] + cm2[1, 1]) / n if n > 0 else 0.5

        result[tname] = {
            'cm2': cm2.tolist(),
            'n': int(n),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'accuracy': float(acc),
            'prevalence_favor': float((y_true == 2).sum() / n) if n > 0 else 0,
        }

    # Global
    z_true_all = (true_ints == 2).astype(int)
    z_pred_all = (pred_ints == 2).astype(int)
    cm2_g = np.zeros((2, 2), dtype=int)
    for zt, zp in zip(z_true_all, z_pred_all):
        cm2_g[zt, zp] += 1
    n0g = cm2_g[0].sum()
    n1g = cm2_g[1].sum()
    result['_global'] = {
        'fpr': float((cm2_g[0, 1] + 1) / (n0g + 2)),
        'fnr': float((cm2_g[1, 0] + 1) / (n1g + 2)),
        'accuracy': float((cm2_g[0, 0] + cm2_g[1, 1]) / len(true_ints)),
    }
    return result


def simulate_llm_annotations(df, base_cm, model_name, error_scale, rng):
    """Simulate a second/third LLM by perturbing the real GPT-4o confusion rates."""
    true_ints = df['label'].values
    preds = []

    for target in TARGETS:
        tname = TARGET_NAMES[target]
        mask = (df['target'] == target).values
        y_true = true_ints[mask]
        z_true_bin = (y_true == 2).astype(int)

        base_fpr = base_cm[tname]['fpr']
        base_fnr = base_cm[tname]['fnr']

        # Perturb per-subgroup error rates to create distinct LLM profile
        fpr_sim = np.clip(base_fpr * error_scale['fpr_mult'] + error_scale.get('fpr_shift', 0), 0.01, 0.80)
        fnr_sim = np.clip(base_fnr * error_scale['fnr_mult'] + error_scale.get('fnr_shift', 0), 0.01, 0.80)

        z_pred = np.empty(len(z_true_bin), dtype=int)
        for i in range(len(z_true_bin)):
            if z_true_bin[i] == 0:
                z_pred[i] = rng.binomial(1, fpr_sim)
            else:
                z_pred[i] = 1 - rng.binomial(1, fnr_sim)

        for zp in z_pred:
            preds.append('favor' if zp == 1 else 'against')

    return preds


# ── EC-HTE Pipeline ────────────────────────────────────────────────────────

def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        col_sum = C[0, zh] + C[1, zh]
        if col_sum > 0:
            M[zh, 0] = C[0, zh] / col_sum
            M[zh, 1] = C[1, zh] / col_sum
        else:
            M[zh, :] = 0.5
    return M


def invert_mixing(M, tau_obs, se_obs):
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy(), np.inf
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    kappa = np.linalg.cond(M)
    return tau_c, se_c, kappa


def run_mc_simulation(cm_data, n_per_subgroup=1000, n_mc=N_MC, seed=SEED):
    rng = np.random.RandomState(seed)
    subgroups = [TARGET_NAMES[t] for t in TARGETS]
    llm_names = [k for k in cm_data if k != '_meta']
    all_results = []

    for llm in llm_names:
        llm_cm = cm_data[llm]
        global_cm = llm_cm['_global']

        C_global = np.array([
            [1 - global_cm['fpr'], global_cm['fpr']],
            [global_cm['fnr'], 1 - global_cm['fnr']]
        ])
        M_global = build_mixing_matrix(C_global)

        for mc in range(n_mc):
            for s in subgroups:
                sc = llm_cm[s]
                fpr_s = sc['fpr']
                fnr_s = sc['fnr']

                C_s = np.array([
                    [1 - fpr_s, fpr_s],
                    [fnr_s, 1 - fnr_s]
                ])
                M_s = build_mixing_matrix(C_s)

                prev = sc['prevalence_favor']
                n = n_per_subgroup

                Z_true = rng.binomial(1, prev, n)
                Z_hat = np.empty(n, dtype=int)
                for i in range(n):
                    if Z_true[i] == 0:
                        Z_hat[i] = rng.binomial(1, fpr_s)
                    else:
                        Z_hat[i] = 1 - rng.binomial(1, fnr_s)

                T = rng.binomial(1, 0.5, n)
                tau_true = np.where(Z_true == 1, TAU_FAVOR, TAU_AGAINST)
                Y0 = rng.normal(0, 1, n)
                Y = Y0 + T * tau_true

                tau_naive = np.zeros(2)
                se_naive = np.zeros(2)
                for z in [0, 1]:
                    mask = Z_hat == z
                    t1 = mask & (T == 1)
                    t0 = mask & (T == 0)
                    n1, n0 = t1.sum(), t0.sum()
                    if n1 < 2 or n0 < 2:
                        tau_naive[z] = np.nan
                        se_naive[z] = np.nan
                    else:
                        tau_naive[z] = Y[t1].mean() - Y[t0].mean()
                        se_naive[z] = np.sqrt(Y[t1].var(ddof=1)/n1 + Y[t0].var(ddof=1)/n0)

                tau_global, se_global, kappa_global = invert_mixing(M_global, tau_naive, se_naive)
                tau_echte, se_echte, kappa_s = invert_mixing(M_s, tau_naive, se_naive)

                tau_true_z = np.array([TAU_AGAINST, TAU_FAVOR])

                for z in [0, 1]:
                    all_results.append({
                        'llm': llm,
                        'subgroup': s,
                        'z': z,
                        'z_label': 'favor' if z == 1 else 'not_favor',
                        'mc': mc,
                        'tau_true': tau_true_z[z],
                        'tau_naive': tau_naive[z],
                        'tau_global': tau_global[z],
                        'tau_echte': tau_echte[z],
                        'se_naive': se_naive[z],
                        'se_global': se_global[z],
                        'se_echte': se_echte[z],
                        'bias_naive': tau_naive[z] - tau_true_z[z],
                        'bias_global': tau_global[z] - tau_true_z[z],
                        'bias_echte': tau_echte[z] - tau_true_z[z],
                        'fpr_s': fpr_s,
                        'fnr_s': fnr_s,
                        'kappa_s': kappa_s,
                        'kappa_global': kappa_global,
                    })

    return pd.DataFrame(all_results)


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze_results(mc_df, cm_data):
    summary_rows = []
    for llm in mc_df['llm'].unique():
        for s in mc_df['subgroup'].unique():
            for z in [0, 1]:
                sub = mc_df[(mc_df['llm'] == llm) & (mc_df['subgroup'] == s) & (mc_df['z'] == z)]
                if len(sub) == 0:
                    continue

                mean_bias_naive = sub['bias_naive'].mean()
                mean_bias_global = sub['bias_global'].mean()
                mean_bias_echte = sub['bias_echte'].mean()

                rmse_naive = np.sqrt((sub['bias_naive'] ** 2).mean())
                rmse_global = np.sqrt((sub['bias_global'] ** 2).mean())
                rmse_echte = np.sqrt((sub['bias_echte'] ** 2).mean())

                tau_true = sub['tau_true'].iloc[0]
                sign_reversal_naive = (np.sign(sub['tau_naive'].mean()) != np.sign(tau_true)) if tau_true != 0 else False
                sign_reversal_global = (np.sign(sub['tau_global'].mean()) != np.sign(tau_true)) if tau_true != 0 else False
                sign_reversal_echte = (np.sign(sub['tau_echte'].mean()) != np.sign(tau_true)) if tau_true != 0 else False

                bias_amplified = abs(mean_bias_global) > abs(mean_bias_naive)

                sc = cm_data[llm][s]
                g = cm_data[llm]['_global']

                denom = 1 - sc['fpr'] - sc['fnr']
                kappa_cm = 1.0 / denom if abs(denom) > 1e-10 else float('inf')

                summary_rows.append({
                    'llm': llm,
                    'subgroup': s,
                    'z': z,
                    'z_label': 'favor' if z == 1 else 'not_favor',
                    'tau_true': tau_true,
                    'n': sc['n'],
                    'fpr': sc['fpr'],
                    'fnr': sc['fnr'],
                    'accuracy': sc['accuracy'],
                    'kappa_s': kappa_cm,
                    'delta_s_fpr': sc['fpr'] - g['fpr'],
                    'delta_s_fnr': sc['fnr'] - g['fnr'],
                    'mean_bias_naive': mean_bias_naive,
                    'mean_bias_global': mean_bias_global,
                    'mean_bias_echte': mean_bias_echte,
                    'rmse_naive': rmse_naive,
                    'rmse_global': rmse_global,
                    'rmse_echte': rmse_echte,
                    'sign_reversal_naive': sign_reversal_naive,
                    'sign_reversal_global': sign_reversal_global,
                    'sign_reversal_echte': sign_reversal_echte,
                    'bias_amplified_by_global': bias_amplified,
                })

    return pd.DataFrame(summary_rows)


def generate_report(summary_df, cm_data):
    lines = []
    lines.append("# Stance Detection (TweetEval) — EC-HTE Validation Report\n")

    lines.append("## Dataset\n")
    lines.append("- **Source**: TweetEval stance detection (cardiffnlp/tweet_eval)")
    lines.append("- **Task**: Stance toward target (favor / against / none)")
    lines.append("- **Binary collapse**: favor (Z=1) vs not-favor (Z=0)")
    lines.append("- **Subgroups (K=5)**: stance targets — Atheism, Climate Change, Feminist Movement, Hillary Clinton, Abortion")
    lines.append(f"- **Synthetic CATE**: τ(favor) = {TAU_FAVOR}, τ(not_favor) = {TAU_AGAINST}")
    lines.append(f"- **MC replications**: {N_MC}")
    lines.append("")

    lines.append("## LLMs\n")
    llm_names = [k for k in cm_data if k != '_meta']
    for llm in llm_names:
        src = cm_data.get('_meta', {}).get(llm, 'real annotations')
        lines.append(f"- **{llm}**: {src}")
    lines.append("")

    # Per-LLM confusion matrix profiles
    lines.append("## Per-Subgroup Confusion Matrix Profiles\n")
    for llm in llm_names:
        lines.append(f"### {llm}\n")
        lines.append("| Target | n | Prev(favor) | FPR | FNR | Accuracy | κ(C_s) | δ_FPR | δ_FNR |")
        lines.append("|--------|---|-------------|-----|-----|----------|--------|-------|-------|")
        g = cm_data[llm]['_global']
        for t in TARGETS:
            tname = TARGET_NAMES[t]
            sc = cm_data[llm][tname]
            denom = 1 - sc['fpr'] - sc['fnr']
            kappa = 1.0 / denom if abs(denom) > 1e-10 else float('inf')
            d_fpr = sc['fpr'] - g['fpr']
            d_fnr = sc['fnr'] - g['fnr']
            lines.append(
                f"| {tname} | {sc['n']} | {sc['prevalence_favor']:.3f} | "
                f"{sc['fpr']:.4f} | {sc['fnr']:.4f} | {sc['accuracy']:.4f} | "
                f"{kappa:.2f} | {d_fpr:+.4f} | {d_fnr:+.4f} |"
            )
        lines.append(f"\n**Global**: FPR={g['fpr']:.4f}, FNR={g['fnr']:.4f}, Acc={g['accuracy']:.4f}\n")

    # Bias comparison table
    lines.append("## EC-HTE Results: Bias Comparison\n")
    lines.append("| LLM | Target | z | τ_true | Bias(naive) | Bias(global) | Bias(EC-HTE) | SignRev(naive) | SignRev(global) | BiasAmp |")
    lines.append("|-----|--------|---|--------|------------|-------------|-------------|---------------|----------------|---------|")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['llm']} | {row['subgroup']} | {row['z_label']} | "
            f"{row['tau_true']:.3f} | {row['mean_bias_naive']:+.4f} | "
            f"{row['mean_bias_global']:+.4f} | {row['mean_bias_echte']:+.4f} | "
            f"{'YES' if row['sign_reversal_naive'] else 'no'} | "
            f"{'YES' if row['sign_reversal_global'] else 'no'} | "
            f"{'YES' if row['bias_amplified_by_global'] else 'no'} |"
        )
    lines.append("")

    # RMSE table
    lines.append("## RMSE Comparison\n")
    lines.append("| LLM | Target | z | RMSE(naive) | RMSE(global) | RMSE(EC-HTE) | Improvement |")
    lines.append("|-----|--------|---|------------|-------------|-------------|------------|")
    for _, row in summary_df.iterrows():
        improvement = (row['rmse_naive'] - row['rmse_echte']) / row['rmse_naive'] * 100 if row['rmse_naive'] > 0 else 0
        lines.append(
            f"| {row['llm']} | {row['subgroup']} | {row['z_label']} | "
            f"{row['rmse_naive']:.4f} | {row['rmse_global']:.4f} | "
            f"{row['rmse_echte']:.4f} | {improvement:+.1f}% |"
        )
    lines.append("")

    # Key numbers
    n_total = len(summary_df)
    n_sign_rev_naive = summary_df['sign_reversal_naive'].sum()
    n_sign_rev_global = summary_df['sign_reversal_global'].sum()
    n_sign_rev_echte = summary_df['sign_reversal_echte'].sum()
    n_bias_amp = summary_df['bias_amplified_by_global'].sum()
    kappa_min = summary_df['kappa_s'].min()
    kappa_max = summary_df['kappa_s'].max()
    mean_rmse_naive = summary_df['rmse_naive'].mean()
    mean_rmse_global = summary_df['rmse_global'].mean()
    mean_rmse_echte = summary_df['rmse_echte'].mean()

    ss_res = (summary_df['mean_bias_echte'] ** 2).sum()
    ss_tot = (summary_df['mean_bias_naive'] ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    lines.append("## Key Numbers\n")
    lines.append(f"- **Sign reversal (naive)**: {n_sign_rev_naive}/{n_total} ({100*n_sign_rev_naive/n_total:.1f}%)")
    lines.append(f"- **Sign reversal (global correction)**: {n_sign_rev_global}/{n_total} ({100*n_sign_rev_global/n_total:.1f}%)")
    lines.append(f"- **Sign reversal (EC-HTE)**: {n_sign_rev_echte}/{n_total} ({100*n_sign_rev_echte/n_total:.1f}%)")
    lines.append(f"- **Bias amplification by global correction**: {n_bias_amp}/{n_total} ({100*n_bias_amp/n_total:.1f}%)")
    lines.append(f"- **κ range**: [{kappa_min:.2f}, {kappa_max:.2f}]")
    lines.append(f"- **Mean RMSE**: naive={mean_rmse_naive:.4f}, global={mean_rmse_global:.4f}, EC-HTE={mean_rmse_echte:.4f}")
    lines.append(f"- **R² (bias reduction EC-HTE vs naive)**: {r2:.4f}")
    lines.append("")

    lines.append("## Takeaways\n")
    if n_sign_rev_global > 0:
        lines.append(f"1. **Sign reversal confirmed**: Global correction causes sign reversal in {n_sign_rev_global}/{n_total} subgroup-z combinations.")
    else:
        lines.append(f"1. **No sign reversal at current CATE gap**: τ(favor)−τ(not_favor) = {TAU_FAVOR-TAU_AGAINST:.2f} is large enough to prevent sign flips.")
    lines.append(f"2. **Bias amplification**: Global correction amplifies bias in {n_bias_amp}/{n_total} ({100*n_bias_amp/n_total:.1f}%) of cases.")
    lines.append(f"3. **EC-HTE improvement**: Mean RMSE: {mean_rmse_naive:.4f} → {mean_rmse_echte:.4f} ({100*(mean_rmse_naive-mean_rmse_echte)/mean_rmse_naive:.1f}% reduction).")
    lines.append(f"4. **Heterogeneity**: κ ranges [{kappa_min:.2f}, {kappa_max:.2f}] across stance targets, showing topic-dependent LLM error structure.")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment B: Stance Detection — EC-HTE Validation")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading TweetEval stance data...")
    df_all = load_stance_data()
    df = sample_data(df_all)
    print(f"  Sampled: {len(df)} total across {len(TARGETS)} targets")

    # Load GPT-4o annotations from checkpoint
    print("\n[2/5] Loading GPT-4o annotations from checkpoint...")
    with open(CHECKPOINT) as f:
        cached = json.load(f)
    print(f"  GPT-4o: {len(cached['GPT-4o'])} annotations")

    # Compute real confusion matrix for GPT-4o
    cm_gpt4o = compute_cm_from_annotations(df, cached['GPT-4o'])

    # Simulate 2 additional LLM profiles
    print("\n[3/5] Simulating GPT-4o-mini and GPT-3.5-Turbo error profiles...")
    rng = np.random.RandomState(SEED + 1)

    # GPT-4o-mini: higher FPR, lower FNR (aggressive positive classifier)
    preds_mini = simulate_llm_annotations(df, cm_gpt4o, 'GPT-4o-mini',
                                          {'fpr_mult': 1.8, 'fnr_mult': 0.6, 'fpr_shift': 0.05}, rng)
    cm_mini = compute_cm_from_annotations(df, preds_mini)

    # GPT-3.5-Turbo: higher both FPR and FNR (generally worse)
    preds_35 = simulate_llm_annotations(df, cm_gpt4o, 'GPT-3.5-Turbo',
                                        {'fpr_mult': 1.4, 'fnr_mult': 1.6, 'fpr_shift': 0.08}, rng)
    cm_35 = compute_cm_from_annotations(df, preds_35)

    # Combine all CMs
    cm_data = {
        'GPT-4o': cm_gpt4o,
        'GPT-4o-mini': cm_mini,
        'GPT-3.5-Turbo': cm_35,
        '_meta': {
            'GPT-4o': 'real annotations via API',
            'GPT-4o-mini': 'simulated (FPR×1.8, FNR×0.6, +0.05 shift — aggressive classifier profile)',
            'GPT-3.5-Turbo': 'simulated (FPR×1.4, FNR×1.6, +0.08 shift — weaker classifier profile)',
        },
    }

    # Print CM summary
    for llm in ['GPT-4o', 'GPT-4o-mini', 'GPT-3.5-Turbo']:
        g = cm_data[llm]['_global']
        print(f"  {llm}: global FPR={g['fpr']:.4f}, FNR={g['fnr']:.4f}, Acc={g['accuracy']:.4f}")
        for t in TARGETS:
            tname = TARGET_NAMES[t]
            sc = cm_data[llm][tname]
            print(f"    {tname}: FPR={sc['fpr']:.4f}, FNR={sc['fnr']:.4f}, Acc={sc['accuracy']:.4f}")

    # Save CMs
    cm_path = RESULTS / "stance_confusion_matrices.json"
    cm_save = {}
    for k, v in cm_data.items():
        if k == '_meta':
            cm_save[k] = v
        else:
            cm_save[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    cm_save[k][sk] = {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                                       for kk, vv in sv.items()}
    with open(cm_path, 'w') as f:
        json.dump(cm_save, f, indent=2, default=str)
    print(f"\n  Saved CMs to {cm_path}")

    # MC simulation
    print("\n[4/5] Running MC simulation (N_MC={})...".format(N_MC))
    mc_df = run_mc_simulation(cm_data)
    summary_df = analyze_results(mc_df, cm_data)
    print(f"  Done: {len(summary_df)} summary rows")

    # Save outputs
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    csv_path = ARTIFACTS / "stance_analysis.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[5/5] Saved {csv_path}")

    report = generate_report(summary_df, cm_data)
    report_path = ARTIFACTS / "stance_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")

    # Key numbers
    n_total = len(summary_df)
    print("\n" + "=" * 60)
    print("KEY NUMBERS")
    print("=" * 60)
    print(f"Sign reversal (naive):  {summary_df['sign_reversal_naive'].sum()}/{n_total}")
    print(f"Sign reversal (global): {summary_df['sign_reversal_global'].sum()}/{n_total}")
    print(f"Sign reversal (EC-HTE): {summary_df['sign_reversal_echte'].sum()}/{n_total}")
    print(f"Bias amplification:     {summary_df['bias_amplified_by_global'].sum()}/{n_total}")
    print(f"κ range:                [{summary_df['kappa_s'].min():.2f}, {summary_df['kappa_s'].max():.2f}]")
    print(f"Mean RMSE naive:        {summary_df['rmse_naive'].mean():.4f}")
    print(f"Mean RMSE global:       {summary_df['rmse_global'].mean():.4f}")
    print(f"Mean RMSE EC-HTE:       {summary_df['rmse_echte'].mean():.4f}")

    ss_res = (summary_df['mean_bias_echte'] ** 2).sum()
    ss_tot = (summary_df['mean_bias_naive'] ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"R² (EC-HTE vs naive):   {r2:.4f}")


if __name__ == '__main__':
    main()
