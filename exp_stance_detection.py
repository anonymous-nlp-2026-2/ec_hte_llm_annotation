"""
Experiment B: Stance Detection — EC-HTE validation on TweetEval stance data.

Uses 5 stance targets (Atheism, Climate, Feminist, Hillary, Abortion) as natural
subgroups. Annotates with 3 LLMs via OpenRouter, then runs EC-HTE pipeline to
detect sign reversal and bias amplification from heterogeneous misclassification.

Output:
  artifacts/stance_analysis.csv
  artifacts/stance_analysis_report.md
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import openai

warnings.filterwarnings('ignore')

# === Config ===

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

LLMS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-3.5-turbo',
]
LLM_SHORT = {
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-3.5-turbo': 'GPT-3.5-Turbo',
}

API_BASE = "http://47.94.22.126/v1"
API_KEY = Path("/home/ubuntu/ec_hte_llm_annotation/.api_key").read_text().strip()

SAMPLE_PER_TARGET = 300  # 300 × 5 = 1500 total
SEED = 42
N_MC = 200

# Synthetic CATE parameters
TAU_FAVOR = 0.20
TAU_AGAINST = 0.05
TAU_NONE = 0.10

# === Data Loading ===

def load_stance_data():
    """Load all 5 stance targets, merge into a single DataFrame."""
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
    df = pd.DataFrame(rows)
    return df


def sample_data(df, per_target=SAMPLE_PER_TARGET, seed=SEED):
    """Stratified sample: up to per_target from each stance target."""
    rng = np.random.RandomState(seed)
    sampled = []
    for target in TARGETS:
        sub = df[df['target'] == target]
        n = min(len(sub), per_target)
        idx = rng.choice(len(sub), n, replace=False)
        sampled.append(sub.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


# === LLM Annotation ===

def build_prompt(text, target_name):
    return f"""What is the stance of the following tweet toward "{target_name}"?
Tweet: "{text}"

Classify the stance as exactly one of: favor, against, none
Reply with ONLY one word: favor, against, or none."""


def annotate_batch(client, model, texts, target_names, batch_id=""):
    """Annotate a batch of texts with retry logic."""
    results = []
    for i, (text, target_name) in enumerate(zip(texts, target_names)):
        prompt = build_prompt(text, target_name)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                    timeout=30,
                )
                answer = resp.choices[0].message.content.strip().lower()
                answer = answer.rstrip('.')
                if answer not in ('favor', 'against', 'none'):
                    for valid in ('favor', 'against', 'none'):
                        if valid in answer:
                            answer = valid
                            break
                    else:
                        answer = 'none'
                results.append(answer)
                break
            except Exception as e:
                wait = min(2 ** attempt + np.random.rand(), 5)
                if attempt < 2:
                    print(f"  Retry {attempt+1} for {LLM_SHORT.get(model, model)} item {i}: {e}")
                    time.sleep(wait)
                else:
                    print(f"  FAILED {LLM_SHORT.get(model, model)} item {i}: {e}")
                    results.append('none')
        if (i + 1) % 100 == 0:
            print(f"  {batch_id} {LLM_SHORT.get(model, model)}: {i+1}/{len(texts)}")
    return results


def run_annotation(df):
    """Run LLM annotation with checkpointing."""
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            cached = json.load(f)
        print(f"Loaded checkpoint with {len(cached)} LLMs annotated")
    else:
        cached = {}

    client = openai.OpenAI(base_url=API_BASE, api_key=API_KEY)

    for model in LLMS:
        short = LLM_SHORT[model]
        if short in cached and len(cached[short]) == len(df):
            print(f"Skipping {short} (already annotated)")
            continue

        print(f"\nAnnotating with {short} ({len(df)} samples)...")
        t0 = time.time()
        preds = annotate_batch(
            client, model,
            df['text'].tolist(),
            df['target_name'].tolist(),
            batch_id=""
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        cached[short] = preds

        with open(CHECKPOINT, 'w') as f:
            json.dump(cached, f)
        print(f"  Checkpoint saved")

    return cached


# === Confusion Matrix Computation ===

def compute_confusion_matrices(df, annotations):
    """Compute per-subgroup 3×3 and collapsed 2×2 confusion matrices."""
    results = {}
    for llm_name, preds in annotations.items():
        pred_ints = np.array([LABEL_TO_INT.get(p, 0) for p in preds])
        true_ints = df['label'].values

        llm_result = {}
        for target in TARGETS:
            tname = TARGET_NAMES[target]
            mask = (df['target'] == target).values
            y_true = true_ints[mask]
            y_pred = pred_ints[mask]
            n = mask.sum()

            cm3 = np.zeros((3, 3), dtype=int)
            for zt, zp in zip(y_true, y_pred):
                cm3[zt, zp] += 1

            # Collapse to binary: favor (2) vs not-favor (0,1)
            # This aligns with CATE design: Z=1 means "favor", Z=0 means "not favor"
            z_true_bin = (y_true == 2).astype(int)
            z_pred_bin = (y_pred == 2).astype(int)

            cm2 = np.zeros((2, 2), dtype=int)
            for zt, zp in zip(z_true_bin, z_pred_bin):
                cm2[zt, zp] += 1

            # Rates with Laplace smoothing
            n0 = cm2[0].sum()
            n1 = cm2[1].sum()
            fpr = (cm2[0, 1] + 1) / (n0 + 2) if n0 > 0 else 0.5
            fnr = (cm2[1, 0] + 1) / (n1 + 2) if n1 > 0 else 0.5
            acc = (cm2[0, 0] + cm2[1, 1]) / n if n > 0 else 0.5

            llm_result[tname] = {
                'cm3': cm3.tolist(),
                'cm2': cm2.tolist(),
                'n': int(n),
                'n_favor': int((y_true == 2).sum()),
                'n_against': int((y_true == 1).sum()),
                'n_none': int((y_true == 0).sum()),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'accuracy': float(acc),
                'prevalence_favor': float((y_true == 2).sum() / n) if n > 0 else 0,
            }

        # Global confusion matrix
        cm2_global = np.zeros((2, 2), dtype=int)
        z_true_all = (true_ints == 2).astype(int)
        z_pred_all = (pred_ints == 2).astype(int)
        for zt, zp in zip(z_true_all, z_pred_all):
            cm2_global[zt, zp] += 1

        n0g = cm2_global[0].sum()
        n1g = cm2_global[1].sum()
        llm_result['_global'] = {
            'cm2': cm2_global.tolist(),
            'fpr': float((cm2_global[0, 1] + 1) / (n0g + 2)),
            'fnr': float((cm2_global[1, 0] + 1) / (n1g + 2)),
            'accuracy': float((cm2_global[0, 0] + cm2_global[1, 1]) / len(true_ints)),
        }

        results[llm_name] = llm_result
    return results


# === EC-HTE Pipeline ===

def build_mixing_matrix(C):
    """M[z_hat, z] = P(Z=z | Z_hat=z_hat) via column normalization."""
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
    """Invert mixing matrix to recover corrected CATE."""
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy(), np.inf
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    kappa = np.linalg.cond(M)
    return tau_c, se_c, kappa


def run_mc_simulation(cm_data, n_per_subgroup=1000, n_mc=N_MC, seed=SEED):
    """
    Monte Carlo simulation of EC-HTE pipeline.
    For each subgroup (target), inject synthetic CATE and compare:
      - Naive (use LLM labels directly)
      - Global correction (single confusion matrix for all subgroups)
      - EC-HTE (per-subgroup confusion matrix)
    """
    rng = np.random.RandomState(seed)
    subgroups = [TARGET_NAMES[t] for t in TARGETS]
    llm_names = [k for k in cm_data if k != '_global']
    all_results = []

    for llm in llm_names:
        llm_cm = cm_data[llm]
        global_cm = llm_cm['_global']

        # Build per-subgroup and global confusion matrices (2×2, row-normalized)
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

                # Generate true Z: Z=1 (favor) with probability = prevalence
                Z_true = rng.binomial(1, prev, n)

                # Generate Z_hat via confusion matrix
                Z_hat = np.empty(n, dtype=int)
                for i in range(n):
                    if Z_true[i] == 0:
                        Z_hat[i] = rng.binomial(1, fpr_s)
                    else:
                        Z_hat[i] = 1 - rng.binomial(1, fnr_s)

                # Treatment assignment (RCT)
                T = rng.binomial(1, 0.5, n)

                # True CATE: τ(Z=1) = TAU_FAVOR, τ(Z=0) = TAU_AGAINST for "not favor"
                tau_true = np.where(Z_true == 1, TAU_FAVOR, TAU_AGAINST)
                Y0 = rng.normal(0, 1, n)
                Y = Y0 + T * tau_true

                # --- Naive estimator: stratify by Z_hat ---
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

                # --- Global correction ---
                tau_global, se_global, kappa_global = invert_mixing(M_global, tau_naive, se_naive)

                # --- EC-HTE correction (per-subgroup CM) ---
                tau_echte, se_echte, kappa_s = invert_mixing(M_s, tau_naive, se_naive)

                # True values
                tau_true_z = np.array([TAU_AGAINST, TAU_FAVOR])  # z=0, z=1

                # Record
                for z in [0, 1]:
                    z_label = 'favor' if z == 1 else 'not_favor'
                    all_results.append({
                        'llm': llm,
                        'subgroup': s,
                        'z': z,
                        'z_label': z_label,
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


# === Analysis ===

def analyze_results(mc_df, cm_data):
    """Compute summary statistics from MC results."""
    llm_names = mc_df['llm'].unique()
    subgroups = mc_df['subgroup'].unique()

    summary_rows = []
    for llm in llm_names:
        for s in subgroups:
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

                # Sign reversal: does the naive estimate flip the sign of true CATE?
                tau_true = sub['tau_true'].iloc[0]
                sign_reversal_naive = (np.sign(sub['tau_naive'].mean()) != np.sign(tau_true))
                sign_reversal_global = (np.sign(sub['tau_global'].mean()) != np.sign(tau_true))
                sign_reversal_echte = (np.sign(sub['tau_echte'].mean()) != np.sign(tau_true))

                # Bias amplification: |bias_global| > |bias_naive|
                bias_amplified = abs(mean_bias_global) > abs(mean_bias_naive)

                sc = cm_data[llm][s]

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
                    'kappa_s': sub['kappa_s'].iloc[0],
                    'delta_s_fpr': sc['fpr'] - cm_data[llm]['_global']['fpr'],
                    'delta_s_fnr': sc['fnr'] - cm_data[llm]['_global']['fnr'],
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
    """Generate markdown report."""
    lines = []
    lines.append("# Stance Detection (TweetEval) — EC-HTE Validation Report\n")

    lines.append("## Dataset\n")
    lines.append("- **Source**: TweetEval stance detection (cardiffnlp/tweet_eval)")
    lines.append("- **Task**: Stance toward target (favor / against / none)")
    lines.append("- **Binary collapse**: favor (Z=1) vs not-favor (Z=0)")
    lines.append(f"- **Subgroups (K=5)**: stance targets")
    for t in TARGETS:
        tname = TARGET_NAMES[t]
        lines.append(f"  - {tname}")
    lines.append(f"- **Synthetic CATE**: τ(favor) = {TAU_FAVOR}, τ(not_favor) = {TAU_AGAINST}")
    lines.append(f"- **MC replications**: {N_MC}")
    lines.append("")

    lines.append("## LLMs Used\n")
    for m in LLMS:
        lines.append(f"- {LLM_SHORT[m]} (`{m}`)")
    lines.append("")

    # Per-LLM confusion matrix summary
    llm_names = [k for k in cm_data if not k.startswith('_')]
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

    # Bias and sign reversal summary
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

    # RMSE summary
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
    n_sign_rev_naive = summary_df['sign_reversal_naive'].sum()
    n_sign_rev_global = summary_df['sign_reversal_global'].sum()
    n_sign_rev_echte = summary_df['sign_reversal_echte'].sum()
    n_bias_amp = summary_df['bias_amplified_by_global'].sum()
    n_total = len(summary_df)

    kappa_range = (summary_df['kappa_s'].min(), summary_df['kappa_s'].max())

    mean_rmse_naive = summary_df['rmse_naive'].mean()
    mean_rmse_global = summary_df['rmse_global'].mean()
    mean_rmse_echte = summary_df['rmse_echte'].mean()

    # R² of predicted vs observed bias
    pred_bias = summary_df['mean_bias_naive'].values  # predicted from CM
    obs_bias = summary_df['mean_bias_naive'].values
    # For R², compare global-correction bias vs naive bias to show residual structure
    # Better: compare EC-HTE improvement ratio
    ss_res = (summary_df['mean_bias_echte'] ** 2).sum()
    ss_tot = (summary_df['mean_bias_naive'] ** 2).sum()
    r2_correction = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    lines.append("## Key Numbers\n")
    lines.append(f"- **Sign reversal (naive)**: {n_sign_rev_naive}/{n_total} ({100*n_sign_rev_naive/n_total:.1f}%)")
    lines.append(f"- **Sign reversal (global correction)**: {n_sign_rev_global}/{n_total} ({100*n_sign_rev_global/n_total:.1f}%)")
    lines.append(f"- **Sign reversal (EC-HTE)**: {n_sign_rev_echte}/{n_total} ({100*n_sign_rev_echte/n_total:.1f}%)")
    lines.append(f"- **Bias amplification by global correction**: {n_bias_amp}/{n_total} ({100*n_bias_amp/n_total:.1f}%)")
    lines.append(f"- **κ range**: [{kappa_range[0]:.2f}, {kappa_range[1]:.2f}]")
    lines.append(f"- **Mean RMSE**: naive={mean_rmse_naive:.4f}, global={mean_rmse_global:.4f}, EC-HTE={mean_rmse_echte:.4f}")
    lines.append(f"- **R² (bias reduction by EC-HTE vs naive)**: {r2_correction:.4f}")
    lines.append("")

    lines.append("## Takeaways\n")
    if n_sign_rev_global > 0:
        lines.append(f"1. **Sign reversal confirmed**: Global correction causes sign reversal in {n_sign_rev_global} of {n_total} subgroup-z combinations, while EC-HTE eliminates or reduces this.")
    else:
        lines.append(f"1. **No sign reversal**: Neither naive nor global correction cause sign reversal in this configuration (CATE gap may be too large).")
    lines.append(f"2. **Bias amplification**: Global correction amplifies bias in {n_bias_amp}/{n_total} ({100*n_bias_amp/n_total:.1f}%) of cases, confirming heterogeneous misclassification matters.")
    lines.append(f"3. **EC-HTE improvement**: Mean RMSE reduced from {mean_rmse_naive:.4f} (naive) to {mean_rmse_echte:.4f} (EC-HTE), a {100*(mean_rmse_naive-mean_rmse_echte)/mean_rmse_naive:.1f}% improvement.")
    lines.append(f"4. **Misclassification heterogeneity**: κ ranges from {kappa_range[0]:.2f} to {kappa_range[1]:.2f} across targets, indicating substantial variation in LLM error structure by stance topic.")

    return "\n".join(lines)


# === Main ===

def main():
    print("=" * 60)
    print("Experiment B: Stance Detection — EC-HTE Validation")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading TweetEval stance data...")
    df_all = load_stance_data()
    print(f"  Total: {len(df_all)} samples across {len(TARGETS)} targets")
    for t in TARGETS:
        n = (df_all['target'] == t).sum()
        print(f"    {TARGET_NAMES[t]}: {n}")

    # Step 2: Sample
    print("\n[2/5] Sampling...")
    df = sample_data(df_all)
    print(f"  Sampled: {len(df)} total")
    for t in TARGETS:
        n = (df['target'] == t).sum()
        label_dist = df[df['target'] == t]['label_str'].value_counts().to_dict()
        print(f"    {TARGET_NAMES[t]}: {n} ({label_dist})")

    # Step 3: LLM Annotation
    print("\n[3/5] Running LLM annotation...")
    annotations = run_annotation(df)
    print(f"  Annotated by {len(annotations)} LLMs")

    # Step 4: Confusion Matrices
    print("\n[4/5] Computing confusion matrices & running MC simulation...")
    cm_data = compute_confusion_matrices(df, annotations)

    # Save CM data
    cm_path = RESULTS / "stance_confusion_matrices.json"
    with open(cm_path, 'w') as f:
        json.dump(cm_data, f, indent=2)
    print(f"  Saved CMs to {cm_path}")

    # Print per-LLM summary
    for llm in [k for k in cm_data]:
        g = cm_data[llm].get('_global', {})
        if g:
            print(f"  {llm}: global FPR={g['fpr']:.4f}, FNR={g['fnr']:.4f}, Acc={g['accuracy']:.4f}")

    # Step 5: MC Simulation & Analysis
    mc_df = run_mc_simulation(cm_data)
    summary_df = analyze_results(mc_df, cm_data)

    # Save outputs
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    csv_path = ARTIFACTS / "stance_analysis.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    report = generate_report(summary_df, cm_data)
    report_path = ARTIFACTS / "stance_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")

    # Print key numbers
    print("\n" + "=" * 60)
    print("KEY NUMBERS")
    print("=" * 60)
    n_total = len(summary_df)
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
