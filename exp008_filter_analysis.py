"""exp-008 analysis with small subgroup filtering per Director instruction.
Excludes subgroups with n<10, recalculates Δπ and Spearman ρ."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANNOTATIONS_CSV = '/home/ubuntu/ec_hte_llm_annotation/results/exp008_multi_llm_annotations.csv'
RESULTS_DIR = '/home/ubuntu/ec_hte_llm_annotation/results'
MODELS_5 = ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini']
MIN_N = 10

def main():
    df = pd.read_csv(ANNOTATIONS_CSV)
    df['true_z'] = (df['true_label'] == 2).astype(int)
    df['subgroup'] = df['has_hashtag'].map({True: 'True', False: 'False'}) + '_' + df['text_length_bin']

    # Check for gpt-5.x results
    all_models = list(MODELS_5)
    try:
        gpt5x_model = open(f'{RESULTS_DIR}/exp008_gpt5x_model_used.txt').read().strip()
        ckpt = f'{RESULTS_DIR}/exp008_checkpoint_{gpt5x_model}.json'
        labels = json.load(open(ckpt))
        if len(labels) == len(df):
            df[gpt5x_model] = labels
            all_models.append(gpt5x_model)
            print(f"6 models loaded (including {gpt5x_model})")
        else:
            print(f"GPT-5.x incomplete ({len(labels)}/{len(df)}), using 5 models")
    except:
        print("5 models loaded (no GPT-5.x yet)")

    for m in all_models:
        df[f'llm_z_{m}'] = (df[m] == 2).astype(int)

    # Subgroup sizes
    sg_sizes = df.groupby('subgroup').size()
    print("\n## Subgroup Sizes")
    for sg in sorted(sg_sizes.index):
        print(f"  {sg}: n={sg_sizes[sg]}")

    small_sgs = sorted([sg for sg in sg_sizes.index if sg_sizes[sg] < MIN_N])
    valid_sgs = sorted([sg for sg in sg_sizes.index if sg_sizes[sg] >= MIN_N])
    print(f"\nExcluded (n<{MIN_N}): {small_sgs}")
    print(f"Retained: {valid_sgs}")

    # Compute misclass rates
    full_table = {}
    for m in all_models:
        full_table[m] = {}
        for sg in sorted(sg_sizes.index):
            sg_data = df[df['subgroup'] == sg]
            full_table[m][sg] = (sg_data['true_z'] != sg_data[f'llm_z_{m}']).mean()

    # Full table
    print("\n## Full Misclassification Rates")
    header = "| subgroup | n |" + "|".join(f" {m} " for m in all_models) + "|"
    print(header)
    print("|" + "---|" * (len(all_models) + 2))
    for sg in sorted(sg_sizes.index):
        row = f"| {sg} | {sg_sizes[sg]} |"
        for m in all_models:
            row += f" {full_table[m][sg]:.4f} |"
        print(row)

    # Filtered Δπ
    print(f"\n## Effective Δπ (excluding n<{MIN_N} subgroups)")
    print(f"Excluded: {', '.join(f'{sg} (n={sg_sizes[sg]})' for sg in small_sgs)}")

    delta_pi_full = {}
    delta_pi_filtered = {}
    for m in all_models:
        rates_all = [full_table[m][sg] for sg in sorted(sg_sizes.index)]
        rates_filtered = [full_table[m][sg] for sg in valid_sgs]
        delta_pi_full[m] = max(rates_all) - min(rates_all)
        delta_pi_filtered[m] = max(rates_filtered) - min(rates_filtered)

    print("\n| Model | Δπ (all) | Δπ (effective, n≥10) | Δπ > 0.05? |")
    print("|-------|----------|----------------------|------------|")
    for m in all_models:
        flag = "YES" if delta_pi_filtered[m] > 0.05 else "NO"
        print(f"| {m} | {delta_pi_full[m]:.4f} | {delta_pi_filtered[m]:.4f} | {flag} |")

    all_above = all(delta_pi_filtered[m] > 0.05 for m in all_models)
    print(f"\nAll models effective Δπ > 0.05: {'YES' if all_above else 'NO'}")

    # Spearman ρ on filtered subgroups
    print(f"\n## Spearman ρ (n≥{MIN_N} subgroups only)")
    model_rates = {}
    for m in all_models:
        model_rates[m] = [full_table[m][sg] for sg in valid_sgs]

    for i in range(len(all_models)):
        for j in range(i + 1, len(all_models)):
            m1, m2 = all_models[i], all_models[j]
            rho, p = spearmanr(model_rates[m1], model_rates[m2])
            sig = "*" if p < 0.05 else ""
            print(f"  {m1} vs {m2}: ρ={rho:.3f} (p={p:.4f}){sig}")

    # Worst subgroup per model (filtered)
    print(f"\n## Worst Subgroup per Model (n≥{MIN_N})")
    for m in all_models:
        worst_sg = max(valid_sgs, key=lambda sg: full_table[m][sg])
        print(f"  {m}: {worst_sg} ({full_table[m][worst_sg]:.4f})")

    # FPR / FNR per filtered subgroup
    print(f"\n## FPR and FNR (n≥{MIN_N} subgroups)")
    for m in all_models:
        print(f"\n### {m}")
        print("| subgroup | n | FPR | FNR |")
        print("|---|---|---|---|")
        for sg in valid_sgs:
            sg_data = df[df['subgroup'] == sg]
            tp = ((sg_data['true_z'] == 1) & (sg_data[f'llm_z_{m}'] == 1)).sum()
            fp = ((sg_data['true_z'] == 0) & (sg_data[f'llm_z_{m}'] == 1)).sum()
            fn = ((sg_data['true_z'] == 1) & (sg_data[f'llm_z_{m}'] == 0)).sum()
            tn = ((sg_data['true_z'] == 0) & (sg_data[f'llm_z_{m}'] == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            print(f"| {sg} | {sg_sizes[sg]} | {fpr:.4f} | {fnr:.4f} |")

    # Save markdown
    out = f'{RESULTS_DIR}/exp008_filtered_analysis.md'
    with open(out, 'w') as f:
        f.write("# exp-008: Multi-LLM Analysis (Filtered, n≥10)\n\n")
        f.write(f"**Models**: {', '.join(all_models)} ({len(all_models)} total)\n")
        f.write(f"**Small subgroups excluded**: {', '.join(f'{sg} (n={sg_sizes[sg]})' for sg in small_sgs)}\n")
        f.write(f"**Retained subgroups**: {', '.join(f'{sg} (n={sg_sizes[sg]})' for sg in valid_sgs)}\n\n")

        f.write("## Effective Δπ\n\n")
        f.write("| Model | Δπ (all subgroups) | Δπ (effective, n≥10) | Δπ > 0.05? |\n")
        f.write("|-------|-------------------|----------------------|------------|\n")
        for m in all_models:
            flag = "YES" if delta_pi_filtered[m] > 0.05 else "NO"
            f.write(f"| {m} | {delta_pi_full[m]:.4f} | {delta_pi_filtered[m]:.4f} | {flag} |\n")
        f.write(f"\n**All models effective Δπ > 0.05: {'YES' if all_above else 'NO'}**\n\n")

        f.write("## Per-Model Per-Subgroup Misclassification (n≥10)\n\n")
        f.write("| subgroup | n |" + "|".join(f" {m} " for m in all_models) + "|\n")
        f.write("|" + "---|" * (len(all_models) + 2) + "\n")
        for sg in valid_sgs:
            row = f"| {sg} | {sg_sizes[sg]} |"
            for m in all_models:
                row += f" {full_table[m][sg]:.4f} |"
            f.write(row + "\n")

        f.write(f"\n## Spearman ρ (n≥{MIN_N} subgroups)\n\n")
        for i in range(len(all_models)):
            for j in range(i + 1, len(all_models)):
                m1, m2 = all_models[i], all_models[j]
                rho, p = spearmanr(model_rates[m1], model_rates[m2])
                f.write(f"- {m1} vs {m2}: ρ={rho:.3f} (p={p:.4f})\n")

        f.write(f"\n## Worst Subgroup per Model (n≥{MIN_N})\n\n")
        for m in all_models:
            worst_sg = max(valid_sgs, key=lambda sg: full_table[m][sg])
            f.write(f"- {m}: {worst_sg} ({full_table[m][worst_sg]:.4f})\n")

    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
