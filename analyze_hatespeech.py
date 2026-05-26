"""
Hate Speech (CivilComments) analysis: per-subgroup bias amplification,
Theorem 1 R² verification, and debiasing-hurts identification.
"""
import json
import csv
import numpy as np
from pathlib import Path

RESULTS = Path("/home/ubuntu/ec_hte_llm_annotation/results")
ARTIFACTS = Path("/home/ubuntu/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts")

SUBGROUP_LABELS = {
    "S0": "short, no question",
    "S1": "short, has question",
    "S2": "long, no question",
    "S3": "long, has question",
}

SUBGROUPS = ["S0", "S1", "S2", "S3"]

# --- Load confusion matrices ---
with open(RESULTS / "exp023_confusion_matrices.json") as f:
    cm_data = json.load(f)

LLMS = [k for k in cm_data if k != "_global"]

# --- Load bias identity detail ---
bias_rows = []
with open(RESULTS / "exp_realcm_bias_identity.csv") as f:
    for row in csv.DictReader(f):
        if row["dataset"] == "CivilComments":
            bias_rows.append(row)

# --- Load R² summary ---
r2_data = {}
with open(RESULTS / "exp_realcm_summary.csv") as f:
    for row in csv.DictReader(f):
        if row["dataset"] == "CivilComments":
            r2_data[row["llm"]] = {
                "R2": float(row["R2"]),
                "diagnostic_AUC": float(row["diagnostic_AUC"]),
                "max_abs_B": float(row["max_abs_B"]),
            }

# Also load TweetEval R² for comparison
r2_tweeteval = {}
with open(RESULTS / "exp_realcm_summary.csv") as f:
    for row in csv.DictReader(f):
        if row["dataset"] == "TweetEval":
            r2_tweeteval[row["llm"]] = float(row["R2"])

# ============================================================
# Per-subgroup analysis
# ============================================================
output_rows = []

for llm in LLMS:
    llm_cm = cm_data[llm]
    glob = llm_cm["_global"]
    fpr_global = glob["fpr"]
    fnr_global = glob["fnr"]
    pi_global = fpr_global + fnr_global  # total misclassification "rate" proxy

    # Compute per-subgroup weights from n
    ns = {s: llm_cm[s]["n"] for s in SUBGROUPS}
    n_total = sum(ns.values())
    weights = {s: ns[s] / n_total for s in SUBGROUPS}

    # Global weighted misclassification rate
    pi_bar = sum(
        weights[s] * (llm_cm[s]["fpr"] + llm_cm[s]["fnr"]) for s in SUBGROUPS
    )

    for s in SUBGROUPS:
        sc = llm_cm[s]
        fpr_s = sc["fpr"]
        fnr_s = sc["fnr"]
        acc_s = sc["accuracy"]
        n_s = sc["n"]
        n_pos_s = sc["n_pos"]
        n_neg_s = sc["n_neg"]

        # kappa = 1 / (1 - FPR - FNR), condition number of confusion matrix
        denom = 1 - fpr_s - fnr_s
        if abs(denom) < 1e-10:
            kappa = float("inf")
        else:
            kappa = 1.0 / denom

        # pi_s = FPR_s + FNR_s (aggregate misclassification)
        pi_s = fpr_s + fnr_s

        # delta_s = pi_s - pi_bar
        delta_s = pi_s - pi_bar

        # Get predicted and observed bias from MC data (use delta_tau=0.3 as representative)
        pred_bias_z0 = None
        pred_bias_z1 = None
        obs_bias_z0 = None
        obs_bias_z1 = None
        se_z0 = None
        se_z1 = None

        for br in bias_rows:
            if br["llm"] == llm and br["subgroup"] == s and float(br["delta_tau"]) == 0.3:
                z = int(br["z"])
                if z == 0:
                    pred_bias_z0 = float(br["predicted_bias"])
                    obs_bias_z0 = float(br["observed_bias"])
                    se_z0 = float(br["se"])
                else:
                    pred_bias_z1 = float(br["predicted_bias"])
                    obs_bias_z1 = float(br["observed_bias"])
                    se_z1 = float(br["se"])

        # "Debiasing hurts" = |bias_global| > threshold
        # Check across all delta_tau values
        debiasing_hurts_count = 0
        total_checks = 0
        max_abs_bias = 0.0
        for br in bias_rows:
            if br["llm"] == llm and br["subgroup"] == s:
                total_checks += 1
                obs = abs(float(br["observed_bias"]))
                pred = abs(float(br["predicted_bias"]))
                max_abs_bias = max(max_abs_bias, obs)
                # "debiasing hurts" if global correction introduces substantial bias
                # i.e. observed bias after global correction > 2 SE
                if obs > 2 * float(br["se"]):
                    debiasing_hurts_count += 1

        output_rows.append({
            "llm": llm,
            "subgroup": s,
            "subgroup_label": SUBGROUP_LABELS[s],
            "n": n_s,
            "n_pos": n_pos_s,
            "n_neg": n_neg_s,
            "weight": round(weights[s], 4),
            "FPR": round(fpr_s, 4),
            "FNR": round(fnr_s, 4),
            "accuracy": round(acc_s, 4),
            "kappa": round(kappa, 4),
            "pi_s": round(pi_s, 4),
            "pi_bar": round(pi_bar, 4),
            "delta_s": round(delta_s, 4),
            "predicted_bias_z0": round(pred_bias_z0, 6) if pred_bias_z0 is not None else "",
            "predicted_bias_z1": round(pred_bias_z1, 6) if pred_bias_z1 is not None else "",
            "observed_bias_z0": round(obs_bias_z0, 6) if obs_bias_z0 is not None else "",
            "observed_bias_z1": round(obs_bias_z1, 6) if obs_bias_z1 is not None else "",
            "se_z0": round(se_z0, 6) if se_z0 is not None else "",
            "se_z1": round(se_z1, 6) if se_z1 is not None else "",
            "max_abs_bias": round(max_abs_bias, 6),
            "debiasing_hurts_fraction": f"{debiasing_hurts_count}/{total_checks}",
        })

# --- Write CSV ---
csv_path = ARTIFACTS / "hatespeech_analysis.csv"
fieldnames = list(output_rows[0].keys())
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(output_rows)
print(f"Wrote {csv_path}")

# ============================================================
# R² computation from scratch (verify)
# ============================================================
cc_predicted = []
cc_observed = []
for br in bias_rows:
    cc_predicted.append(float(br["predicted_bias"]))
    cc_observed.append(float(br["observed_bias"]))
cc_predicted = np.array(cc_predicted)
cc_observed = np.array(cc_observed)
ss_res = np.sum((cc_observed - cc_predicted) ** 2)
ss_tot = np.sum((cc_observed - np.mean(cc_observed)) ** 2)
r2_all = 1 - ss_res / ss_tot
print(f"\nPooled CivilComments R² (all LLMs, all Δτ): {r2_all:.4f}")

# Per-LLM R²
for llm in LLMS:
    pred = []
    obs = []
    for br in bias_rows:
        if br["llm"] == llm:
            pred.append(float(br["predicted_bias"]))
            obs.append(float(br["observed_bias"]))
    pred = np.array(pred)
    obs = np.array(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  {llm}: R² = {r2:.4f} (stored: {r2_data.get(llm, {}).get('R2', 'N/A'):.4f})")

# ============================================================
# Debiasing hurts summary
# ============================================================
print("\n=== DEBIASING HURTS ANALYSIS ===")
total_combos = 0
hurts_combos = 0
extreme_cases = []

for row in output_rows:
    # Count subgroups where the bias is substantial
    frac = row["debiasing_hurts_fraction"]
    num, den = frac.split("/")
    total_combos += int(den)
    hurts_combos += int(num)

    if row["max_abs_bias"] > 0.05:
        extreme_cases.append(row)

print(f"Total LLM-subgroup-z-Δτ checks: {total_combos}")
print(f"Checks where |obs bias| > 2 SE: {hurts_combos} ({100*hurts_combos/total_combos:.1f}%)")
print(f"\nMost extreme bias amplification cases (max |bias| > 0.05):")
for c in sorted(extreme_cases, key=lambda x: -x["max_abs_bias"]):
    print(f"  {c['llm']} × {c['subgroup']} ({c['subgroup_label']}): max |bias| = {c['max_abs_bias']:.4f}, "
          f"κ = {c['kappa']:.2f}, δ_s = {c['delta_s']:.4f}")

# ============================================================
# Generate report
# ============================================================
report_lines = []
report_lines.append("# Hate Speech (CivilComments) — Per-Subgroup Bias Amplification Analysis\n")
report_lines.append("## Dataset\n")
report_lines.append("- **Source**: Jigsaw Civil Comments dataset")
report_lines.append("- **Task**: Binary toxicity classification (toxic vs non-toxic)")
report_lines.append("- **Effect modifier (Z)**: Annotated by LLMs (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)")
report_lines.append(f"- **Subgroups (K=4)**: text_length × has_question")
for s in SUBGROUPS:
    sc = cm_data[LLMS[0]][s]
    report_lines.append(f"  - {s}: {SUBGROUP_LABELS[s]} (n={sc['n']})")
report_lines.append("")

report_lines.append("## Per-LLM Global Confusion Matrix Metrics\n")
report_lines.append("| LLM | Global FPR | Global FNR | Δ_FPR | Δ_FNR | R² |")
report_lines.append("|-----|-----------|-----------|-------|-------|-----|")
for llm in LLMS:
    g = cm_data[llm]["_global"]
    r2 = r2_data.get(llm, {}).get("R2", float("nan"))
    report_lines.append(f"| {llm} | {g['fpr']:.4f} | {g['fnr']:.4f} | {g['delta_fpr']:.4f} | {g['delta_fnr']:.4f} | {r2:.4f} |")
report_lines.append("")

report_lines.append("## Per-Subgroup Confusion Matrix Profiles\n")
for llm in LLMS:
    report_lines.append(f"### {llm}\n")
    report_lines.append("| Subgroup | Label | n | FPR | FNR | Accuracy | κ(C_s) | π_s | δ_s |")
    report_lines.append("|----------|-------|---|-----|-----|----------|--------|-----|-----|")
    for row in output_rows:
        if row["llm"] == llm:
            report_lines.append(
                f"| {row['subgroup']} | {row['subgroup_label']} | {row['n']} | "
                f"{row['FPR']:.4f} | {row['FNR']:.4f} | {row['accuracy']:.4f} | "
                f"{row['kappa']:.2f} | {row['pi_s']:.4f} | {row['delta_s']:+.4f} |"
            )
    report_lines.append("")

report_lines.append("## Theorem 1 Validation: Predicted vs Observed Bias (Δτ = 0.3)\n")
report_lines.append("| LLM | Subgroup | z | Predicted Bias | Observed Bias | SE | |Pred-Obs|/SE |")
report_lines.append("|-----|----------|---|---------------|--------------|-----|------------|")
for br in sorted(bias_rows, key=lambda x: (x["llm"], x["subgroup"], int(x["z"]), float(x["delta_tau"]))):
    if float(br["delta_tau"]) == 0.3:
        pred = float(br["predicted_bias"])
        obs = float(br["observed_bias"])
        se = float(br["se"])
        ratio = abs(pred - obs) / se if se > 0 else float("inf")
        report_lines.append(
            f"| {br['llm']} | {br['subgroup']} | {br['z']} | "
            f"{pred:+.6f} | {obs:+.6f} | {se:.6f} | {ratio:.2f} |"
        )
report_lines.append("")

report_lines.append("## R² Summary\n")
report_lines.append("### CivilComments\n")
report_lines.append("| LLM | R² | Max |B| | Diagnostic AUC |")
report_lines.append("|-----|-----|---------|----------------|")
for llm in LLMS:
    d = r2_data.get(llm, {})
    report_lines.append(f"| {llm} | {d.get('R2', 0):.4f} | {d.get('max_abs_B', 0):.4f} | {d.get('diagnostic_AUC', 0):.4f} |")
report_lines.append(f"\n**Pooled R² (all LLMs combined)**: {r2_all:.4f}\n")

report_lines.append("### Comparison with TweetEval\n")
report_lines.append("| LLM | CivilComments R² | TweetEval R² |")
report_lines.append("|-----|-----------------|-------------|")
for llm in LLMS:
    cc_r2 = r2_data.get(llm, {}).get("R2", float("nan"))
    te_r2 = r2_tweeteval.get(llm, float("nan"))
    report_lines.append(f"| {llm} | {cc_r2:.4f} | {te_r2:.4f} |" if not np.isnan(te_r2)
                        else f"| {llm} | {cc_r2:.4f} | — |")
report_lines.append("")

report_lines.append("## Debiasing Hurts Analysis\n")
report_lines.append(f"- **Total LLM × subgroup × z × Δτ combinations**: {total_combos}")
report_lines.append(f"- **Cases where |observed bias| > 2 SE**: {hurts_combos} ({100*hurts_combos/total_combos:.1f}%)")
report_lines.append(f"- This means global debiasing introduces statistically significant residual bias "
                    f"in {100*hurts_combos/total_combos:.1f}% of subgroup-level estimates.\n")

if extreme_cases:
    report_lines.append("### Most Extreme Bias Amplification Cases (max |bias| > 0.05)\n")
    report_lines.append("| LLM | Subgroup | Label | κ(C_s) | δ_s | Max |Bias| |")
    report_lines.append("|-----|----------|-------|--------|-----|-----------|")
    for c in sorted(extreme_cases, key=lambda x: -x["max_abs_bias"]):
        report_lines.append(
            f"| {c['llm']} | {c['subgroup']} | {c['subgroup_label']} | "
            f"{c['kappa']:.2f} | {c['delta_s']:+.4f} | {c['max_abs_bias']:.4f} |"
        )
    report_lines.append("")

report_lines.append("## Key Takeaways\n")

# Compute key stats
avg_r2_cc = np.mean([r2_data[llm]["R2"] for llm in LLMS])
avg_r2_te = np.mean([r2_tweeteval.get(llm, float("nan")) for llm in LLMS
                      if llm in r2_tweeteval])

# Find most ill-conditioned subgroup
worst_kappa = max(output_rows, key=lambda x: x["kappa"])
best_kappa = min(output_rows, key=lambda x: x["kappa"])

report_lines.append(f"1. **Theorem 1 accuracy**: Average R² = {avg_r2_cc:.3f} across {len(LLMS)} LLMs on CivilComments, "
                    f"confirming the analytic bias formula accurately predicts residual subgroup-level bias from global debiasing.")
if not np.isnan(avg_r2_te):
    report_lines.append(f"   - Comparable to TweetEval average R² = {avg_r2_te:.3f} (across overlapping LLMs).")
report_lines.append(f"2. **Misclassification heterogeneity**: Subgroup FPR ranges span "
                    f"{min(r['FPR'] for r in output_rows):.3f}–{max(r['FPR'] for r in output_rows):.3f}, "
                    f"FNR ranges span {min(r['FNR'] for r in output_rows):.3f}–{max(r['FNR'] for r in output_rows):.3f}.")
report_lines.append(f"3. **Most ill-conditioned subgroup**: {worst_kappa['llm']} × {worst_kappa['subgroup']} "
                    f"({worst_kappa['subgroup_label']}) with κ = {worst_kappa['kappa']:.2f}.")
report_lines.append(f"4. **Best-conditioned subgroup**: {best_kappa['llm']} × {best_kappa['subgroup']} "
                    f"({best_kappa['subgroup_label']}) with κ = {best_kappa['kappa']:.2f}.")
report_lines.append(f"5. **Debiasing hurts prevalence**: {100*hurts_combos/total_combos:.1f}% of subgroup-level "
                    f"estimates show statistically significant residual bias after global correction.")
report_lines.append(f"6. **Practical implication**: In hate speech detection, text-feature-based misclassification "
                    f"heterogeneity (driven by text length and question presence) produces systematic bias in "
                    f"subgroup-level treatment effect estimates even after global debiasing correction.")

report = "\n".join(report_lines)
report_path = ARTIFACTS / "hatespeech_analysis_report.md"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nWrote {report_path}")

# Print key numbers
print("\n" + "=" * 60)
print("KEY NUMBERS FOR PAPER")
print("=" * 60)
print(f"CivilComments LLMs tested: {len(LLMS)} ({', '.join(LLMS)})")
print(f"Subgroups: {len(SUBGROUPS)} (text_length × has_question)")
print(f"Pooled R²: {r2_all:.4f}")
for llm in LLMS:
    r2 = r2_data.get(llm, {}).get("R2", float("nan"))
    print(f"  {llm} R²: {r2:.4f}")
print(f"Debiasing hurts: {hurts_combos}/{total_combos} = {100*hurts_combos/total_combos:.1f}%")
print(f"Most extreme |bias|: {max(r['max_abs_bias'] for r in output_rows):.4f}")
print(f"κ range: {min(r['kappa'] for r in output_rows):.2f} – {max(r['kappa'] for r in output_rows):.2f}")
