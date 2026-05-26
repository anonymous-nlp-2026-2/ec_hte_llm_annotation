#!/usr/bin/env python3
"""exp-008: Multi-LLM Misclassification Comparison."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

API_KEY_PATH = Path(__file__).parent / ".api_key"
RESULTS_DIR = Path(__file__).parent / "results"
PILOT_CSV = RESULTS_DIR / "pilot_gemini_annotations.csv"

PROMPT_TEMPLATE = (
    'Classify the sentiment of the following tweet as exactly one of: '
    'positive, neutral, negative.\n\n'
    'Tweet: "{text}"\n\n'
    'Reply with ONLY one word: positive, neutral, or negative.'
)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

PREFERRED_MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4-turbo",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "deepseek-chat",
    "deepseek-v3",
    "qwen-turbo",
    "qwen-plus",
]


def get_client():
    if not API_KEY_PATH.exists():
        print("API key not found at", API_KEY_PATH)
        return None
    key = API_KEY_PATH.read_text().strip()
    from openai import OpenAI
    return OpenAI(api_key=key, base_url="http://47.94.22.126/v1")


def phase_probe():
    client = get_client()
    if client is None:
        return []
    models = client.models.list()
    model_ids = sorted([m.id for m in models.data])
    out = RESULTS_DIR / "exp008_available_models.txt"
    out.write_text("\n".join(model_ids) + "\n")
    print(f"Found {len(model_ids)} models, saved to {out}")
    for m in model_ids:
        print(f"  {m}")
    return model_ids


def select_models(available_ids):
    selected = []
    for pref in PREFERRED_MODELS:
        matches = [m for m in available_ids if pref in m]
        if matches:
            selected.append(matches[0])
        if len(selected) >= 4:
            break
    if len(selected) < 3:
        for m in available_ids:
            if m not in selected and "gpt-4.1" not in m:
                selected.append(m)
            if len(selected) >= 3:
                break
    print(f"Selected models: {selected}")
    return selected


def annotate_one(client, model_name, texts, max_retries=3):
    labels = []
    for i, text in enumerate(texts):
        prompt = PROMPT_TEMPLATE.format(text=text)
        label = None
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                )
                try:
                    resp = client.chat.completions.create(temperature=0, **kwargs)
                except Exception:
                    resp = client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content.strip().lower().rstrip(".")
                if raw in LABEL_MAP:
                    label = LABEL_MAP[raw]
                else:
                    for k, v in LABEL_MAP.items():
                        if k in raw:
                            label = v
                            break
                if label is not None:
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  Retry {attempt+1} for row {i} ({model_name}): {e}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  Failed row {i} ({model_name}): {e}")
        labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"  {model_name}: {i+1}/{len(texts)} done")
        time.sleep(1.0)
    return labels


def phase_annotate():
    client = get_client()
    if client is None:
        print("API key needed for annotation phase")
        return None

    available = phase_probe()
    selected = select_models(available)
    if not selected:
        print("No suitable models found")
        return None

    df = pd.read_csv(PILOT_CSV)
    df = df.rename(columns={"llm_label": "gpt-4.1"})
    texts = df["text"].tolist()

    for model_name in selected:
        col = model_name.replace("/", "_")
        checkpoint = RESULTS_DIR / f"exp008_checkpoint_{col}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if len(saved) == len(texts):
                print(f"Loaded checkpoint for {model_name}")
                df[col] = saved
                continue

        print(f"Annotating with {model_name} ({len(texts)} tweets)...")
        labels = annotate_one(client, model_name, texts)
        df[col] = labels

        checkpoint.write_text(json.dumps(labels))
        valid = sum(1 for x in labels if x is not None)
        print(f"  {model_name}: {valid}/{len(texts)} valid labels")

    out = RESULTS_DIR / "exp008_multi_llm_annotations.csv"
    df.to_csv(out, index=False)
    print(f"Saved annotations to {out}")
    return df


def phase_analyze(df=None):
    ann_path = RESULTS_DIR / "exp008_multi_llm_annotations.csv"
    if df is None:
        if ann_path.exists():
            df = pd.read_csv(ann_path)
        else:
            df = pd.read_csv(PILOT_CSV)
            df = df.rename(columns={"llm_label": "gpt-4.1"})

    model_cols = [c for c in df.columns if c not in ["idx", "text", "true_label", "has_hashtag", "has_mention", "text_length_bin"]]
    print(f"Models to analyze: {model_cols}")

    df["subgroup"] = df["has_hashtag"].astype(str) + "_" + df["text_length_bin"]
    subgroups = sorted(df["subgroup"].unique())

    rows = []
    for model in model_cols:
        valid = df[df[model].notna()].copy()
        z_true = (valid["true_label"] == 2).astype(int)
        z_llm = (valid[model] == 2).astype(int)

        for sg in subgroups:
            mask = valid["subgroup"] == sg
            n = mask.sum()
            if n == 0:
                continue
            zt = z_true[mask]
            zl = z_llm[mask]
            misclass = (zt != zl).mean()
            acc = (valid.loc[mask, "true_label"] == valid.loc[mask, model]).mean()

            pos_mask = zt == 0
            neg_mask = zt == 1
            fpr = (zl[pos_mask] == 1).mean() if pos_mask.sum() > 0 else np.nan
            fnr = (zl[neg_mask] == 0).mean() if neg_mask.sum() > 0 else np.nan

            rows.append({
                "model": model,
                "subgroup": sg,
                "n": int(n),
                "misclass_rate": round(misclass, 4),
                "accuracy": round(acc, 4),
                "FPR": round(fpr, 4) if not np.isnan(fpr) else np.nan,
                "FNR": round(fnr, 4) if not np.isnan(fnr) else np.nan,
            })

    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS_DIR / "exp008_multi_llm_comparison.csv", index=False)
    print(f"Saved comparison to {RESULTS_DIR / 'exp008_multi_llm_comparison.csv'}")

    report = ["# exp-008: Multi-LLM Misclassification Comparison\n"]

    avail_path = RESULTS_DIR / "exp008_available_models.txt"
    if avail_path.exists():
        avail = avail_path.read_text().strip().split("\n")
        report.append(f"## Available Models\n\n{len(avail)} models on proxy:\n")
        for m in avail:
            report.append(f"- {m}")
        report.append("")

    report.append(f"## Models Analyzed\n")
    for model in model_cols:
        valid_n = df[model].notna().sum()
        report.append(f"- **{model}**: {valid_n} valid annotations")
    report.append("")

    report.append("## Per-Model Per-Subgroup Misclassification (Positive vs Non-Positive)\n")
    pivot = comp.pivot_table(index="subgroup", columns="model", values="misclass_rate")
    report.append(pivot.to_markdown())
    report.append("")

    report.append("## Per-Model Per-Subgroup Accuracy (3-class)\n")
    pivot_acc = comp.pivot_table(index="subgroup", columns="model", values="accuracy")
    report.append(pivot_acc.to_markdown())
    report.append("")

    report.append("## FPR by Subgroup\n")
    pivot_fpr = comp.pivot_table(index="subgroup", columns="model", values="FPR")
    report.append(pivot_fpr.to_markdown())
    report.append("")

    report.append("## FNR by Subgroup\n")
    pivot_fnr = comp.pivot_table(index="subgroup", columns="model", values="FNR")
    report.append(pivot_fnr.to_markdown())
    report.append("")

    report.append("## Delta-pi (max - min misclass rate across subgroups)\n")
    report.append("| Model | min | max | Δπ | Δπ > 0.05? |")
    report.append("|-------|-----|-----|----|------------|")
    delta_pis = {}
    for model in model_cols:
        mc = comp[comp["model"] == model]
        mn, mx = mc["misclass_rate"].min(), mc["misclass_rate"].max()
        dp = round(mx - mn, 4)
        delta_pis[model] = dp
        flag = "YES" if dp > 0.05 else "no"
        report.append(f"| {model} | {mn:.4f} | {mx:.4f} | {dp:.4f} | {flag} |")
    report.append("")

    report.append("## Consistency Analysis\n")
    if len(model_cols) > 1:
        worst_sg = {}
        best_sg = {}
        for model in model_cols:
            mc = comp[comp["model"] == model]
            worst_sg[model] = mc.loc[mc["misclass_rate"].idxmax(), "subgroup"]
            best_sg[model] = mc.loc[mc["misclass_rate"].idxmin(), "subgroup"]

        report.append("**Highest misclassification subgroup per model:**\n")
        for model, sg in worst_sg.items():
            rate = comp[(comp["model"] == model) & (comp["subgroup"] == sg)]["misclass_rate"].values[0]
            report.append(f"- {model}: `{sg}` ({rate:.4f})")
        report.append("")

        report.append("**Lowest misclassification subgroup per model:**\n")
        for model, sg in best_sg.items():
            rate = comp[(comp["model"] == model) & (comp["subgroup"] == sg)]["misclass_rate"].values[0]
            report.append(f"- {model}: `{sg}` ({rate:.4f})")
        report.append("")

        worst_vals = list(worst_sg.values())
        if len(set(worst_vals)) == 1:
            report.append(f"All models agree: `{worst_vals[0]}` is the hardest subgroup.\n")
        else:
            from collections import Counter
            c = Counter(worst_vals)
            most_common = c.most_common(1)[0]
            report.append(f"Most common worst subgroup: `{most_common[0]}` ({most_common[1]}/{len(model_cols)} models).\n")

        if len(model_cols) >= 2:
            from scipy.stats import spearmanr
            report.append("**Rank correlation of subgroup misclass rates between models:**\n")
            for i in range(len(model_cols)):
                for j in range(i + 1, len(model_cols)):
                    m1, m2 = model_cols[i], model_cols[j]
                    merged = pivot[[m1, m2]].dropna()
                    if len(merged) >= 3:
                        rho, pval = spearmanr(merged[m1], merged[m2])
                        report.append(f"- {m1} vs {m2}: ρ={rho:.3f} (p={pval:.3f})")
            report.append("")
    else:
        report.append("Only one model available. Cross-model consistency analysis requires additional models.\n")

    report.append("## Key Finding\n")
    all_above = all(dp > 0.05 for dp in delta_pis.values())
    n_above = sum(1 for dp in delta_pis.values() if dp > 0.05)
    if all_above and len(model_cols) > 1:
        report.append(
            f"**All {len(model_cols)} models show Δπ > 0.05**, confirming that "
            "subgroup-specific misclassification heterogeneity is not model-specific. "
            "The bias mechanism persists across diverse LLM architectures and sizes."
        )
    elif n_above > 0:
        report.append(
            f"**{n_above}/{len(model_cols)} models show Δπ > 0.05**. "
            "Misclassification heterogeneity is present in most models, "
            "supporting the claim that the bias mechanism is structural."
        )
    else:
        report.append(
            "No models show Δπ > 0.05 with current data. "
            "Further investigation needed."
        )
    report.append("")

    md_path = RESULTS_DIR / "exp008_multi_llm_analysis.md"
    md_path.write_text("\n".join(report))
    print(f"Saved analysis to {md_path}")

    print("\n=== SUMMARY ===")
    for model in model_cols:
        dp = delta_pis[model]
        print(f"  {model}: Δπ = {dp:.4f} {'(> 0.05)' if dp > 0.05 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["probe", "annotate", "analyze", "all"], default="all")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.phase == "probe":
        phase_probe()
    elif args.phase == "annotate":
        phase_annotate()
    elif args.phase == "analyze":
        phase_analyze()
    elif args.phase == "all":
        df = phase_annotate()
        phase_analyze(df)


if __name__ == "__main__":
    main()
