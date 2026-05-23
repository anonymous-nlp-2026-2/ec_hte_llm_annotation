"""
exp-023 Phase 1: Civil Comments dataset preparation and LLM annotation.

Downloads Civil Comments, defines subgroups (text_length × insult_signal, K=4),
stratified-samples ~3000 comments with ~30% toxic, then annotates with 3 API models.

Output:
  results/exp023_civil_comments_pool.parquet   – full annotated pool
  results/exp023_llm_annotations.csv           – LLM predictions per model
  results/exp023_confusion_matrices.json       – per-subgroup CMs per model
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

PROJ = Path(__file__).parent
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)

API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

PROMPT_TEMPLATE = (
    'Is the following comment toxic? A comment is toxic if it is rude, '
    'disrespectful, or likely to make someone leave a discussion.\n\n'
    'Comment: "{text}"\n\n'
    'Reply with ONLY one word: toxic or non-toxic.'
)

SAMPLE_N = 3000
TOXIC_FRAC = 0.30
SEED = 42


def load_and_prepare():
    """Load Civil Comments, compute features, stratified sample."""
    print("Loading Civil Comments from HuggingFace...")
    ds = load_dataset("google/civil_comments", split="train")

    texts = ds["text"]
    toxicity = np.array(ds["toxicity"])
    insult = np.array(ds["insult"])
    identity_attack = np.array(ds["identity_attack"])
    obscene = np.array(ds["obscene"])

    y_true = (toxicity >= 0.5).astype(int)
    lengths = np.array([len(t) for t in texts])
    has_insult = (insult > 0).astype(int)

    print(f"Total: {len(texts)}, Toxic: {y_true.sum()} ({y_true.mean():.3f})")
    print(f"Insult>0: {has_insult.sum()} ({has_insult.mean():.3f})")

    median_len = np.median(lengths)
    len_bin = (lengths > median_len).astype(int)
    print(f"Median text length: {median_len:.0f}")

    rng = np.random.RandomState(SEED)

    n_toxic = int(SAMPLE_N * TOXIC_FRAC)
    n_clean = SAMPLE_N - n_toxic

    idx_toxic = np.where(y_true == 1)[0]
    idx_clean = np.where(y_true == 0)[0]

    chosen_toxic = rng.choice(idx_toxic, n_toxic, replace=False)
    chosen_clean = rng.choice(idx_clean, n_clean, replace=False)
    idx = np.concatenate([chosen_toxic, chosen_clean])
    rng.shuffle(idx)

    df = pd.DataFrame({
        "idx": idx,
        "text": [texts[i] for i in idx],
        "y_true": y_true[idx],
        "toxicity": toxicity[idx],
        "insult": insult[idx],
        "identity_attack": identity_attack[idx],
        "obscene": obscene[idx],
        "text_length": lengths[idx],
        "has_insult": has_insult[idx],
        "len_bin": len_bin[idx],
    })

    median_sample = np.median(df["text_length"].values)
    df["len_bin"] = (df["text_length"] > median_sample).astype(int)
    df["has_question"] = np.array(["?" in t for t in df["text"].values], dtype=int)

    df["subgroup"] = "S0"
    df.loc[(df["len_bin"] == 0) & (df["has_question"] == 0), "subgroup"] = "S0"
    df.loc[(df["len_bin"] == 0) & (df["has_question"] == 1), "subgroup"] = "S1"
    df.loc[(df["len_bin"] == 1) & (df["has_question"] == 0), "subgroup"] = "S2"
    df.loc[(df["len_bin"] == 1) & (df["has_question"] == 1), "subgroup"] = "S3"

    print(f"\nSample: {len(df)}, Toxic: {df['y_true'].sum()} ({df['y_true'].mean():.3f})")
    print("\nSubgroup distribution:")
    for s in ["S0", "S1", "S2", "S3"]:
        sub = df[df["subgroup"] == s]
        print(f"  {s}: n={len(sub)}, toxic={sub['y_true'].sum()} ({sub['y_true'].mean():.3f})")

    out = RESULTS / "exp023_civil_comments_pool.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved pool to {out}")
    return df


def get_client():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=API_BASE)


def annotate_one(client, model, text, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(text=text[:1500])
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().lower()
            if "non-toxic" in raw or "non_toxic" in raw or "nontoxic" in raw:
                return 0, raw
            elif "toxic" in raw:
                return 1, raw
            else:
                return -1, raw
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return -1, str(e)
    return -1, "max_retries"


def annotate_model(df, model):
    """Annotate all comments with one model, with checkpointing."""
    ckpt_path = RESULTS / f"exp023_checkpoint_{model}.json"

    existing = {}
    if ckpt_path.exists():
        existing = json.loads(ckpt_path.read_text())
        print(f"  Resuming {model}: {len(existing)} already done")

    client = get_client()
    preds = []
    raws = []

    for i, row in df.iterrows():
        key = str(row["idx"])
        if key in existing:
            preds.append(existing[key]["pred"])
            raws.append(existing[key]["raw"])
            continue

        pred, raw = annotate_one(client, model, row["text"])
        preds.append(pred)
        raws.append(raw)
        existing[key] = {"pred": pred, "raw": raw}

        if len(existing) % 50 == 0:
            ckpt_path.write_text(json.dumps(existing))
            done = len(existing)
            total = len(df)
            print(f"  {model}: {done}/{total} ({done/total*100:.1f}%)")

        time.sleep(0.1)

    ckpt_path.write_text(json.dumps(existing))
    print(f"  {model}: done, {len(existing)} annotations")
    return preds, raws


def compute_confusion_matrices(df, models):
    """Compute per-subgroup confusion matrices for each model."""
    subgroups = ["S0", "S1", "S2", "S3"]
    results = {}

    for model in models:
        col = f"pred_{model}"
        valid = df[df[col] >= 0]
        print(f"\n{model}: {len(valid)} valid / {len(df)} total")

        model_cms = {}
        for sg in subgroups:
            sub = valid[valid["subgroup"] == sg]
            z_true = sub["y_true"].values
            z_hat = sub[col].values

            tp = ((z_true == 1) & (z_hat == 1)).sum()
            tn = ((z_true == 0) & (z_hat == 0)).sum()
            fp = ((z_true == 0) & (z_hat == 1)).sum()
            fn = ((z_true == 1) & (z_hat == 0)).sum()

            n_pos = (z_true == 1).sum()
            n_neg = (z_true == 0).sum()
            fpr = fp / max(n_neg, 1)
            fnr = fn / max(n_pos, 1)
            acc = (tp + tn) / max(len(sub), 1)

            model_cms[sg] = {
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                "fpr": round(fpr, 4), "fnr": round(fnr, 4),
                "accuracy": round(acc, 4), "n": len(sub),
                "n_pos": int(n_pos), "n_neg": int(n_neg),
            }
            print(f"  {sg}: n={len(sub)}, FPR={fpr:.3f}, FNR={fnr:.3f}, Acc={acc:.3f}")

        # Global
        z_true = valid["y_true"].values
        z_hat = valid[col].values
        tp = ((z_true == 1) & (z_hat == 1)).sum()
        tn = ((z_true == 0) & (z_hat == 0)).sum()
        fp = ((z_true == 0) & (z_hat == 1)).sum()
        fn = ((z_true == 1) & (z_hat == 0)).sum()
        fpr = fp / max((z_true == 0).sum(), 1)
        fnr = fn / max((z_true == 1).sum(), 1)
        print(f"  Global: FPR={fpr:.3f}, FNR={fnr:.3f}")

        delta_fpr = max(m["fpr"] for m in model_cms.values()) - min(m["fpr"] for m in model_cms.values())
        delta_fnr = max(m["fnr"] for m in model_cms.values()) - min(m["fnr"] for m in model_cms.values())
        print(f"  Δ_FPR={delta_fpr:.3f}, Δ_FNR={delta_fnr:.3f}")

        model_cms["_global"] = {
            "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "delta_fpr": round(delta_fpr, 4), "delta_fnr": round(delta_fnr, 4),
        }
        results[model] = model_cms

    return results


def main():
    pool_path = RESULTS / "exp023_civil_comments_pool.parquet"
    if pool_path.exists():
        print(f"Loading existing pool from {pool_path}")
        df = pd.read_parquet(pool_path)
    else:
        df = load_and_prepare()

    print(f"\n=== LLM Annotation ({len(df)} comments × {len(MODELS)} models) ===")
    for model in MODELS:
        print(f"\nAnnotating with {model}...")
        preds, raws = annotate_model(df, model)
        df[f"pred_{model}"] = preds
        df[f"raw_{model}"] = raws

    ann_path = RESULTS / "exp023_llm_annotations.csv"
    df.to_csv(ann_path, index=False)
    print(f"\nSaved annotations to {ann_path}")

    print("\n=== Confusion Matrices ===")
    cms = compute_confusion_matrices(df, MODELS)

    cm_path = RESULTS / "exp023_confusion_matrices.json"
    cm_path.write_text(json.dumps(cms, indent=2))
    print(f"\nSaved CMs to {cm_path}")


if __name__ == "__main__":
    main()
