"""
exp-023: Parallel LLM annotation for Civil Comments.
Runs 3 models concurrently with 4 workers each.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJ = Path(__file__).parent
RESULTS = PROJ / "results"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

PROMPT_TEMPLATE = (
    'Is the following comment toxic? A comment is toxic if it is rude, '
    'disrespectful, or likely to make someone leave a discussion.\n\n'
    'Comment: "{text}"\n\n'
    'Reply with ONLY one word: toxic or non-toxic.'
)


def get_client():
    return OpenAI(api_key=API_KEY, base_url=API_BASE)


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
                time.sleep(5 * (attempt + 1))
            else:
                return -1, str(e)
    return -1, "max_retries"


def annotate_model(model, df):
    ckpt_path = RESULTS / f"exp023_checkpoint_{model}.json"
    existing = {}
    if ckpt_path.exists():
        existing = json.loads(ckpt_path.read_text())
        print(f"[{model}] Resuming: {len(existing)}/3000 done")

    client = get_client()
    todo = [(i, row) for i, row in df.iterrows() if str(row["idx"]) not in existing]
    print(f"[{model}] {len(todo)} remaining")

    done_count = len(existing)
    for i, row in todo:
        key = str(row["idx"])
        pred, raw = annotate_one(client, model, row["text"])
        existing[key] = {"pred": pred, "raw": raw}
        done_count += 1
        if done_count % 100 == 0:
            ckpt_path.write_text(json.dumps(existing))
            print(f"[{model}] {done_count}/3000", flush=True)

    ckpt_path.write_text(json.dumps(existing))
    print(f"[{model}] DONE: {len(existing)}")
    return existing


def compute_cms(df, model_preds, models):
    subgroups = ["S0", "S1", "S2", "S3"]
    results = {}

    for model in models:
        preds = model_preds[model]
        pred_col = []
        for _, row in df.iterrows():
            key = str(row["idx"])
            if key in preds:
                pred_col.append(preds[key]["pred"])
            else:
                pred_col.append(-1)
        df[f"pred_{model}"] = pred_col

        valid = df[df[f"pred_{model}"] >= 0]
        print(f"\n{model}: {len(valid)} valid / {len(df)} total")

        model_cms = {}
        for sg in subgroups:
            sub = valid[valid["subgroup"] == sg]
            z_true = sub["y_true"].values
            z_hat = sub[f"pred_{model}"].values

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

        z_true = valid["y_true"].values
        z_hat = valid[f"pred_{model}"].values
        fp = ((z_true == 0) & (z_hat == 1)).sum()
        fn = ((z_true == 1) & (z_hat == 0)).sum()
        fpr = fp / max((z_true == 0).sum(), 1)
        fnr = fn / max((z_true == 1).sum(), 1)

        delta_fpr = max(m["fpr"] for m in model_cms.values()) - min(m["fpr"] for m in model_cms.values())
        delta_fnr = max(m["fnr"] for m in model_cms.values()) - min(m["fnr"] for m in model_cms.values())
        print(f"  Global: FPR={fpr:.3f}, FNR={fnr:.3f}, Δ_FPR={delta_fpr:.3f}, Δ_FNR={delta_fnr:.3f}")

        model_cms["_global"] = {
            "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "delta_fpr": round(delta_fpr, 4), "delta_fnr": round(delta_fnr, 4),
        }
        results[model] = model_cms

    return results, df


def main():
    df = pd.read_parquet(RESULTS / "exp023_civil_comments_pool.parquet")
    print(f"Pool: {len(df)} comments, {df['y_true'].sum()} toxic")

    model_preds = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(annotate_model, m, df): m for m in MODELS}
        for future in as_completed(futures):
            model = futures[future]
            model_preds[model] = future.result()

    cms, df = compute_cms(df, model_preds, MODELS)

    df.to_csv(RESULTS / "exp023_llm_annotations.csv", index=False)
    (RESULTS / "exp023_confusion_matrices.json").write_text(json.dumps(cms, indent=2))
    print("\nDone. Saved annotations and confusion matrices.")


if __name__ == "__main__":
    main()
