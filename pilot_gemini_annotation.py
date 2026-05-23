# GPT-4.1 sentiment annotation pilot (OpenAI-compatible SDK via proxy)
# Classifies 500 TweetEval sentiment tweets (3-class) and analyzes per-subgroup misclassification rates.

import argparse
import os
import random
import re
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import confusion_matrix


def compute_subgroup_features(text):
    has_hashtag = bool(re.search(r'#\w+', text))
    has_mention = bool(re.search(r'@\w+', text))
    text_length = len(text)
    if text_length < 80:
        text_length_bin = 'short'
    elif text_length < 140:
        text_length_bin = 'medium'
    else:
        text_length_bin = 'long'
    return has_hashtag, has_mention, text_length_bin


def make_classifier(dry_run):
    if dry_run:
        def classify_tweet(text, max_retries=3):
            return random.randint(0, 2)
        return classify_tweet

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
    )
    model_name = 'gpt-4.1'

    PROMPT_TEMPLATE = (
        'Classify the sentiment of the following tweet as exactly one of: '
        'positive, neutral, negative.\n\n'
        'Tweet: "{text}"\n\n'
        'Reply with ONLY one word: positive, neutral, or negative.'
    )

    def classify_tweet(text, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
                    max_tokens=10,
                    temperature=0
                )
                label_str = response.choices[0].message.content.strip().lower()
                label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                if label_str in label_map:
                    return label_map[label_str]
                for key in label_map:
                    if key in label_str:
                        return label_map[key]
                print(f"  Warning: unparseable response '{label_str}', retry {attempt+1}")
            except Exception as e:
                print(f"  Error: {e}, retry {attempt+1}")
                time.sleep(10)
        return -1

    return classify_tweet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--dry-run', action='store_true', help='Skip API calls, use random labels')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading TweetEval sentiment test split...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")

    n = min(args.n_samples, len(ds))
    indices = random.sample(range(len(ds)), n)
    print(f"Sampled {n} tweets (seed={args.seed}), dry_run={args.dry_run}")

    classify_tweet = make_classifier(args.dry_run)

    results = []
    for i, idx in enumerate(indices):
        text = ds[idx]['text']
        true_label = ds[idx]['label']

        llm_label = classify_tweet(text)
        has_hashtag, has_mention, text_length_bin = compute_subgroup_features(text)

        results.append({
            'idx': idx,
            'text': text,
            'true_label': true_label,
            'llm_label': llm_label,
            'has_hashtag': has_hashtag,
            'has_mention': has_mention,
            'text_length_bin': text_length_bin,
        })

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{n}] done, {sum(1 for r in results if r['llm_label'] >= 0)} successful")

        if not args.dry_run:
            time.sleep(1.0)

    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/pilot_gemini_annotations.csv', index=False)
    print(f"\nSaved to results/pilot_gemini_annotations.csv")

    df_valid = df[df['llm_label'] >= 0].copy()
    print(f"Total: {len(df)}, Valid: {len(df_valid)}, Failed: {len(df) - len(df_valid)}")

    acc = (df_valid['true_label'] == df_valid['llm_label']).mean()
    print(f"Overall accuracy: {acc:.3f}")

    df_valid['true_z'] = (df_valid['true_label'] == 2).astype(int)
    df_valid['llm_z'] = (df_valid['llm_label'] == 2).astype(int)

    df_valid['subgroup'] = (
        df_valid['has_hashtag'].map({True: 'hash', False: 'nohash'})
        + '_'
        + df_valid['text_length_bin']
    )

    print("\n=== Per-subgroup misclassification rates (binary Z: positive vs non-positive) ===")
    for sg in sorted(df_valid['subgroup'].unique()):
        sg_data = df_valid[df_valid['subgroup'] == sg]
        misclass_rate = (sg_data['true_z'] != sg_data['llm_z']).mean()
        n_sg = len(sg_data)
        print(f"  {sg:20s}  n={n_sg:4d}  misclass_rate={misclass_rate:.3f}")

    rates = df_valid.groupby('subgroup').apply(lambda x: (x['true_z'] != x['llm_z']).mean())
    delta_pi = rates.max() - rates.min()
    print(f"\nΔπ (max - min misclass rate) = {delta_pi:.3f}")
    if delta_pi > 0.05:
        print("✓ Meaningful heterogeneity detected (Δπ > 0.05)")
    else:
        print("⚠ Low heterogeneity (Δπ ≤ 0.05)")

    print("\n=== Per-subgroup confusion matrices (3-class) ===")
    for sg in sorted(df_valid['subgroup'].unique()):
        sg_data = df_valid[df_valid['subgroup'] == sg]
        print(f"\n--- {sg} (n={len(sg_data)}) ---")
        cm = confusion_matrix(sg_data['true_label'], sg_data['llm_label'], labels=[0, 1, 2])
        print(f"  Confusion matrix (rows=true, cols=pred):")
        print(f"         neg  neu  pos")
        for j, label in enumerate(['neg', 'neu', 'pos']):
            print(f"  {label}  {cm[j]}")
        acc_sg = (sg_data['true_label'] == sg_data['llm_label']).mean()
        print(f"  Accuracy: {acc_sg:.3f}")


if __name__ == '__main__':
    main()
