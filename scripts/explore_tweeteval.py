#!/usr/bin/env python3
"""Explore all TweetEval subtasks for EC-HTE feasibility analysis."""

import json
import re
import sys
from collections import Counter
from datasets import load_dataset

SUBTASKS = [
    "stance_feminist", "stance_abortion", "stance_atheism",
    "stance_climate", "stance_hillary",
    "sentiment", "hate", "irony", "offensive", "emoji",
]

def text_length_stats(texts):
    lengths = [len(t) for t in texts]
    lengths.sort()
    n = len(lengths)
    return {
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
        "min": min(lengths),
        "max": max(lengths),
    }

def subgroup_features(texts):
    has_hashtag = sum(1 for t in texts if "#" in t)
    has_mention = sum(1 for t in texts if "@" in t)
    has_url = sum(1 for t in texts if "http" in t.lower())
    all_caps = sum(1 for t in texts if t == t.upper() and len(t) > 10)
    has_emoji_pat = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U0001f900-\U0001f9FF"
        "]+", flags=re.UNICODE
    )
    has_emoji = sum(1 for t in texts if has_emoji_pat.search(t))
    n = len(texts)
    lengths = [len(t) for t in texts]
    lengths.sort()
    p33 = lengths[n // 3]
    p66 = lengths[2 * n // 3]
    short = sum(1 for t in texts if len(t) <= p33)
    medium = sum(1 for t in texts if p33 < len(t) <= p66)
    long_ = sum(1 for t in texts if len(t) > p66)
    return {
        "n": n,
        "has_hashtag": has_hashtag,
        "has_mention": has_mention,
        "has_url": has_url,
        "all_caps": all_caps,
        "has_emoji": has_emoji,
        "length_short": short,
        "length_medium": medium,
        "length_long": long_,
        "length_thresholds": (p33, p66),
    }

results = {}

for task in SUBTASKS:
    print(f"\n{'='*60}")
    print(f"Loading: {task}")
    print(f"{'='*60}")
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", task, trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading {task}: {e}")
        results[task] = {"error": str(e)}
        continue

    info = {}
    # Splits and sizes
    splits = {}
    for split_name in ds:
        splits[split_name] = len(ds[split_name])
    info["splits"] = splits
    print(f"  Splits: {splits}")

    # Determine text and label column names
    cols = ds[list(ds.keys())[0]].column_names
    text_col = "text" if "text" in cols else cols[0]
    label_col = "label" if "label" in cols else cols[1]
    info["columns"] = cols

    # Label distribution (train)
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    labels = ds[train_split][label_col]
    label_counts = Counter(labels)
    info["label_distribution_train"] = dict(label_counts)

    # Try to get label names
    try:
        label_names = ds[train_split].features[label_col].names
        info["label_names"] = label_names
    except:
        info["label_names"] = None

    print(f"  Labels: {info.get('label_names', label_counts)}")
    for lbl, cnt in sorted(label_counts.items()):
        name = info["label_names"][lbl] if info["label_names"] else lbl
        pct = cnt / len(labels) * 100
        print(f"    {name}: {cnt} ({pct:.1f}%)")

    # Text length stats
    texts = ds[train_split][text_col]
    tl_stats = text_length_stats(texts)
    info["text_length_stats"] = tl_stats
    print(f"  Text length: mean={tl_stats['mean']:.1f}, median={tl_stats['median']}, "
          f"min={tl_stats['min']}, max={tl_stats['max']}")

    # First 5 examples
    info["examples"] = []
    print(f"  First 5 examples:")
    for i in range(min(5, len(ds[train_split]))):
        ex = ds[train_split][i]
        t = ex[text_col]
        l = ex[label_col]
        lname = info["label_names"][l] if info["label_names"] else l
        info["examples"].append({"text": t, "label": lname})
        print(f"    [{lname}] {t[:120]}")

    # Subgroup features (all data combined)
    all_texts = []
    for split_name in ds:
        all_texts.extend(ds[split_name][text_col])
    sg = subgroup_features(all_texts)
    info["subgroup_features"] = sg
    print(f"  Subgroup features (all splits, n={sg['n']}):")
    print(f"    Hashtag: {sg['has_hashtag']} ({sg['has_hashtag']/sg['n']*100:.1f}%)")
    print(f"    @mention: {sg['has_mention']} ({sg['has_mention']/sg['n']*100:.1f}%)")
    print(f"    URL: {sg['has_url']} ({sg['has_url']/sg['n']*100:.1f}%)")
    print(f"    ALL CAPS: {sg['all_caps']} ({sg['all_caps']/sg['n']*100:.1f}%)")
    print(f"    Emoji: {sg['has_emoji']} ({sg['has_emoji']/sg['n']*100:.1f}%)")
    print(f"    Length groups (thresholds {sg['length_thresholds']}): "
          f"short={sg['length_short']}, medium={sg['length_medium']}, long={sg['length_long']}")

    # Test split label distribution
    test_split = "test" if "test" in ds else None
    if test_split:
        test_labels = ds[test_split][label_col]
        test_label_counts = Counter(test_labels)
        info["label_distribution_test"] = dict(test_label_counts)

    results[task] = info

# Save raw results
with open("results/tweeteval_raw.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nDone. Raw results saved.")
