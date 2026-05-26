#!/usr/bin/env python3
"""exp004_hate_speech.py — UC Berkeley Hate Speech EC-HTE Pipeline (REVISED)

Unit of analysis: COMMENT (not annotation).
Subgroups S: text features (primary_target_type × text_length_bin).

Usage:
  python3 exp004_hate_speech.py --phase prep
  python3 exp004_hate_speech.py --phase annotate [--resume]
  python3 exp004_hate_speech.py --phase analyze
  python3 exp004_hate_speech.py --phase all
  python3 exp004_hate_speech.py --n-comments 2000
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RESULTS_DIR = 'results'
DATA_DIR = 'data'
API_KEY_PATH = '/home/ubuntu/ec_hte_llm_annotation/.api_key'


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Download & Preprocessing (comment-level)
# ══════════════════════════════════════════════════════════════════════════════

def phase_prep(n_comments=2500, seed=42):
    from datasets import load_dataset
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    cache_path = os.path.join(DATA_DIR, 'exp004_raw.parquet')
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Downloading UC Berkeley Measuring Hate Speech dataset...")
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
        df = ds.to_pandas()
        df.to_parquet(cache_path)
        print(f"Cached to {cache_path}")

    n_annot = len(df)
    n_comments_total = df['comment_id'].nunique()
    print(f"Loaded {n_annot} annotations, {n_comments_total} unique comments")

    # ── Identify target columns ──
    target_cols = ['target_race', 'target_gender', 'target_religion',
                   'target_sexuality', 'target_origin']
    present_target_cols = [c for c in target_cols if c in df.columns]
    if not present_target_cols:
        race_cols = [c for c in df.columns if c.startswith('target_race')]
        print(f"No aggregate target_race flag. Subcategory columns: {race_cols}")
        if race_cols:
            df['target_race'] = df[race_cols].max(axis=1)
        gender_cols = [c for c in df.columns if c.startswith('target_gender')]
        if gender_cols:
            df['target_gender'] = df[gender_cols].max(axis=1)
        religion_cols = [c for c in df.columns if c.startswith('target_religion')]
        if religion_cols:
            df['target_religion'] = df[religion_cols].max(axis=1)
        present_target_cols = [c for c in target_cols if c in df.columns]

    print(f"Target columns available: {present_target_cols}")

    # ── Aggregate to comment level ──
    print("\nAggregating to comment level...")

    agg_dict = {
        'text': 'first',
        'hate_speech_score': 'first',
        'dehumanize': 'mean',
        'platform': 'first',
    }
    for col in present_target_cols:
        agg_dict[col] = 'mean'

    comment_df = df.groupby('comment_id').agg(agg_dict).reset_index()
    comment_df['n_annotators'] = df.groupby('comment_id').size().values

    # Y = hate_speech_score (continuous, already per-comment)
    comment_df['Y'] = comment_df['hate_speech_score']

    # T = majority vote of target_race
    if 'target_race' in comment_df.columns:
        comment_df['T'] = (comment_df['target_race'] >= 0.5).astype(int)
    else:
        print("ERROR: target_race not available")
        sys.exit(1)

    # Z_human = majority vote of dehumanize >= 2
    comment_df['Z_human'] = (comment_df['dehumanize'] >= 2.0).astype(int)

    # ── S: platform_group × text_length_bin ──
    # Drop platform 1 (only 70 comments). Group: platform 0 vs platforms 2+3.
    # This avoids confounding S with T (primary_target_type "race" ≡ T=1).
    comment_df = comment_df[comment_df['platform'] != 1].copy()

    comment_df['platform_group'] = np.where(
        comment_df['platform'] == 0, 'plat0', 'plat23')

    comment_df['text_length'] = comment_df['text'].str.len()
    comment_df['text_length_bin'] = np.where(
        comment_df['text_length'] < 100, 'short', 'long')

    comment_df['S'] = comment_df['platform_group'] + '_' + comment_df['text_length_bin']

    n_total = len(comment_df)
    print(f"\nComment-level dataset: {n_total} comments (dropped platform 1)")
    print(f"  Y (hate_speech_score): mean={comment_df['Y'].mean():.3f}, "
          f"std={comment_df['Y'].std():.3f}")
    print(f"  T (target_race majority): {comment_df['T'].mean():.3f}")
    print(f"  Z_human (dehumanize>=2 majority): {comment_df['Z_human'].mean():.3f}")

    print(f"\n  platform_group distribution:")
    for t, n in comment_df['platform_group'].value_counts().items():
        print(f"    {t}: {n} ({100*n/n_total:.1f}%)")

    print(f"\n  text_length_bin distribution:")
    for t, n in comment_df['text_length_bin'].value_counts().items():
        print(f"    {t}: {n} ({100*n/n_total:.1f}%)")

    print(f"\n  S (subgroup) distribution:")
    for s, n in comment_df['S'].value_counts().sort_index().items():
        print(f"    {s}: {n} ({100*n/n_total:.1f}%)")

    # ── Cross-tabulation T × Z × S ──
    print(f"\n  T × Z_human × S cell sizes:")
    for s in sorted(comment_df['S'].unique()):
        for t in [0, 1]:
            for z in [0, 1]:
                n = ((comment_df['T'] == t) & (comment_df['Z_human'] == z) &
                     (comment_df['S'] == s)).sum()
                print(f"    T={t} Z={z} S={s:20s}: {n}")

    # Check minimum cell sizes
    min_cell = float('inf')
    for s in comment_df['S'].unique():
        for t in [0, 1]:
            n = ((comment_df['T'] == t) & (comment_df['S'] == s)).sum()
            min_cell = min(min_cell, n)
    print(f"\n  Minimum (T, S) cell size: {min_cell}")
    if min_cell < 200:
        print("  WARNING: Some cells below 200. Consider merging subgroups.")

    # ── Save processed comment-level data ──
    keep_cols = ['comment_id', 'text', 'Y', 'T', 'Z_human', 'S',
                 'platform_group', 'text_length_bin',
                 'text_length', 'platform', 'n_annotators',
                 'hate_speech_score', 'dehumanize']
    keep_cols = [c for c in keep_cols if c in comment_df.columns]
    proc_path = os.path.join(DATA_DIR, 'exp004_comments_processed.parquet')
    comment_df[keep_cols].to_parquet(proc_path)
    print(f"\nSaved processed comments: {proc_path} ({len(comment_df)} rows)")

    # ── Sample comments for LLM annotation ──
    rng = np.random.RandomState(seed)
    subgroups = sorted(comment_df['S'].unique())

    # Stratified sampling: balance across T and S
    sampled_parts = []
    target_per_cell = n_comments // (2 * len(subgroups))
    target_per_cell = max(target_per_cell, 250)

    for s in subgroups:
        for t in [0, 1]:
            pool = comment_df[(comment_df['S'] == s) & (comment_df['T'] == t)]
            n_sample = min(target_per_cell, len(pool))
            sampled_parts.append(pool.sample(n=n_sample, random_state=rng))

    sampled = pd.concat(sampled_parts).drop_duplicates(subset='comment_id')
    print(f"\nSampled {len(sampled)} comments for LLM annotation:")
    for s in subgroups:
        for t in [0, 1]:
            n = ((sampled['S'] == s) & (sampled['T'] == t)).sum()
            print(f"  T={t} S={s:20s}: {n}")

    out_path = os.path.join(RESULTS_DIR, 'exp004_comments_to_annotate.csv')
    sampled[['comment_id', 'text', 'T', 'S', 'Z_human', 'Y']].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return comment_df


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: LLM Annotation
# ══════════════════════════════════════════════════════════════════════════════

def phase_annotate(resume=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(API_KEY_PATH):
        print(f"API key not found at {API_KEY_PATH}")
        return None

    api_key = open(API_KEY_PATH).read().strip()
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url='http://47.94.22.126/v1')

    comments_path = os.path.join(RESULTS_DIR, 'exp004_comments_to_annotate.csv')
    if not os.path.exists(comments_path):
        print("Comments file not found — run prep first")
        return None

    comments = pd.read_csv(comments_path)
    print(f"Loaded {len(comments)} comments to annotate")

    final_path = os.path.join(RESULTS_DIR, 'exp004_llm_annotations.csv')
    partial_path = os.path.join(RESULTS_DIR, 'exp004_llm_annotations_partial.csv')

    if not resume and os.path.exists(final_path):
        existing_df = pd.read_csv(final_path)
        if 'S' in existing_df.columns and 'Y' in existing_df.columns:
            print(f"Final annotations exist at {final_path} ({len(existing_df)} rows)")
            return existing_df

    existing = []
    done_ids = set()
    if resume:
        for path in [partial_path, final_path]:
            if os.path.exists(path):
                existing = pd.read_csv(path).to_dict('records')
                done_ids = {r['comment_id'] for r in existing}
                print(f"Resuming from {path}: {len(done_ids)} completed")
                break

    PROMPT = (
        'Read the following comment and classify whether it contains dehumanizing language '
        '(treating people as less than human, comparing them to animals/objects/diseases, '
        'denying their humanity).\n\n'
        'Comment: "{text}"\n\n'
        'Reply with ONLY one word: high or low.'
    )

    results = list(existing)
    remaining = comments[~comments['comment_id'].isin(done_ids)]
    print(f"Remaining: {len(remaining)}")

    t0 = time.time()
    for i, (_, row) in enumerate(remaining.iterrows()):
        text = str(row['text'])[:2000]
        cid = row['comment_id']

        z_llm = -1
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4.1',
                    messages=[{"role": "user",
                               "content": PROMPT.format(text=text)}],
                    max_tokens=10,
                    temperature=0
                )
                raw = resp.choices[0].message.content.strip().lower()
                if 'high' in raw:
                    z_llm = 1
                elif 'low' in raw:
                    z_llm = 0
                else:
                    print(f"  Unparseable '{raw}' cid={cid} attempt={attempt+1}")
                    continue
                break
            except Exception as e:
                print(f"  Error: {e}, attempt={attempt+1}")
                time.sleep(10)

        results.append({
            'comment_id': cid,
            'text': text,
            'Z_llm': z_llm,
            'Z_human': int(row['Z_human']),
            'T': int(row['T']),
            'S': row['S'],
            'Y': float(row['Y']),
        })

        done_count = i + 1
        if done_count % 50 == 0:
            elapsed = time.time() - t0
            rate = done_count / elapsed
            eta = (len(remaining) - done_count) / rate if rate > 0 else 0
            n_valid = sum(1 for r in results if r['Z_llm'] >= 0)
            print(f"  [{done_count}/{len(remaining)}] {n_valid} valid, "
                  f"{rate:.1f}/s, ETA {eta/60:.0f}min")

        if done_count % 200 == 0:
            pd.DataFrame(results).to_csv(partial_path, index=False)

        time.sleep(1.0)

    df_out = pd.DataFrame(results)
    df_out.to_csv(final_path, index=False)
    n_valid = (df_out['Z_llm'] >= 0).sum()
    print(f"Saved: {final_path} ({len(df_out)} total, {n_valid} valid)")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Analysis
# ══════════════════════════════════════════════════════════════════════════════

def build_cm(z_true, z_pred):
    C = np.zeros((2, 2))
    for zt in [0, 1]:
        for zp in [0, 1]:
            C[zt, zp] = ((z_true == zt) & (z_pred == zp)).sum()
    return C


def cm_metrics(C):
    fpr = C[0, 1] / max(C[0, :].sum(), 1)
    fnr = C[1, 0] / max(C[1, :].sum(), 1)
    mc = (C[0, 1] + C[1, 0]) / max(C.sum(), 1)
    return fpr, fnr, mc


def build_mixing(C):
    """C[z_true, z_hat] → M[z_hat, z_true] = P(Z_true=z | Z_hat=zh)."""
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        total = C[0, zh] + C[1, zh]
        if total > 0:
            M[zh, 0] = C[0, zh] / total
            M[zh, 1] = C[1, zh] / total
        else:
            M[zh, :] = 0.5
    return M


def cate_dim(Y, T, Z, z_val, mask):
    sel = mask & (Z == z_val)
    t1 = sel & (T == 1)
    t0 = sel & (T == 0)
    if t1.sum() < 2 or t0.sum() < 2:
        return np.nan
    return Y[t1].mean() - Y[t0].mean()


def compute_all_cates(Y, T, Z_human, Z_llm, S, subgroups):
    C_global = build_cm(Z_human, Z_llm)
    M_global = build_mixing(C_global)

    C_s, M_s = {}, {}
    for s in subgroups:
        ms = S == s
        C_s[s] = build_cm(Z_human[ms], Z_llm[ms])
        M_s[s] = build_mixing(C_s[s])

    rows = []
    for s in subgroups:
        ms = S == s
        oracle = np.array([cate_dim(Y, T, Z_human, z, ms) for z in [0, 1]])
        naive = np.array([cate_dim(Y, T, Z_llm, z, ms) for z in [0, 1]])

        if np.any(np.isnan(naive)):
            global_c = np.array([np.nan, np.nan])
            ec_hte = np.array([np.nan, np.nan])
        else:
            det_g = np.linalg.det(M_global)
            global_c = np.linalg.solve(M_global, naive) if abs(det_g) > 1e-10 else naive.copy()
            det_s = np.linalg.det(M_s[s])
            ec_hte = np.linalg.solve(M_s[s], naive) if abs(det_s) > 1e-10 else naive.copy()

        for z in [0, 1]:
            rows.append({
                'z': z, 's': s,
                'oracle': oracle[z], 'naive': naive[z],
                'global_corrected': global_c[z], 'ec_hte': ec_hte[z]
            })

    return rows, C_global, C_s, M_global, M_s


def bootstrap_cates(Y, T, Z_human, Z_llm, S, comment_ids, subgroups,
                    n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    unique_cids = np.unique(comment_ids)
    n_cids = len(unique_cids)

    cid_to_idx = {}
    for i, cid in enumerate(comment_ids):
        cid_to_idx.setdefault(cid, []).append(i)

    boot_rows = []
    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            print(f"  Bootstrap {b+1}/{n_boot}")

        sampled_cids = rng.choice(unique_cids, size=n_cids, replace=True)

        idx_list = []
        for cid in sampled_cids:
            idx_list.extend(cid_to_idx[cid])
        idx = np.array(idx_list)

        Y_b = Y[idx]
        T_b = T[idx]
        Zh_b = Z_human[idx]
        Zl_b = Z_llm[idx]
        S_b = S[idx]

        rows, _, _, _, _ = compute_all_cates(Y_b, T_b, Zh_b, Zl_b, S_b, subgroups)
        for r in rows:
            r['boot'] = b
        boot_rows.extend(rows)

    return pd.DataFrame(boot_rows)


def phase_analyze(n_boot=500):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load annotation results
    llm_path = os.path.join(RESULTS_DIR, 'exp004_llm_annotations.csv')
    if not os.path.exists(llm_path):
        print(f"LLM annotations not found at {llm_path} — run annotate first")
        return

    df = pd.read_csv(llm_path)
    df = df[df['Z_llm'] >= 0].copy()
    print(f"Loaded {len(df)} valid annotated comments")

    subgroups = sorted(df['S'].unique())
    print(f"Subgroups: {subgroups}")
    print(f"Subgroup sizes: {df['S'].value_counts().to_dict()}")

    Y = df['Y'].values.astype(float)
    T = df['T'].values.astype(int)
    Z_human = df['Z_human'].values.astype(int)
    Z_llm = df['Z_llm'].values.astype(int)
    S = df['S'].values
    comment_ids = df['comment_id'].values

    # ── Confusion matrices ──
    print("\n=== Confusion Matrices (Z_llm vs Z_human) ===")
    cate_rows, C_global, C_s, M_global, M_s = compute_all_cates(
        Y, T, Z_human, Z_llm, S, subgroups)

    cm_rows = []
    fpr_g, fnr_g, mc_g = cm_metrics(C_global)
    cm_rows.append({
        'subgroup': 'Global',
        'TP': C_global[1, 1], 'FP': C_global[0, 1],
        'FN': C_global[1, 0], 'TN': C_global[0, 0],
        'FPR': fpr_g, 'FNR': fnr_g, 'misclass_rate': mc_g
    })
    print(f"\nGlobal: FPR={fpr_g:.4f}  FNR={fnr_g:.4f}  misclass={mc_g:.4f}")

    for s in subgroups:
        C = C_s[s]
        fpr, fnr, mc = cm_metrics(C)
        cm_rows.append({
            'subgroup': s,
            'TP': C[1, 1], 'FP': C[0, 1],
            'FN': C[1, 0], 'TN': C[0, 0],
            'FPR': fpr, 'FNR': fnr, 'misclass_rate': mc
        })
        print(f"  {s:20s}: FPR={fpr:.4f}  FNR={fnr:.4f}  misclass={mc:.4f}")

    cm_df = pd.DataFrame(cm_rows)
    sg_rates = cm_df[cm_df['subgroup'] != 'Global']['misclass_rate']
    delta_pi = sg_rates.max() - sg_rates.min()
    print(f"\n  Δπ = {delta_pi:.4f}")
    cm_df.to_csv(os.path.join(RESULTS_DIR, 'exp004_confusion_matrices.csv'), index=False)

    # ── CATE point estimates ──
    cate_df = pd.DataFrame(cate_rows)
    print("\n=== CATE Point Estimates ===")
    for _, r in cate_df.iterrows():
        print(f"  Z={int(r['z'])} S={r['s']:20s}  "
              f"oracle={r['oracle']:+.4f}  naive={r['naive']:+.4f}  "
              f"global={r['global_corrected']:+.4f}  ec_hte={r['ec_hte']:+.4f}")

    # ── Bootstrap ──
    print(f"\n=== Bootstrap ({n_boot} resamples, comment-level) ===")
    t0 = time.time()
    boot_df = bootstrap_cates(Y, T, Z_human, Z_llm, S, comment_ids, subgroups,
                              n_boot=n_boot, seed=42)
    print(f"  Bootstrap done in {time.time()-t0:.1f}s")

    ci_rows = []
    for _, r in cate_df.iterrows():
        z, s = r['z'], r['s']
        bsub = boot_df[(boot_df['z'] == z) & (boot_df['s'] == s)]
        ci = {'z': int(z), 's': s}
        for method in ['oracle', 'naive', 'global_corrected', 'ec_hte']:
            vals = bsub[method].dropna()
            ci[method] = r[method]
            ci[f'{method}_ci_lo'] = vals.quantile(0.025) if len(vals) > 10 else np.nan
            ci[f'{method}_ci_hi'] = vals.quantile(0.975) if len(vals) > 10 else np.nan
            ci[f'{method}_se'] = vals.std() if len(vals) > 10 else np.nan
        ci_rows.append(ci)

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(RESULTS_DIR, 'exp004_cate_comparison.csv'), index=False)
    print(f"Saved: {RESULTS_DIR}/exp004_cate_comparison.csv")

    print("\n=== CATE with 95% CIs ===")
    for _, r in ci_df.iterrows():
        print(f"  Z={int(r['z'])} S={r['s']:20s}")
        for m in ['oracle', 'naive', 'global_corrected', 'ec_hte']:
            pt = r[m]
            lo, hi = r[f'{m}_ci_lo'], r[f'{m}_ci_hi']
            if np.isnan(pt):
                print(f"    {m:20s}: NaN")
            else:
                print(f"    {m:20s}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ── Figure ──
    methods = ['oracle', 'naive', 'global_corrected', 'ec_hte']
    method_labels = ['Oracle', 'Naive', 'Global', 'EC-HTE']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    n_sg = len(subgroups)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, s in enumerate(subgroups):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        pos = 0

        for z in [0, 1]:
            row = ci_df[(ci_df['z'] == z) & (ci_df['s'] == s)]
            if row.empty:
                pos += len(methods) + 1
                continue
            row = row.iloc[0]

            for m_idx, method in enumerate(methods):
                pt = row[method]
                lo = row[f'{method}_ci_lo']
                hi = row[f'{method}_ci_hi']

                if not np.isnan(pt):
                    err = [[max(pt - lo, 0) if not np.isnan(lo) else 0],
                           [max(hi - pt, 0) if not np.isnan(hi) else 0]]
                    ax.bar(pos, pt, color=colors[m_idx], width=0.8,
                           yerr=err, capsize=3,
                           label=method_labels[m_idx] if z == 0 else None)
                pos += 1
            pos += 1

        s_display = s.replace('_', ' ').title()
        ax.set_title(f'S = {s_display}', fontsize=11)

        z0_center = (len(methods) - 1) / 2
        z1_center = len(methods) + 1 + (len(methods) - 1) / 2
        ax.set_xticks([z0_center, z1_center])
        ax.set_xticklabels(['Z=0\n(low dehum.)', 'Z=1\n(high dehum.)'])
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_ylabel('CATE')
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='best')

    for ax_idx in range(len(subgroups), len(axes)):
        axes[ax_idx].set_visible(False)

    plt.suptitle('EC-HTE vs Global Correction: Hate Speech CATEs\n'
                 '(Text-Feature Subgroups)', fontsize=13, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'exp004_cate_plot.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()

    # ── Report ──
    _write_report(ci_df, cm_df, delta_pi, df, subgroups, C_global, C_s)


def _write_report(ci_df, cm_df, delta_pi, df, subgroups, C_global, C_s):
    lines = []
    lines.append("# Exp-004: UC Berkeley Hate Speech — EC-HTE Analysis")
    lines.append("## (Revised: Text-Feature Subgroups, Comment-Level)\n")

    lines.append("## Design\n")
    lines.append("- **Unit of analysis**: Comment (aggregated from multiple annotators)")
    lines.append("- **T (treatment)**: `target_race` majority vote — does the comment target a racial group?")
    lines.append("- **Y (outcome)**: `hate_speech_score` — continuous per-comment Rasch score")
    lines.append("- **Z (effect modifier)**: Dehumanization level (binary: high/low)")
    lines.append("  - Z_human: majority vote of annotator `dehumanize` >= 2")
    lines.append("  - Z_llm: GPT-4.1 text classification")
    lines.append("- **S (subgroup)**: Text features — `platform_group` (platform 0 vs platforms 2+3) × `text_length` (short vs long)")
    lines.append("")

    lines.append("## Data Summary\n")
    lines.append(f"- **Dataset**: UC Berkeley Measuring Hate Speech")
    lines.append(f"- **Comments analyzed**: {len(df)}")
    lines.append(f"- **Valid LLM annotations**: {(df['Z_llm'] >= 0).sum()}")
    for s in subgroups:
        n = (df['S'] == s).sum()
        lines.append(f"- **{s}**: {n} comments")
    lines.append("")

    lines.append("## Misclassification Heterogeneity\n")
    lines.append("### Global Confusion Matrix\n")
    lines.append("```")
    lines.append(f"           Z_llm=0  Z_llm=1")
    lines.append(f"Z_human=0  {C_global[0,0]:8.0f}  {C_global[0,1]:8.0f}")
    lines.append(f"Z_human=1  {C_global[1,0]:8.0f}  {C_global[1,1]:8.0f}")
    lines.append("```\n")

    lines.append("### Per-Subgroup Metrics\n")
    lines.append("| Subgroup | TP | FP | FN | TN | FPR | FNR | Misclass Rate |")
    lines.append("|----------|----|----|----|----|-----|-----|---------------|")
    for _, r in cm_df.iterrows():
        lines.append(f"| {r['subgroup']} | {r['TP']:.0f} | {r['FP']:.0f} | "
                     f"{r['FN']:.0f} | {r['TN']:.0f} | {r['FPR']:.4f} | "
                     f"{r['FNR']:.4f} | {r['misclass_rate']:.4f} |")
    lines.append(f"\n**Δπ** (max − min misclass rate across text-feature subgroups) = **{delta_pi:.4f}**\n")

    lines.append("### Per-Subgroup Confusion Matrices\n")
    for s in subgroups:
        C = C_s[s]
        lines.append(f"**{s}**:")
        lines.append("```")
        lines.append(f"           Z_llm=0  Z_llm=1")
        lines.append(f"Z_human=0  {C[0,0]:8.0f}  {C[0,1]:8.0f}")
        lines.append(f"Z_human=1  {C[1,0]:8.0f}  {C[1,1]:8.0f}")
        lines.append("```\n")

    lines.append("## CATE Estimates (with 95% Bootstrap CIs)\n")
    lines.append("| Z | Subgroup (S) | Oracle | Naive | Global Corrected | EC-HTE |")
    lines.append("|---|-------------|--------|-------|------------------|--------|")
    for _, r in ci_df.iterrows():
        def fmt(m):
            pt = r[m]
            lo, hi = r[f'{m}_ci_lo'], r[f'{m}_ci_hi']
            if np.isnan(pt):
                return "NaN"
            return f"{pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"| {int(r['z'])} | {r['s']} | {fmt('oracle')} | "
                     f"{fmt('naive')} | {fmt('global_corrected')} | {fmt('ec_hte')} |")
    lines.append("")

    lines.append("## Bias Decomposition\n")
    lines.append("| Z | Subgroup (S) | Naive Bias | Global Bias | EC-HTE Bias |")
    lines.append("|---|-------------|-----------|-------------|-------------|")
    for _, r in ci_df.iterrows():
        o = r['oracle']
        nb = r['naive'] - o if not np.isnan(o) else np.nan
        gb = r['global_corrected'] - o if not np.isnan(o) else np.nan
        eb = r['ec_hte'] - o if not np.isnan(o) else np.nan

        def fmt_b(v):
            return f"{v:+.4f}" if not np.isnan(v) else "NaN"

        lines.append(f"| {int(r['z'])} | {r['s']} | {fmt_b(nb)} | "
                     f"{fmt_b(gb)} | {fmt_b(eb)} |")
    lines.append("")

    avg_naive = np.nanmean([abs(r['naive'] - r['oracle']) for _, r in ci_df.iterrows()])
    avg_global = np.nanmean([abs(r['global_corrected'] - r['oracle'])
                             for _, r in ci_df.iterrows()])
    avg_echte = np.nanmean([abs(r['ec_hte'] - r['oracle']) for _, r in ci_df.iterrows()])

    lines.append("## Summary\n")
    lines.append(f"- **Average |bias|**: Naive={avg_naive:.4f}, "
                 f"Global={avg_global:.4f}, EC-HTE={avg_echte:.4f}")
    if avg_echte < avg_global:
        pct = (1 - avg_echte / max(avg_global, 1e-10)) * 100
        lines.append(f"- **EC-HTE reduces bias by {pct:.1f}% vs Global correction**")
    elif avg_echte < avg_naive:
        pct = (1 - avg_echte / max(avg_naive, 1e-10)) * 100
        lines.append(f"- EC-HTE reduces bias by {pct:.1f}% vs Naive (Global slightly better)")
    else:
        lines.append(f"- Global correction slightly outperforms EC-HTE "
                     f"(likely low heterogeneity)")

    if delta_pi > 0.03:
        lines.append(f"- **Δπ={delta_pi:.4f}** indicates meaningful misclassification "
                     f"heterogeneity across text-feature subgroups")
    else:
        lines.append(f"- Δπ={delta_pi:.4f} indicates low heterogeneity — "
                     f"Global and EC-HTE should perform similarly")

    lines.append("")
    lines.append("## Key Findings\n")
    lines.append("The subgroups S are defined by **text features** (target type × text length), "
                 "not annotator demographics. This design tests whether LLM misclassification "
                 "of dehumanization varies across **types of text**, which is the claim relevant "
                 "for EC-HTE: if the LLM error pattern depends on text characteristics, then "
                 "subgroup-specific correction (EC-HTE) should outperform a single global correction.")

    report_path = os.path.join(RESULTS_DIR, 'exp004_hate_speech_analysis.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exp-004: UC Berkeley Hate Speech EC-HTE Pipeline (Revised)')
    parser.add_argument('--phase', choices=['prep', 'annotate', 'analyze', 'all'],
                        default='all')
    parser.add_argument('--resume', action='store_true',
                        help='Resume annotation from partial results')
    parser.add_argument('--n-comments', type=int, default=2500,
                        help='Target comments to sample')
    parser.add_argument('--n-boot', type=int, default=500,
                        help='Bootstrap resamples')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.phase in ('prep', 'all'):
        phase_prep(n_comments=args.n_comments, seed=args.seed)

    if args.phase in ('annotate', 'all'):
        phase_annotate(resume=args.resume)

    if args.phase in ('analyze', 'all'):
        phase_analyze(n_boot=args.n_boot)
