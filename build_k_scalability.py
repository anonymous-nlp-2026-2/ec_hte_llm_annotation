#!/usr/bin/env python3
"""Build K-scalability comparison table and figure from K=2, K=4, K=8 results."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARTIFACTS = os.path.expanduser(
    '~/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts/')

# ── K=2 and K=4 from exp_centered_eb_table1_dim.csv ────────────────────────

dim = pd.read_csv('results/exp_centered_eb_table1_dim.csv')

def agg_dim(dim_df, config_name, k_val):
    sub = dim_df[dim_df['config'] == config_name].copy()
    rows = []
    method_map = {
        'naive': 'Naive',
        'global_corrected': 'Global',
        'hb_centered_eb': 'EC-HTE',
        'oracle': 'Oracle',
    }
    for method_raw, method_nice in method_map.items():
        ms = sub[sub['method'] == method_raw]
        if ms.empty:
            continue
        abs_bias = ms[['abs_bias_z0', 'abs_bias_z1']].values.mean()
        rmse = ms[['rmse_z0', 'rmse_z1']].values.mean()
        coverage = ms[['coverage_z0', 'coverage_z1']].values.mean()
        rows.append({
            'K': k_val,
            'Method': method_nice,
            '|Bias|': abs_bias,
            'RMSE': rmse,
            'Coverage': coverage,
        })
    return pd.DataFrame(rows)

k2 = agg_dim(dim, 'DIM_K2_extreme', 2)
k4 = agg_dim(dim, 'DIM_K4_extreme', 4)

# ── K=8 from exp_k8_subgroups_summary.csv ───────────────────────────────────

k8_raw = pd.read_csv('results/exp_k8_subgroups_summary.csv')
method_map_k8 = {
    'Naive': 'Naive',
    'Global': 'Global',
    'HB EC-HTE': 'EC-HTE',
    'Oracle': 'Oracle',
}
k8_rows = []
for _, r in k8_raw.iterrows():
    if r['method'] in method_map_k8:
        k8_rows.append({
            'K': 8,
            'Method': method_map_k8[r['method']],
            '|Bias|': r['mean_abs_bias'],
            'RMSE': r['rmse'],
            'Coverage': r['coverage'],
        })
k8 = pd.DataFrame(k8_rows)

# ── Combine ─────────────────────────────────────────────────────────────────

df = pd.concat([k2, k4, k8], ignore_index=True)
method_order = ['Oracle', 'Naive', 'Global', 'EC-HTE']
df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
df = df.sort_values(['K', 'Method']).reset_index(drop=True)

os.makedirs(ARTIFACTS, exist_ok=True)
df.to_csv(os.path.join(ARTIFACTS, 'k_scalability_comparison.csv'), index=False)

print("=" * 65)
print("K-SCALABILITY COMPARISON TABLE")
print("=" * 65)
print(f"{'K':>3}  {'Method':>8}  {'|Bias|':>8}  {'RMSE':>8}  {'Coverage':>8}")
print("-" * 65)
for _, r in df.iterrows():
    print(f"{r['K']:>3}  {r['Method']:>8}  {r['|Bias|']:8.4f}  {r['RMSE']:8.4f}  {r['Coverage']:8.3f}")

# ── K=8 adaptive results ────────────────────────────────────────────────────

k8a_raw = pd.read_csv('results/exp_k8_adaptive_summary.csv')
print("\n" + "=" * 65)
print("K=8 ADAPTIVE VARIANTS")
print("=" * 65)
print(f"{'Method':>18}  {'|Bias|':>8}  {'RMSE':>8}  {'Coverage':>8}")
print("-" * 50)
for _, r in k8a_raw.iterrows():
    print(f"{r['method']:>18}  {r['mean_abs_bias']:8.4f}  {r['rmse']:8.4f}  {r['coverage']:8.3f}")

# ── Visualization ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
methods_plot = ['Naive', 'Global', 'EC-HTE']
colors = {'Naive': '#e74c3c', 'Global': '#3498db', 'EC-HTE': '#2ecc71'}
ks = [2, 4, 8]

for ax, metric, label in zip(axes, ['|Bias|', 'RMSE', 'Coverage'],
                              ['|Bias|', 'RMSE', 'Coverage']):
    for m in methods_plot:
        vals = [df[(df['K'] == k) & (df['Method'] == m)][metric].values[0] for k in ks]
        ax.plot(ks, vals, 'o-', label=m, color=colors[m], linewidth=2, markersize=8)
    if metric == 'Coverage':
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% nominal')
    ax.set_xlabel('K (number of subgroups)')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(ARTIFACTS, f'k_scalability_figure.{ext}'), dpi=200, bbox_inches='tight')
plt.close()

# ── Bar chart ───────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(ks))
width = 0.25

for ax, metric, title in zip(axes2, ['|Bias|', 'RMSE'], ['|Bias|', 'RMSE']):
    for i, m in enumerate(methods_plot):
        vals = [df[(df['K'] == k) & (df['Method'] == m)][metric].values[0] for k in ks]
        ax.bar(x + i * width, vals, width, label=m, color=colors[m])
    ax.set_xlabel('K')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'K={k}' for k in ks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig2.savefig(os.path.join(ARTIFACTS, f'k_scalability_bars.{ext}'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nSaved: {ARTIFACTS}k_scalability_comparison.csv")
print(f"Saved: {ARTIFACTS}k_scalability_figure.pdf/png")
print(f"Saved: {ARTIFACTS}k_scalability_bars.pdf/png")
