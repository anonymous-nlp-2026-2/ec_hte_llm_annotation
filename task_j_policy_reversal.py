"""
Task J: Policy Reversal Zone τ Sweep

Sweeps τ(0) values and plots "policy reversal probability vs true CATE magnitude".
Sign reversal = CATE estimate has opposite sign to true CATE.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import os

# ── Config ──────────────────────────────────────────────────────
BASE_MISCLASS = 0.15
TAU_Z0_SWEEP = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
ARTIFACTS_DIR = os.path.expanduser(
    '~/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Load exp002 CSV ─────────────────────────────────────────────
df_all = pd.read_csv('results/exp002_bias_bound.csv')
df_z0 = df_all[df_all['z_value'] == 0].copy()

# Compute naive bias: bias_naive(z=0, s) = π_s × Δτ
# π_A = BASE + dm/2, π_B = BASE - dm/2
df_z0['pi_s'] = np.where(
    df_z0['subgroup'] == 'A',
    BASE_MISCLASS + df_z0['delta_misclass'] / 2.0,
    BASE_MISCLASS - df_z0['delta_misclass'] / 2.0
)
df_z0['bias_naive'] = df_z0['pi_s'] * df_z0['delta_cate']
df_z0['bias_global'] = df_z0['analytic_bias']

# ── Sweep τ(0) ──────────────────────────────────────────────────
rows = []
for tau0 in TAU_Z0_SWEEP:
    for _, r in df_z0.iterrows():
        dm = r['delta_misclass']
        dc = r['delta_cate']
        sg = r['subgroup']

        true_cate = tau0  # τ(0) > 0 always in our sweep

        est_naive = tau0 + r['bias_naive']
        est_global = tau0 + r['bias_global']

        rev_naive = (true_cate > 0 and est_naive < 0) or (true_cate < 0 and est_naive > 0)
        rev_global = (true_cate > 0 and est_global < 0) or (true_cate < 0 and est_global > 0)

        rows.append({
            'tau_z0': tau0,
            'delta_misclass': dm,
            'delta_cate': dc,
            'subgroup': sg,
            'true_cate': true_cate,
            'est_naive': est_naive,
            'est_global': est_global,
            'reversal_naive': rev_naive,
            'reversal_global': rev_global,
        })

df_sweep = pd.DataFrame(rows)

# ── Aggregate stats ─────────────────────────────────────────────

# Overall reversal rate by τ(0)
agg_overall = df_sweep.groupby('tau_z0').agg(
    n_configs=('reversal_global', 'count'),
    n_rev_global=('reversal_global', 'sum'),
    n_rev_naive=('reversal_naive', 'sum'),
).reset_index()
agg_overall['pct_rev_global'] = 100.0 * agg_overall['n_rev_global'] / agg_overall['n_configs']
agg_overall['pct_rev_naive'] = 100.0 * agg_overall['n_rev_naive'] / agg_overall['n_configs']

# By (τ, δ_misclass)
agg_dm = df_sweep.groupby(['tau_z0', 'delta_misclass']).agg(
    n_configs=('reversal_global', 'count'),
    n_rev_global=('reversal_global', 'sum'),
    n_rev_naive=('reversal_naive', 'sum'),
).reset_index()
agg_dm['pct_rev_global'] = 100.0 * agg_dm['n_rev_global'] / agg_dm['n_configs']
agg_dm['pct_rev_naive'] = 100.0 * agg_dm['n_rev_naive'] / agg_dm['n_configs']

# By (τ, δ_misclass, δ_cate)
agg_full = df_sweep.groupby(['tau_z0', 'delta_misclass', 'delta_cate']).agg(
    n_configs=('reversal_global', 'count'),
    n_rev_global=('reversal_global', 'sum'),
).reset_index()
agg_full['pct_rev_global'] = 100.0 * agg_full['n_rev_global'] / agg_full['n_configs']

# ── Split by δ_misclass high/low for Panel A ────────────────────
dm_vals = sorted(df_z0['delta_misclass'].unique())
dm_median = np.median(dm_vals)

agg_dm_split = df_sweep.copy()
agg_dm_split['dm_group'] = np.where(
    agg_dm_split['delta_misclass'] <= dm_median, 'low', 'high')

agg_split = agg_dm_split.groupby(['tau_z0', 'dm_group']).agg(
    n_configs=('reversal_global', 'count'),
    n_rev_global=('reversal_global', 'sum'),
).reset_index()
agg_split['pct_rev_global'] = 100.0 * agg_split['n_rev_global'] / agg_split['n_configs']

# ── Save CSV ────────────────────────────────────────────────────
csv_path = os.path.join(ARTIFACTS_DIR, 'policy_reversal_zone.csv')
agg_full.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")

# ── Key numbers ─────────────────────────────────────────────────
print("\n=== Key Results ===")

# Critical τ threshold (below which >50% configs have reversal)
above50 = agg_overall[agg_overall['pct_rev_global'] > 50]
if len(above50) > 0:
    crit_tau = above50['tau_z0'].max()
    print(f"Critical τ threshold (>50% reversal): τ(0) ≤ {crit_tau}")
else:
    # Find highest reversal rate
    max_rev = agg_overall.loc[agg_overall['pct_rev_global'].idxmax()]
    print(f"No τ with >50% overall reversal. Max: {max_rev['pct_rev_global']:.1f}% at τ={max_rev['tau_z0']}")
    # Check per dm_group
    high_dm = agg_split[agg_split['dm_group'] == 'high']
    above50_high = high_dm[high_dm['pct_rev_global'] > 50]
    if len(above50_high) > 0:
        crit_tau_high = above50_high['tau_z0'].max()
        print(f"  For high δ_misclass: >50% reversal at τ(0) ≤ {crit_tau_high}")

# τ(0)=0.03 reversal rate
r003 = agg_overall[agg_overall['tau_z0'] == 0.03]
if len(r003) > 0:
    print(f"τ(0)=0.03: {r003.iloc[0]['pct_rev_global']:.1f}% global reversal, "
          f"{r003.iloc[0]['pct_rev_naive']:.1f}% naive reversal")

# Most dangerous combination
worst = agg_dm.loc[agg_dm['pct_rev_global'].idxmax()]
print(f"Most dangerous combo: τ(0)={worst['tau_z0']}, δ_misclass={worst['delta_misclass']}, "
      f"reversal={worst['pct_rev_global']:.1f}%")

print(f"\nOverall reversal rates:")
for _, row in agg_overall.iterrows():
    print(f"  τ(0)={row['tau_z0']:.2f}: global={row['pct_rev_global']:5.1f}%, "
          f"naive={row['pct_rev_naive']:5.1f}%")

# ── Visualization ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.15]})

# Panel A: Reversal probability vs τ(0) curves
ax = axes[0]

# Overall curve
ax.plot(agg_overall['tau_z0'], agg_overall['pct_rev_global'],
        'o-', color='#c0392b', lw=2.5, ms=7, label='All configs (global)', zorder=5)
ax.plot(agg_overall['tau_z0'], agg_overall['pct_rev_naive'],
        's--', color='#7f8c8d', lw=1.8, ms=5, label='All configs (naive)', zorder=4)

# Split by high/low misclass
for dm_grp, style, color, lbl in [
    ('high', 'D-', '#e74c3c', r'High $\delta_{misclass}$'),
    ('low', 'v-', '#2980b9', r'Low $\delta_{misclass}$'),
]:
    sub = agg_split[agg_split['dm_group'] == dm_grp]
    ax.plot(sub['tau_z0'], sub['pct_rev_global'],
            style, color=color, lw=1.5, ms=5, alpha=0.8, label=lbl)

# Danger zone shading
danger_tau = agg_overall[agg_overall['pct_rev_global'] > 5]['tau_z0']
if len(danger_tau) > 0:
    ax.axvspan(0, danger_tau.max(), alpha=0.08, color='red', zorder=0)
    ax.text(danger_tau.max() / 2, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 10 else 85,
            'DANGER\nZONE', ha='center', va='top', fontsize=11, fontweight='bold',
            color='#c0392b', alpha=0.7)

ax.set_xlabel(r'True CATE $\tau(0)$', fontsize=13)
ax.set_ylabel('Sign reversal probability (%)', fontsize=13)
ax.set_title('A. Policy reversal probability', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.set_xlim(0, max(TAU_Z0_SWEEP) * 1.05)
ax.set_ylim(-2, max(agg_overall['pct_rev_global'].max(),
                     agg_split['pct_rev_global'].max()) * 1.15)
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Heatmap — x=τ(0), y=δ_misclass, color=reversal probability
ax2 = axes[1]

pivot = agg_dm.pivot_table(
    index='delta_misclass', columns='tau_z0', values='pct_rev_global', aggfunc='mean')
pivot = pivot.sort_index(ascending=False)

# Custom red colormap
cmap = LinearSegmentedColormap.from_list(
    'reversal', ['#ffffff', '#fee0d2', '#fc9272', '#de2d26', '#67000d'])

im = ax2.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0,
                vmax=max(pivot.values.max(), 1), interpolation='nearest')

ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels([f'{v:.2f}' for v in pivot.columns], fontsize=9, rotation=45)
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=8)

ax2.set_xlabel(r'True CATE $\tau(0)$', fontsize=13)
ax2.set_ylabel(r'$\delta_{misclass}$', fontsize=13)
ax2.set_title('B. Reversal probability heatmap (%)', fontsize=14, fontweight='bold')

# Annotate cells with values > 0
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if val > 0:
            text_color = 'white' if val > 50 else 'black'
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=7, color=text_color, fontweight='bold')

cb = fig.colorbar(im, ax=ax2, shrink=0.85, label='Reversal probability (%)')
cb.ax.tick_params(labelsize=10)

plt.tight_layout()

for ext in ['pdf', 'png']:
    path = os.path.join(ARTIFACTS_DIR, f'policy_reversal_zone.{ext}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved {path}")

plt.close()
print("\nDone.")
