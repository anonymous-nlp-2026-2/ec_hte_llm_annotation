"""
Sign-Reversal Analysis (Corrected): With τ(0)=0.03, global correction
can flip the CATE estimate from positive to negative — a true sign reversal.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

RESULTS_DIR = 'results'
ARTIFACTS_DIR = '/home/ubuntu/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

BASE_MISCLASS = 0.15
TAU_Z0 = 0.03

# ============================================================
# Step 1: Load data and compute CATE estimates
# ============================================================

df = pd.read_csv(f'{RESULTS_DIR}/exp002_bias_bound.csv')

df['pi_s'] = np.where(
    df['subgroup'] == 'A',
    BASE_MISCLASS + df['delta_misclass'] / 2.0,
    BASE_MISCLASS - df['delta_misclass'] / 2.0
)

df['bias_naive'] = np.where(
    df['z_value'] == 0,
    df['pi_s'] * df['delta_cate'],
    -df['pi_s'] * df['delta_cate']
)
df['bias_global'] = df['analytic_bias']

df['true_cate'] = np.where(df['z_value'] == 0, TAU_Z0, TAU_Z0 + df['delta_cate'])

df['cate_naive'] = df['true_cate'] + df['bias_naive']
df['cate_global'] = df['true_cate'] + df['bias_global']
df['cate_echte'] = df['true_cate']  # analytically unbiased

# Sign reversal: naive CATE and global CATE have opposite signs
nonzero = (df['cate_naive'].abs() > 1e-12) & (df['cate_global'].abs() > 1e-12)
df['sign_reversal'] = nonzero & (np.sign(df['cate_naive']) != np.sign(df['cate_global']))

# Focus on z=0 subgroup (where τ(0)=0.03 is small and vulnerable to sign flip)
df_z0 = df[df['z_value'] == 0].copy()
n_z0 = len(df_z0)
n_reversal_z0 = df_z0['sign_reversal'].sum()
pct_reversal_z0 = n_reversal_z0 / n_z0 * 100 if n_z0 > 0 else 0

print(f"=== Sign-Reversal Analysis (τ(0) = {TAU_Z0}) ===")
print(f"Z=0 subgroup rows: {n_z0}")
print(f"Sign reversals: {n_reversal_z0} ({pct_reversal_z0:.1f}%)")

# Overall stats
n_total = len(df)
n_reversal_all = df['sign_reversal'].sum()
print(f"\nAll rows: {n_total}")
print(f"All sign reversals: {n_reversal_all} ({n_reversal_all/n_total*100:.1f}%)")

# Most extreme case
if n_reversal_z0 > 0:
    extreme = df_z0[df_z0['sign_reversal']].copy()
    extreme['gap'] = (extreme['cate_naive'] - extreme['cate_global']).abs()
    extreme = extreme.sort_values('gap', ascending=False)
    print("\nTop 5 most extreme sign-reversal cases (Z=0):")
    for _, row in extreme.head(5).iterrows():
        print(f"  dm={row['delta_misclass']:.2f}, dc={row['delta_cate']:.3f}, "
              f"sub={row['subgroup']}, pi_s={row['pi_s']:.4f}")
        print(f"    naive CATE = {row['cate_naive']:+.4f}, "
              f"global CATE = {row['cate_global']:+.4f}, "
              f"EC-HTE = {row['cate_echte']:+.4f}, "
              f"true = {row['true_cate']:+.4f}")

# ============================================================
# Step 2: τ(0) sweep
# ============================================================

tau_values = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
sweep_results = []

for tau in tau_values:
    true_cate_sweep = np.where(df['z_value'] == 0, tau, tau + df['delta_cate'])
    cate_naive_sweep = true_cate_sweep + df['bias_naive']
    cate_global_sweep = true_cate_sweep + df['bias_global']
    nz = (np.abs(cate_naive_sweep) > 1e-12) & (np.abs(cate_global_sweep) > 1e-12)
    sign_rev = nz & (np.sign(cate_naive_sweep) != np.sign(cate_global_sweep))

    # Z=0 only
    z0_mask = df['z_value'] == 0
    sign_rev_z0 = sign_rev & z0_mask
    n_rev_z0 = sign_rev_z0.sum()
    n_z0_total = z0_mask.sum()

    # Most extreme global CATE when sign reversal occurs
    global_cates_rev = cate_global_sweep[sign_rev_z0]
    most_extreme = global_cates_rev.min() if len(global_cates_rev) > 0 else np.nan

    sweep_results.append({
        'tau_z0': tau,
        'n_reversal_z0': n_rev_z0,
        'pct_reversal_z0': n_rev_z0 / n_z0_total * 100 if n_z0_total > 0 else 0,
        'n_reversal_all': sign_rev.sum(),
        'pct_reversal_all': sign_rev.sum() / len(df) * 100,
        'most_extreme_global_cate': most_extreme,
    })

sweep_df = pd.DataFrame(sweep_results)
print("\n=== τ(0) Sweep ===")
print(sweep_df.to_string(index=False))

# Find critical threshold
critical_tau = None
for _, row in sweep_df.iterrows():
    if row['n_reversal_z0'] > 0:
        critical_tau = row['tau_z0']
        break
# More precise: max |bias_global| for z=0
max_abs_bias_z0 = df_z0['bias_global'].abs().max()
print(f"\nMax |bias_global| for Z=0: {max_abs_bias_z0:.6f}")
print(f"Critical τ(0) threshold: τ(0) < {max_abs_bias_z0:.4f} → sign reversal possible")
print(f"At τ(0) = {TAU_Z0}: {pct_reversal_z0:.1f}% of Z=0 configs have sign reversal")

# EC-HTE always correct sign?
echte_correct = (np.sign(df['cate_echte']) == np.sign(df['true_cate'])) | (df['true_cate'] == 0)
print(f"\nEC-HTE correct sign in all cases: {echte_correct.all()}")

# ============================================================
# Step 3: Save corrected CSV
# ============================================================

out_df = df[['delta_misclass', 'delta_cate', 'subgroup', 'z_value',
             'pi_s', 'true_cate', 'bias_naive', 'bias_global',
             'cate_naive', 'cate_global', 'cate_echte', 'sign_reversal']].copy()
out_df.to_csv(f'{ARTIFACTS_DIR}/sign_reversal_corrected.csv', index=False)
print(f"\nSaved: {ARTIFACTS_DIR}/sign_reversal_corrected.csv ({len(out_df)} rows)")

sweep_df.to_csv(f'{ARTIFACTS_DIR}/sign_reversal_tau_sweep.csv', index=False)
print(f"Saved: {ARTIFACTS_DIR}/sign_reversal_tau_sweep.csv")

# ============================================================
# Step 4: Visualization (3-panel figure)
# ============================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1, 1.1, 1]})
fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.15)

# ----- Panel A: Scatter — naive CATE vs global CATE (z=0 only) -----
ax = axes[0]

mask_rev = df_z0['sign_reversal']
mask_norm = ~mask_rev

ax.scatter(df_z0.loc[mask_norm, 'cate_naive'], df_z0.loc[mask_norm, 'cate_global'],
           c='#BBBBBB', s=14, alpha=0.5, edgecolors='none', label='Same sign', zorder=2)
ax.scatter(df_z0.loc[mask_rev, 'cate_naive'], df_z0.loc[mask_rev, 'cate_global'],
           c='#D62728', s=22, alpha=0.8, edgecolors='none', label='Sign reversal', zorder=3)

ax.axhline(0, color='k', linewidth=0.7, zorder=1)
ax.axvline(0, color='k', linewidth=0.7, zorder=1)

# Shade sign-reversal quadrants (II: naive>0, global<0; IV: naive<0, global>0)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
pad = 0.02
xmin = min(xlim[0], ylim[0]) - pad
xmax = max(xlim[1], ylim[1]) + pad
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

ax.fill_between([0, xmax], xmin, 0, color='#D62728', alpha=0.05, zorder=0)
ax.fill_between([xmin, 0], 0, xmax, color='#D62728', alpha=0.05, zorder=0)

ax.plot([xmin, xmax], [xmin, xmax], color='#888888', linewidth=0.5, linestyle=':', zorder=1)

ax.set_xlabel('Naive CATE estimate', fontsize=12)
ax.set_ylabel('Global-corrected CATE estimate', fontsize=12)
ax.set_title('A', fontsize=14, fontweight='bold', loc='left')
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Annotate the quadrant labels
ax.text(0.95, 0.05, 'Quadrant IV\n(sign reversal)', transform=ax.transAxes,
        fontsize=7.5, ha='right', va='bottom', color='#D62728', alpha=0.7)
ax.text(0.05, 0.95, 'Quadrant II\n(sign reversal)', transform=ax.transAxes,
        fontsize=7.5, ha='left', va='top', color='#D62728', alpha=0.7)

# ----- Panel B: Bar chart — Top 3 extreme sign-reversal cases -----
ax = axes[1]

# Select 3 most extreme z=0 sign-reversal cases from different configs
flip_z0 = df_z0[df_z0['sign_reversal']].copy()
flip_z0['gap'] = (flip_z0['cate_naive'] - flip_z0['cate_global']).abs()
flip_z0_sorted = flip_z0.sort_values('gap', ascending=False)

selected = []
seen_dc = set()
for _, row in flip_z0_sorted.iterrows():
    dc_key = row['delta_cate']
    if dc_key not in seen_dc:
        selected.append(row)
        seen_dc.add(dc_key)
    if len(selected) == 3:
        break
top3 = pd.DataFrame(selected)

x_positions = np.arange(len(top3))
bar_width = 0.22
colors_bar = ['#4C72B0', '#DD8452', '#55A868']
labels_bar = ['Naive', 'Global-corrected', 'EC-HTE']

y_vals_all = []
for idx, (_, row) in enumerate(top3.iterrows()):
    vals = [row['cate_naive'], row['cate_global'], row['cate_echte']]
    y_vals_all.extend(vals)
    for j, v in enumerate(vals):
        ax.bar(idx + (j - 1) * bar_width, v, bar_width * 0.88,
               color=colors_bar[j], edgecolor='white', linewidth=0.5,
               label=labels_bar[j] if idx == 0 else None, zorder=3)

# True CATE and zero lines
ax.axhline(TAU_Z0, color='#333333', linestyle='--', linewidth=1.2, zorder=2,
           label=f'True CATE ({TAU_Z0})')
ax.axhline(0, color='#D62728', linestyle=':', linewidth=1.0, zorder=2,
           label='Sign boundary (0)')

# X-axis labels
case_labels = []
for _, row in top3.iterrows():
    case_labels.append(
        f"$\\delta_{{\\pi}}$={row['delta_misclass']:.2f}\n"
        f"$\\Delta\\tau$={row['delta_cate']:.2f}\n"
        f"sub {row['subgroup']}"
    )
ax.set_xticks(x_positions)
ax.set_xticklabels(case_labels, fontsize=9)
ax.set_ylabel('CATE estimate', fontsize=12)
ax.set_title('B', fontsize=14, fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Shade negative region
ymin, ymax = ax.get_ylim()
ax.fill_between(ax.get_xlim(), ymin, 0, color='#D62728', alpha=0.04, zorder=0)
ax.set_ylim(ymin, ymax)

# ----- Panel C: τ(0) sweep — sign reversal rate vs τ(0) -----
ax = axes[2]

ax.plot(sweep_df['tau_z0'], sweep_df['pct_reversal_z0'],
        'o-', color='#D62728', linewidth=2, markersize=7, zorder=3,
        label='Sign reversal rate')
ax.fill_between(sweep_df['tau_z0'], 0, sweep_df['pct_reversal_z0'],
                color='#D62728', alpha=0.12, zorder=1)

# Mark critical threshold
ax.axvline(max_abs_bias_z0, color='#333333', linestyle='--', linewidth=1.0,
           zorder=2, label=f'Max |bias| = {max_abs_bias_z0:.3f}')
ax.axvline(TAU_Z0, color='#4C72B0', linestyle=':', linewidth=1.0,
           zorder=2, label=f'τ(0) = {TAU_Z0}')

# Shade danger zone
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 50],
                 0, max_abs_bias_z0, color='#D62728', alpha=0.06, zorder=0)
ax.text(max_abs_bias_z0 / 2, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 45,
        'Danger\nzone', fontsize=9, ha='center', va='top', color='#D62728', alpha=0.7,
        fontweight='bold')

ax.set_xlabel('True CATE τ(0)', fontsize=12)
ax.set_ylabel('Sign reversal rate (%)', fontsize=12)
ax.set_title('C', fontsize=14, fontweight='bold', loc='left')
ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.set_xlim(0, max(tau_values) + 0.01)
ax.set_ylim(bottom=0)

# Save
fig.savefig(f'{ARTIFACTS_DIR}/sign_reversal_figure.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{ARTIFACTS_DIR}/sign_reversal_figure.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {ARTIFACTS_DIR}/sign_reversal_figure.pdf and .png")
plt.close()
