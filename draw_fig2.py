import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

df = pd.read_csv('results/exp002_bias_heatmap.csv')

dm_vals = np.sort(df['delta_misclass'].unique())
dc_vals = np.sort(df['delta_cate'].unique())
n_dm, n_dc = len(dm_vals), len(dc_vals)

ratio_A = np.full((n_dc, n_dm), np.nan)
ratio_B = np.full((n_dc, n_dm), np.nan)
hurts_B = np.full((n_dc, n_dm), False)

for _, row in df.iterrows():
    i_dm = np.searchsorted(dm_vals, row['delta_misclass'])
    i_dc = np.searchsorted(dc_vals, row['delta_cate'])
    ratio_A[i_dc, i_dm] = row['max_bias_ratio_A']
    ratio_B[i_dc, i_dm] = row['max_bias_ratio_B']
    hurts_B[i_dc, i_dm] = row['debiasing_hurts_B']

fig = plt.figure(figsize=(7.5, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.30)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])

vmax_global = min(np.nanmax(ratio_B), 5.0)
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax_global)
cmap = plt.cm.RdBu_r

# --- Panel A ---
im_a = ax_a.pcolormesh(dm_vals, dc_vals, ratio_A, cmap=cmap, norm=norm,
                        shading='nearest', rasterized=True)
ax_a.set_xlabel(r'$\Delta\pi$ (misclassification gap)')
ax_a.set_ylabel(r'$\Delta\tau$ (CATE heterogeneity)')
ax_a.set_title('(a) Subgroup A (low error)', fontsize=11, pad=6)
ax_a.spines['top'].set_visible(True)
ax_a.spines['right'].set_visible(True)

# --- Panel B ---
im_b = ax_b.pcolormesh(dm_vals, dc_vals, ratio_B, cmap=cmap, norm=norm,
                        shading='nearest', rasterized=True)

cs = ax_b.contour(dm_vals, dc_vals, ratio_B, levels=[1.0],
                   colors='black', linewidths=1.5, linestyles='--')
ax_b.text(0.115, 0.46, 'ratio = 1', fontsize=8, color='black', rotation=90,
          ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85,
                    edgecolor='none'))

hurts_mask = hurts_B.astype(float)
ax_b.contourf(dm_vals, dc_vals, hurts_mask, levels=[0.5, 1.5],
              hatches=['///'], colors='none', alpha=0)

ax_b.set_xlabel(r'$\Delta\pi$ (misclassification gap)')
ax_b.set_ylabel('')
ax_b.set_title('(b) Subgroup B (high error)', fontsize=11, pad=6)
ax_b.spines['top'].set_visible(True)
ax_b.spines['right'].set_visible(True)

ax_b.annotate('debiasing\nhurts', xy=(0.20, 0.30), fontsize=9, color='#B71C1C',
              fontweight='bold', ha='center', va='center',
              path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
ax_b.annotate('debiasing\nhelps', xy=(0.06, 0.30), fontsize=9, color='#1565C0',
              fontweight='bold', ha='center', va='center',
              path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

cbar = fig.colorbar(im_b, cax=cax)
cbar.set_label(r'Bias ratio ($|\mathrm{bias_{global}}|/|\mathrm{bias_{naive}}|$)',
               fontsize=9)
cbar.ax.axhline(1.0, color='black', linewidth=1.0)
cbar.ax.tick_params(labelsize=9)

out_base = 'artifacts/figures/fig2_heatmap'
plt.savefig(f'{out_base}.pdf')
plt.savefig(f'{out_base}.png')
plt.close()
print(f'Saved: {out_base}.pdf and {out_base}.png')
