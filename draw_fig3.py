import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.6,
})

# Colors (project palette — EC-HTE = green, consistent with Fig 1)
COLORS = {
    'global_corrected': '#0072B2',
    'hb_ec_hte': '#009E73',
    'stratified_mle': '#E69F00',
}
LINESTYLES = {
    'global_corrected': '-',
    'hb_ec_hte': '-',
    'stratified_mle': '--',
}
LABELS = {
    'global_corrected': 'Global Corrected',
    'hb_ec_hte': 'HB EC-HTE',
    'stratified_mle': 'Stratified MLE',
}

# Load data
df = pd.read_csv('/home/ubuntu/ec_hte_llm_annotation/results/exp006_budget_sensitivity.csv')
crossover_df = pd.read_csv('/home/ubuntu/ec_hte_llm_annotation/results/exp006_crossover.csv')

# Filter to methods of interest
methods = ['global_corrected', 'hb_ec_hte', 'stratified_mle']
df = df[df['method'].isin(methods)].copy()

# Compute average RMSE: mean of rmse_z0 and rmse_z1, then average across subgroups
df['avg_rmse'] = (df['rmse_z0'] + df['rmse_z1']) / 2.0

agg = df.groupby(['regime', 'n_expert', 'k_subgroups', 'method']).agg(
    mean_rmse=('avg_rmse', 'mean'),
    std_rmse=('avg_rmse', 'std'),
).reset_index()

# Panel layout: rows=K(2,4), cols=regime(extreme, moderate)
fig, axes = plt.subplots(2, 2, figsize=(7.5, 3.2), sharex=True)

panel_configs = [
    (0, 0, 2, 'extreme'),
    (0, 1, 2, 'moderate'),
    (1, 0, 4, 'extreme'),
    (1, 1, 4, 'moderate'),
]

panel_titles = {
    (2, 'extreme'): 'K=2, Extreme',
    (2, 'moderate'): 'K=2, Moderate',
    (4, 'extreme'): 'K=4, Extreme',
    (4, 'moderate'): 'K=4, Moderate',
}

ylim_caps = {2: 0.45, 4: 0.55}

for row, col, k, regime in panel_configs:
    ax = axes[row, col]
    subset = agg[(agg['k_subgroups'] == k) & (agg['regime'] == regime)]

    for method in methods:
        mdata = subset[subset['method'] == method].sort_values('n_expert')
        rmse_vals = mdata['mean_rmse'].values
        std_vals = mdata['std_rmse'].values
        n_vals = mdata['n_expert'].values

        ax.plot(
            n_vals, rmse_vals,
            color=COLORS[method], linestyle=LINESTYLES[method],
            label=LABELS[method], marker='o', markersize=4,
        )
        ax.fill_between(
            n_vals,
            np.clip(rmse_vals - std_vals, 0, None),
            rmse_vals + std_vals,
            color=COLORS[method], alpha=0.10,
        )

    ax.set_xscale('log')
    ax.set_xticks([50, 100, 250, 500, 1000, 2000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, ylim_caps[k])
    ax.set_title(panel_titles[(k, regime)], fontsize=10, pad=4)

    cross_row = crossover_df[(crossover_df['k_subgroups'] == k) & (crossover_df['regime'] == regime)]
    cross_points = cross_row[cross_row['hb_beats_global'] == True]

    if len(cross_points) > 0:
        crossover_n = cross_points['n_expert'].min()
        ax.axvline(crossover_n, color='#555555', linestyle=':', linewidth=1.0, alpha=0.7)
        t = ax.text(
            crossover_n * 1.15, ylim_caps[k] * 0.92,
            f'n={crossover_n}', fontsize=7.5, color='#555555',
            ha='left', va='top',
        )
        t.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])
    else:
        t = ax.text(
            0.97, 0.95, 'No crossover\n(Global always lower)',
            transform=ax.transAxes, fontsize=7, color='#555555',
            ha='right', va='top', fontstyle='italic',
        )
        t.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

    if row == 1:
        ax.set_xlabel('Expert budget ($n$)')
    if col == 0:
        ax.set_ylabel('Avg RMSE')

# Single legend at top
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)

plt.tight_layout(h_pad=0.8, w_pad=0.6)

outdir = '/home/ubuntu/ec_hte_llm_annotation/artifacts/figures'
paper_dir = '/home/ubuntu/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/docs/paper/figures'
fig.savefig(f'{outdir}/fig3_budget.pdf', bbox_inches='tight')
fig.savefig(f'{outdir}/fig3_budget.png', bbox_inches='tight')
fig.savefig(f'{paper_dir}/fig3_budget.pdf', bbox_inches='tight')
fig.savefig(f'{paper_dir}/fig3_budget.png', bbox_inches='tight')
plt.close()
print(f'Saved: {outdir}/fig3_budget.pdf/.png')
print(f'Saved: {paper_dir}/fig3_budget.pdf/.png')
