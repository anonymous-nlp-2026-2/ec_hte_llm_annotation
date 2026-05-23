"""
Applicability Frontier: When does EC-HTE help?
Combines data from exp-001, exp-003, exp-013, exp-015.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# ── Data extraction ──────────────────────────────────────────────────────

def load_exp001():
    df = pd.read_csv('results/exp001_hb_results_v2.csv')
    df = df[df['regime'] == 'extreme']
    df['avg_rmse'] = (df['rmse_z0'] + df['rmse_z1']) / 2

    rows = []
    for k in [2, 4]:
        worst_misclass = {2: 0.25, 4: 0.30}[k]
        hb = df[(df['method'] == 'hb_dirichlet') & (df['k_subgroups'] == k)]
        gl = df[(df['method'] == 'global_corrected') & (df['k_subgroups'] == k)]
        hb_avg = hb.groupby('n_expert')['avg_rmse'].mean()
        gl_avg = gl.groupby('n_expert')['avg_rmse'].mean()
        for n in hb_avg.index:
            ratio = hb_avg[n] / gl_avg[n]
            rows.append({
                'exp': f'exp-001 (K={k})',
                'n_per_K': n / k,
                'worst_misclass': worst_misclass,
                'hb_rmse': hb_avg[n],
                'gl_rmse': gl_avg[n],
                'ratio': ratio,
            })
    return pd.DataFrame(rows)


def load_exp003():
    df = pd.read_csv('results/exp003_phase_a_v3.csv')
    df = df[(df['regime'] == 'extreme') & (df['level'] != 'marginal')]
    df['avg_rmse'] = (df['rmse_z0'] + df['rmse_z1']) / 2

    K = 4
    worst_misclass = 0.25
    hb = df[df['method'] == 'hb_ec_hte'].groupby('n_expert')['avg_rmse'].mean()
    gl = df[df['method'] == 'global_corrected'].groupby('n_expert')['avg_rmse'].mean()

    rows = []
    for n in hb.index:
        ratio = hb[n] / gl[n]
        rows.append({
            'exp': 'exp-003',
            'n_per_K': n / K,
            'worst_misclass': worst_misclass,
            'hb_rmse': hb[n],
            'gl_rmse': gl[n],
            'ratio': ratio,
        })
    return pd.DataFrame(rows)


def load_exp013():
    df = pd.read_csv('results/exp013_boundary_condition.csv')
    df['sq_err'] = (df['tau_corrected'] - df['tau_true'])**2
    agg = df.groupby(['method']).agg(mean_se=('sq_err', 'mean')).reset_index()
    agg['rmse'] = np.sqrt(agg['mean_se'])

    hb_rmse = agg.loc[agg['method'] == 'hb_eb', 'rmse'].values[0]
    gl_rmse = agg.loc[agg['method'] == 'global_corrected', 'rmse'].values[0]

    return pd.DataFrame([{
        'exp': 'exp-013',
        'n_per_K': 12.5,
        'worst_misclass': 0.30,
        'hb_rmse': hb_rmse,
        'gl_rmse': gl_rmse,
        'ratio': hb_rmse / gl_rmse,
    }])


def load_exp015():
    df = pd.read_csv('results/exp015_semi_real_raw.csv')
    agg = df.groupby(['n_expert', 'method']).agg(mean_se=('squared_error', 'mean')).reset_index()
    agg['rmse'] = np.sqrt(agg['mean_se'])

    K = 4
    worst_fnr = 0.526
    rows = []
    for n in sorted(df['n_expert'].unique()):
        hb_rmse = agg.loc[(agg['n_expert'] == n) & (agg['method'] == 'hb_ec_hte'), 'rmse'].values[0]
        gl_rmse = agg.loc[(agg['n_expert'] == n) & (agg['method'] == 'global_corrected'), 'rmse'].values[0]
        rows.append({
            'exp': 'exp-015',
            'n_per_K': n / K,
            'worst_misclass': worst_fnr,
            'hb_rmse': hb_rmse,
            'gl_rmse': gl_rmse,
            'ratio': hb_rmse / gl_rmse,
        })
    return pd.DataFrame(rows)


# ── Classify outcome ─────────────────────────────────────────────────────

def classify(ratio):
    if ratio < 0.95:
        return 'green'
    elif ratio <= 1.05:
        return 'yellow'
    else:
        return 'red'


# ── Load all data ────────────────────────────────────────────────────────

data = pd.concat([load_exp001(), load_exp003(), load_exp013(), load_exp015()], ignore_index=True)
data['outcome'] = data['ratio'].apply(classify)

# ── Plot ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5.5))

# Background shading for regions
x_range = np.array([5, 2000])

# Red zone: high FNR (>0.5) — always fails
ax.axhspan(0.45, 0.60, color='#FFCCCC', alpha=0.30, zorder=0)

# Yellow/transitional zone
yellow_x = [25, 1500, 1500, 100]
yellow_y = [0.17, 0.17, 0.45, 0.45]
ax.fill(yellow_x, yellow_y, color='#FFF9C4', alpha=0.20, zorder=0)

# Green zone: moderate misclass + high budget
green_x = [50, 1500, 1500, 200]
green_y = [0.17, 0.17, 0.32, 0.32]
ax.fill(green_x, green_y, color='#C8E6C9', alpha=0.35, zorder=0)

# Marker styles per experiment
marker_map = {
    'exp-001 (K=2)': ('o', 55),
    'exp-001 (K=4)': ('s', 55),
    'exp-003': ('^', 60),
    'exp-013': ('X', 80),
    'exp-015': ('D', 55),
}

color_map = {
    'green': '#2E7D32',
    'yellow': '#F9A825',
    'red': '#C62828',
}

# Plot each experiment's points
for exp_name, (marker, ms) in marker_map.items():
    sub = data[data['exp'] == exp_name]
    for _, row in sub.iterrows():
        c = color_map[row['outcome']]
        edgecolor = 'black' if row['outcome'] != 'red' else '#8B0000'
        ax.scatter(row['n_per_K'], row['worst_misclass'], marker=marker,
                   s=ms, c=c, edgecolors=edgecolor, linewidths=0.8, zorder=5)

# Threshold lines
ax.axvline(x=25, color='#555555', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
ax.axhline(y=0.50, color='#555555', linestyle=':', linewidth=1.0, alpha=0.7, zorder=2)

# Threshold annotations
ax.annotate(r'$n_{\mathrm{expert}}/K = 25$',
            xy=(25, 0.185), fontsize=7.5, color='#555555', ha='right',
            fontstyle='italic', va='top',
            rotation=90)
ax.annotate('worst FNR = 0.5',
            xy=(1200, 0.505), fontsize=7.5, color='#555555', ha='right',
            va='bottom', fontstyle='italic')

# Annotate specific crossover points for exp-001
ax.annotate('K=2 crossover\n' + r'($n/K \approx 50$)',
            xy=(50, 0.25), xytext=(15, 0.21),
            fontsize=7, color='#2E7D32', ha='center',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))

ax.annotate('K=4 crossover\n' + r'($n/K \approx 250$)',
            xy=(250, 0.30), xytext=(600, 0.355),
            fontsize=7, color='#2E7D32', ha='center',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))

# Annotate exp-013 failure
ax.annotate('exp-013: 2.6× worse\nthan naive (n/K=12.5)',
            xy=(12.5, 0.30), xytext=(8, 0.40),
            fontsize=7, color='#C62828', ha='center',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=0.8))

# Annotate exp-015 failures
ax.annotate('exp-015: real GPT-4.1\n(poor classifier quality)',
            xy=(62.5, 0.526), xytext=(200, 0.555),
            fontsize=7, color='#C62828', ha='center',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=0.8))

# Region labels
ax.text(500, 0.22, 'EC-HTE\nbeneficial', fontsize=9, color='#1B5E20',
        ha='center', va='center', fontstyle='italic', fontweight='bold', alpha=0.7)
ax.text(600, 0.545, 'EC-HTE fails', fontsize=9, color='#B71C1C',
        ha='center', va='center', fontstyle='italic', fontweight='bold', alpha=0.6)

# Axes
ax.set_xscale('log')
ax.set_xlabel(r'Expert budget per subgroup ($n_{\mathrm{expert}} / K$)')
ax.set_ylabel('Worst-subgroup misclassification rate (or FNR)')
ax.set_title('Applicability Frontier: When Does EC-HTE Help?', fontweight='bold', pad=12)
ax.set_xlim(8, 1500)
ax.set_ylim(0.17, 0.58)

# Light grid
ax.grid(True, which='major', color='#E0E0E0', linewidth=0.5, zorder=0)
ax.grid(True, which='minor', color='#F0F0F0', linewidth=0.3, zorder=0, alpha=0.5)
ax.set_axisbelow(True)

# Custom x-tick labels
ax.set_xticks([10, 25, 50, 100, 250, 500, 1000])
ax.set_xticklabels(['10', '25', '50', '100', '250', '500', '1000'])

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=7, markeredgecolor='black', label='exp-001 (K=2, synthetic)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markersize=7, markeredgecolor='black', label='exp-001 (K=4, synthetic)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
           markersize=7, markeredgecolor='black', label='exp-003 (TweetEval semi-synth.)'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='black', label='exp-013 (boundary, K=4, n=50)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
           markersize=6, markeredgecolor='black', label='exp-015 (real GPT-4.1 CM)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32',
           markersize=7, markeredgecolor='black', label='HB wins (ratio < 0.95)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F9A825',
           markersize=7, markeredgecolor='black', label='Comparable (0.95–1.05)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
           markersize=7, markeredgecolor='#8B0000', label='HB worse (ratio > 1.05)'),
]

ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9,
          edgecolor='#CCCCCC', ncol=2, columnspacing=1.0, handletextpad=0.5)

fig.tight_layout()

# Save
fig.savefig('artifacts/figures/applicability_frontier.pdf', bbox_inches='tight')
fig.savefig('artifacts/figures/applicability_frontier.png', bbox_inches='tight', dpi=300)
print('Saved to artifacts/figures/applicability_frontier.{pdf,png}')
