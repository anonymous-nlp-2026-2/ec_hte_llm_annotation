"""Generate Fig 1: Bias mechanism schematic showing how global correction
over-corrects low-error subgroups and under-corrects high-error subgroups."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

C_ORACLE = '#888888'
C_NAIVE  = '#0072B2'
C_GLOBAL = '#D55E00'
C_ECHTE  = '#009E73'

tau_true = {'z0': 0.10, 'z1': 0.50}
delta_tau = tau_true['z1'] - tau_true['z0']

pi_A = 0.05
pi_B = 0.25
pi_bar = 0.15

# Naive CATE estimates (attenuated toward pooled mean)
# tau_obs(z, s) = (1-pi_s)*tau(z) + pi_s*tau(1-z)
def naive_cate(z, pi):
    if z == 0:
        return (1 - pi) * tau_true['z0'] + pi * tau_true['z1']
    else:
        return (1 - pi) * tau_true['z1'] + pi * tau_true['z0']

# Global correction: bias = delta_s * delta_tau / (1 - 2*pi_bar)
def global_corrected(z, pi_s):
    delta_s = pi_s - pi_bar
    bias = delta_s * delta_tau / (1 - 2 * pi_bar)
    if z == 0:
        return tau_true['z0'] + bias
    else:
        return tau_true['z1'] - bias

# EC-HTE (subgroup-stratified): recovers true CATE exactly
def echte_corrected(z):
    return tau_true[f'z{z}']

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

subgroups = [
    {'name': 'Subgroup A (low error)', 'pi': pi_A, 'ax_idx': 0},
    {'name': 'Subgroup B (high error)', 'pi': pi_B, 'ax_idx': 1},
]

methods = ['Oracle', 'Naive', 'Global', 'EC-HTE']
colors = [C_ORACLE, C_NAIVE, C_GLOBAL, C_ECHTE]
markers = ['_', 'o', 's', 'D']
x_positions = [0, 1]
z_labels = [r'$\tau(z{=}0)$', r'$\tau(z{=}1)$']

for sg in subgroups:
    ax = axes[sg['ax_idx']]
    pi = sg['pi']

    values = {
        'Oracle': [tau_true['z0'], tau_true['z1']],
        'Naive': [naive_cate(0, pi), naive_cate(1, pi)],
        'Global': [global_corrected(0, pi), global_corrected(1, pi)],
        'EC-HTE': [echte_corrected(0), echte_corrected(1)],
    }

    offsets = [-0.18, -0.06, 0.06, 0.18]

    for i, (method, color, marker) in enumerate(zip(methods, colors, markers)):
        vals = values[method]
        xs = [x + offsets[i] for x in x_positions]
        if method == 'Oracle':
            ax.scatter(xs, vals, color=color, marker='_', s=200, linewidths=2.5,
                      zorder=5, label=method)
        else:
            ax.scatter(xs, vals, color=color, marker=marker, s=50, zorder=5,
                      label=method, edgecolors='white', linewidths=0.5)

    # Highlight "debiasing hurts" zone for global correction
    if pi < pi_bar:
        for z_idx in range(2):
            global_val = values['Global'][z_idx]
            true_val = values['Oracle'][z_idx]
            naive_val = values['Naive'][z_idx]
            if abs(global_val - true_val) > abs(naive_val - true_val):
                ax.annotate('', xy=(x_positions[z_idx] + offsets[2], global_val),
                           xytext=(x_positions[z_idx] + offsets[2], true_val),
                           arrowprops=dict(arrowstyle='<->', color=C_GLOBAL,
                                         lw=1.2, ls='--'))
                mid_y = (global_val + true_val) / 2
                ax.text(x_positions[z_idx] + offsets[2] + 0.12, mid_y,
                       'over-\ncorrection', fontsize=7.5, color=C_GLOBAL,
                       ha='left', va='center', style='italic')

    ax.set_title(sg['name'] + f'\n($\\pi_s = {pi:.2f}$)', fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(z_labels)
    ax.set_xlim(-0.4, 1.4)
    ax.axhline(y=tau_true['z0'], color=C_ORACLE, lw=0.6, ls=':', alpha=0.4)
    ax.axhline(y=tau_true['z1'], color=C_ORACLE, lw=0.6, ls=':', alpha=0.4)

axes[0].set_ylabel('Estimated CATE')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
          bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle(f'Global correction uses pooled $\\bar{{\\pi}} = {pi_bar:.2f}$',
            fontsize=11, y=1.02, style='italic', color='#555555')

plt.tight_layout()

import os
out_dir = 'artifacts/figures'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig1_bias_mechanism.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig1_bias_mechanism.png'), bbox_inches='tight')

out_dir2 = 'artifacts/figures'
os.makedirs(out_dir2, exist_ok=True)
fig.savefig(os.path.join(out_dir2, 'fig1_bias_mechanism.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(out_dir2, 'fig1_bias_mechanism.png'), bbox_inches='tight')

print('Fig 1 saved to both paper/figures/ and artifacts/figures/')
