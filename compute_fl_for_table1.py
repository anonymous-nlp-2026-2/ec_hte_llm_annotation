#!/usr/bin/env python3
"""Compute F-L scalar and per-subgroup metrics for Table 1 DIM_K2_extreme."""
import sys
sys.path.insert(0, '/home/ubuntu/ec_hte_llm_annotation')

import numpy as np
import pandas as pd
from exp_centered_eb_table1 import (
    generate_dim_data, diff_in_means, DIM_CONFIGS, N_MC
)

config_name = 'DIM_K2_extreme'
cfg = DIM_CONFIGS[config_name]
labels = cfg['labels']
tau_z0, tau_z1 = cfg['tau_z0'], cfg['tau_z1']

rows = []
for mc in range(N_MC):
    data = generate_dim_data(cfg, seed=mc)
    Y, T, Z, Z_hat, S = data['Y'], data['T'], data['Z'], data['Z_hat'], data['S']
    em = data['expert_mask']

    # Estimate misclassification rates from expert sample
    pi_hat_global = (Z[em] != Z_hat[em]).mean()
    pi_hat_s = {}
    for s in labels:
        ms = S[em] == s
        if ms.sum() > 0:
            pi_hat_s[s] = (Z[em][ms] != Z_hat[em][ms]).mean()
        else:
            pi_hat_s[s] = pi_hat_global

    c_scalar = 1.0 - 2.0 * pi_hat_global

    for s in labels:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        if np.isnan(th0_n) or np.isnan(th1_n):
            continue

        # F-L scalar
        if abs(c_scalar) > 1e-6:
            th0_fls = th0_n / c_scalar
            th1_fls = th1_n / c_scalar
            se0_fls = se0_n / abs(c_scalar)
            se1_fls = se1_n / abs(c_scalar)
        else:
            th0_fls, th1_fls = th0_n, th1_n
            se0_fls, se1_fls = se0_n, se1_n

        # F-L per-subgroup
        c_sub = 1.0 - 2.0 * pi_hat_s[s]
        if abs(c_sub) > 1e-6:
            th0_flp = th0_n / c_sub
            th1_flp = th1_n / c_sub
            se0_flp = se0_n / abs(c_sub)
            se1_flp = se1_n / abs(c_sub)
        else:
            th0_flp, th1_flp = th0_n, th1_n
            se0_flp, se1_flp = se0_n, se1_n

        base = {'mc_run': mc, 'subgroup': s}
        rows.append({**base, 'method': 'fl_scalar',
                     'tau_hat_z0': th0_fls, 'tau_hat_z1': th1_fls,
                     'se_z0': se0_fls, 'se_z1': se1_fls,
                     'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1})
        rows.append({**base, 'method': 'fl_per_sub',
                     'tau_hat_z0': th0_flp, 'tau_hat_z1': th1_flp,
                     'se_z0': se0_flp, 'se_z1': se1_flp,
                     'tau_true_z0': tau_z0, 'tau_true_z1': tau_z1})

df = pd.DataFrame(rows)

# Aggregate: same logic as aggregate_dim
agg_rows = []
for (sub, method), grp in df.groupby(['subgroup', 'method']):
    row = {'subgroup': sub, 'method': method, 'n_mc': len(grp)}
    for z in ['z0', 'z1']:
        err = grp[f'tau_hat_{z}'].values - grp[f'tau_true_{z}'].values
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        ci_lo = grp[f'tau_hat_{z}'].values - 1.96 * grp[f'se_{z}'].values
        ci_hi = grp[f'tau_hat_{z}'].values + 1.96 * grp[f'se_{z}'].values
        cov = ((ci_lo <= grp[f'tau_true_{z}'].values) &
               (grp[f'tau_true_{z}'].values <= ci_hi)).mean()
        row[f'abs_bias_{z}'] = abs(bias)
        row[f'rmse_{z}'] = rmse
        row[f'coverage_{z}'] = cov
    agg_rows.append(row)

agg = pd.DataFrame(agg_rows)
print("Per-subgroup results:")
print(agg.to_string(index=False))

# Table 1 format: average over subgroups and z-levels
print("\n\nTable 1 format (avg over subgroups & z-levels):")
print(f"{'Method':<20s} {'|Bias|':>8s} {'RMSE':>8s} {'Cov':>8s}")
for m in ['fl_scalar', 'fl_per_sub']:
    ms = agg[agg['method'] == m]
    ab = ms[['abs_bias_z0', 'abs_bias_z1']].values.mean()
    rm = ms[['rmse_z0', 'rmse_z1']].values.mean()
    cv = ms[['coverage_z0', 'coverage_z1']].values.mean()
    print(f"{m:<20s} {ab:8.3f} {rm:8.3f} {cv:8.2f}")
