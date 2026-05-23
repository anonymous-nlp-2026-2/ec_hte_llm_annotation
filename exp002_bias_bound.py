"""
exp002_bias_bound.py — Formal Bias Bound Verification (exp-002)

Verifies the exact identity under A1-A4:
  bias_global(z=0, s) = +δ_s · Δτ / (1 - 2π̄)
  bias_global(z=1, s) = -δ_s · Δτ / (1 - 2π̄)

where δ_s = π_s - π̄, Δτ = τ(Z=1) - τ(Z=0).

Parts:
  1. Derivation (see results/exp002_derivation.md)
  2. K=2 systematic parameter grid validation
  3. "Debiasing can hurt" analysis
  4. K=4 extension
"""

import numpy as np
import pandas as pd
import time
import os

N = 5000
N_MC = 1000
BASE_MISCLASS = 0.15
TAU_Z0 = 0.1
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Analytic formulas (exact under A1-A4)
# ============================================================

def analytic_bias_global(pi_s, pi_bar, delta_tau):
    """bias[z=0] = +δ_s·Δτ/(1-2π̄), bias[z=1] = -δ_s·Δτ/(1-2π̄)."""
    delta_s = pi_s - pi_bar
    denom = 1.0 - 2.0 * pi_bar
    if abs(denom) < 1e-12:
        return 0.0, 0.0
    b = delta_s * delta_tau / denom
    return b, -b


def analytic_bias_naive(pi_s, delta_tau):
    """bias[z=0] = +π_s·Δτ, bias[z=1] = -π_s·Δτ."""
    return pi_s * delta_tau, -pi_s * delta_tau


def bias_ratio(pi_s, pi_bar):
    """Ratio |bias_global|/|bias_naive| (independent of Δτ)."""
    denom = (1.0 - 2.0 * pi_bar) * pi_s
    if abs(denom) < 1e-12:
        return np.nan
    return abs(pi_s - pi_bar) / denom


# ============================================================
# MC simulation — vectorized K=2
# ============================================================

def mc_simulate_k2(pi_A, pi_B, tau_z0, tau_z1, n, n_mc, seed=0):
    """Run n_mc Monte Carlo simulations with known global confusion matrix."""
    pi_bar = 0.5 * (pi_A + pi_B)
    denom = 1.0 - 2.0 * pi_bar
    M_inv = np.array([[1 - pi_bar, -pi_bar],
                       [-pi_bar, 1 - pi_bar]]) / denom
    tau_true = np.array([tau_z0, tau_z1])

    rng = np.random.RandomState(seed)
    T = rng.binomial(1, 0.5, (n_mc, n))
    Z = rng.binomial(1, 0.5, (n_mc, n))
    S = rng.binomial(1, 0.5, (n_mc, n))  # 0=A, 1=B

    tau = np.where(Z == 1, tau_z1, tau_z0)
    Y = 1.0 + tau * T + 0.5 * rng.randn(n_mc, n) + rng.randn(n_mc, n)

    mc_rates = np.where(S == 0, pi_A, pi_B)
    Z_hat = np.where(rng.rand(n_mc, n) < mc_rates, 1 - Z, Z)

    # Group index: S*4 + Z_hat*2 + T → 8 groups
    group = (S * 4 + Z_hat * 2 + T).ravel()
    mc_idx = np.repeat(np.arange(n_mc), n)
    combined = mc_idx * 8 + group

    sums = np.bincount(combined, weights=Y.ravel(), minlength=n_mc * 8).reshape(n_mc, 8)
    counts = np.bincount(combined, minlength=n_mc * 8).reshape(n_mc, 8)
    means = sums / np.maximum(counts, 1)

    # Diff-in-means: group = S*4 + Zh*2 + T
    # A(S=0): (zh=0,T=0)→0, (zh=0,T=1)→1, (zh=1,T=0)→2, (zh=1,T=1)→3
    # B(S=1): (zh=0,T=0)→4, (zh=0,T=1)→5, (zh=1,T=0)→6, (zh=1,T=1)→7
    tau_obs_A = np.column_stack([means[:, 1] - means[:, 0],
                                  means[:, 3] - means[:, 2]])
    tau_obs_B = np.column_stack([means[:, 5] - means[:, 4],
                                  means[:, 7] - means[:, 6]])

    # M_inv is symmetric, so tau_obs @ M_inv == (M_inv @ tau_obs.T).T
    return {
        'A': {'global': tau_obs_A @ M_inv - tau_true,
               'naive':  tau_obs_A - tau_true},
        'B': {'global': tau_obs_B @ M_inv - tau_true,
               'naive':  tau_obs_B - tau_true},
    }


# ============================================================
# MC simulation — vectorized K=4
# ============================================================

def mc_simulate_k4(pi_values, tau_z0, tau_z1, n, n_mc, seed=0):
    """Run n_mc Monte Carlo simulations for K=4 subgroups with known global CM."""
    K = len(pi_values)
    pi_bar = np.mean(pi_values)
    denom = 1.0 - 2.0 * pi_bar
    M_inv = np.array([[1 - pi_bar, -pi_bar],
                       [-pi_bar, 1 - pi_bar]]) / denom
    tau_true = np.array([tau_z0, tau_z1])

    rng = np.random.RandomState(seed)
    T = rng.binomial(1, 0.5, (n_mc, n))
    Z = rng.binomial(1, 0.5, (n_mc, n))
    S = rng.choice(K, (n_mc, n))

    tau = np.where(Z == 1, tau_z1, tau_z0)
    Y = 1.0 + tau * T + 0.5 * rng.randn(n_mc, n) + rng.randn(n_mc, n)

    pi_arr = np.array(pi_values)
    mc_rates = pi_arr[S]
    Z_hat = np.where(rng.rand(n_mc, n) < mc_rates, 1 - Z, Z)

    n_groups = K * 4
    group = (S * 4 + Z_hat * 2 + T).ravel()
    mc_idx = np.repeat(np.arange(n_mc), n)
    combined = mc_idx * n_groups + group

    sums = np.bincount(combined, weights=Y.ravel(),
                        minlength=n_mc * n_groups).reshape(n_mc, n_groups)
    counts = np.bincount(combined, minlength=n_mc * n_groups).reshape(n_mc, n_groups)
    means = sums / np.maximum(counts, 1)

    results = {}
    for s in range(K):
        base = s * 4
        tau_obs = np.column_stack([means[:, base + 1] - means[:, base + 0],
                                    means[:, base + 3] - means[:, base + 2]])
        results[s] = {
            'global': tau_obs @ M_inv - tau_true,
            'naive':  tau_obs - tau_true,
        }
    return results


# ============================================================
# Part 2: K=2 parameter grid
# ============================================================

def run_k2_grid(dm_vals, dc_vals, n_mc, label=""):
    """Run K=2 grid and return DataFrame of results."""
    rows = []
    t0 = time.time()
    total = len(dm_vals) * len(dc_vals)
    count = 0

    for i, dm in enumerate(dm_vals):
        pi_A = BASE_MISCLASS + dm / 2.0
        pi_B = BASE_MISCLASS - dm / 2.0
        if pi_B < 0.001:
            continue
        pi_bar = 0.5 * (pi_A + pi_B)  # = BASE_MISCLASS

        for j, dc in enumerate(dc_vals):
            tau_z1 = TAU_Z0 + dc
            seed = i * 10000 + j
            mc = mc_simulate_k2(pi_A, pi_B, TAU_Z0, tau_z1, N, n_mc, seed=seed)

            for s_val, pi_s in [('A', pi_A), ('B', pi_B)]:
                ab0, ab1 = analytic_bias_global(pi_s, pi_bar, dc)
                mc_g = mc[s_val]['global']

                for z, ab in [(0, ab0), (1, ab1)]:
                    mc_vals = mc_g[:, z]
                    mc_mean = np.mean(mc_vals)
                    mc_se = np.std(mc_vals, ddof=1) / np.sqrt(n_mc)
                    tr = abs(mc_mean) / abs(ab) if abs(ab) > 1e-8 else np.nan

                    rows.append({
                        'delta_misclass': round(dm, 4),
                        'delta_cate': round(dc, 4),
                        'subgroup': s_val,
                        'z_value': z,
                        'analytic_bias': ab,
                        'mc_bias': mc_mean,
                        'mc_bias_se': mc_se,
                        'tightness_ratio': tr,
                    })

            count += 1
            if label and count % max(1, total // 10) == 0:
                elapsed = time.time() - t0
                eta = elapsed / count * (total - count)
                print(f"  {label} [{count}/{total}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    elapsed = time.time() - t0
    if label:
        print(f"  {label} done in {elapsed:.1f}s ({total} grid points, {n_mc} MC each)")
    return pd.DataFrame(rows)


def report_k2_results(df):
    """Print K=2 grid summary statistics."""
    valid = df[df['analytic_bias'].abs() > 1e-8].copy()
    if len(valid) == 0:
        print("  No valid (non-zero bias) points.")
        return

    ss_res = ((valid['mc_bias'] - valid['analytic_bias']) ** 2).sum()
    ss_tot = ((valid['mc_bias'] - valid['mc_bias'].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tr = valid['tightness_ratio'].dropna()
    print(f"  R² (analytic vs MC): {r2:.6f}")
    print(f"  Tightness ratio: mean={tr.mean():.4f} median={tr.median():.4f} "
          f"std={tr.std():.4f} [{tr.min():.4f}, {tr.max():.4f}]")

    # By regime
    regimes = {
        'extreme':     (0.20, 0.40),
        'moderate':    (0.05, 0.15),
        'homogeneous': (0.00, 0.05),
    }
    print("\n  Tightness by regime:")
    for name, (dm_t, dc_t) in regimes.items():
        sub = df[(abs(df['delta_misclass'] - dm_t) < 0.005) &
                  (abs(df['delta_cate'] - dc_t) < 0.013)]
        if len(sub) == 0:
            print(f"    {name}: no matching grid point")
            continue
        ab_max = sub['analytic_bias'].abs().max()
        mc_sub = sub['mc_bias']
        mc_se_sub = sub['mc_bias_se']
        if ab_max < 1e-8:
            # Homogeneous: report MC bias magnitude vs 0
            print(f"    {name}: analytic_bias=0, MC bias mean={mc_sub.abs().mean():.6f}, "
                  f"max SE={mc_se_sub.max():.6f} (consistent with 0)")
        else:
            tr_sub = sub['tightness_ratio'].dropna()
            if len(tr_sub) > 0:
                print(f"    {name}: tightness mean={tr_sub.mean():.4f} "
                      f"[{tr_sub.min():.4f}, {tr_sub.max():.4f}] (n={len(tr_sub)})")

    # Approximation quality note
    print("\n  Note: A2 (symmetric misclassification) is exact in our DGP.")
    print("  In practice, FPR ≠ FNR introduces O(|FPR-FNR|) additional bias.")
    print("  The tightness ratio deviates from 1.0 only due to MC sampling noise.")


# ============================================================
# Part 3: "Debiasing can hurt"
# ============================================================

def run_debiasing_hurts():
    """Analytic analysis of when global correction increases bias."""
    dm_vals = np.arange(0, 0.26, 0.01)
    dc_vals = np.arange(0, 0.525, 0.025)

    heatmap_rows = []
    for dm in dm_vals:
        pi_A = BASE_MISCLASS + dm / 2.0
        pi_B = BASE_MISCLASS - dm / 2.0
        if pi_B < 0.001:
            continue
        pi_bar = BASE_MISCLASS

        ratio_A = bias_ratio(pi_A, pi_bar)
        ratio_B = bias_ratio(pi_B, pi_bar)

        for dc in dc_vals:
            hurts_A = (ratio_A > 1.0) and (dc > 0) if not np.isnan(ratio_A) else False
            hurts_B = (ratio_B > 1.0) and (dc > 0) if not np.isnan(ratio_B) else False

            heatmap_rows.append({
                'delta_misclass': round(dm, 4),
                'delta_cate': round(dc, 4),
                'debiasing_hurts_A': hurts_A,
                'debiasing_hurts_B': hurts_B,
                'max_bias_ratio_A': ratio_A,
                'max_bias_ratio_B': ratio_B,
            })

    df_heat = pd.DataFrame(heatmap_rows)
    df_heat.to_csv(f'{RESULTS_DIR}/exp002_bias_heatmap.csv', index=False)

    # Statistics
    active = df_heat[df_heat['delta_cate'] > 0]
    n_total = len(active)
    n_hurts_A = active['debiasing_hurts_A'].sum()
    n_hurts_B = active['debiasing_hurts_B'].sum()
    n_hurts_any = (active['debiasing_hurts_A'] | active['debiasing_hurts_B']).sum()

    print(f"  Grid points with Δτ > 0: {n_total}")
    print(f"  'Debiasing hurts' subgroup A: {n_hurts_A}/{n_total} = {n_hurts_A/max(n_total,1)*100:.1f}%")
    print(f"  'Debiasing hurts' subgroup B: {n_hurts_B}/{n_total} = {n_hurts_B/max(n_total,1)*100:.1f}%")
    print(f"  'Debiasing hurts' any:        {n_hurts_any}/{n_total} = {n_hurts_any/max(n_total,1)*100:.1f}%")

    # Threshold
    # Condition for B: (π̄ - π_B)/((1-2π̄)·π_B) > 1
    # With π̄ = BASE, π_B = BASE - Δ/2:
    # Δ/(2·(1-2·BASE)) > BASE - Δ/2
    # Δ·(1/(2·(1-2·BASE)) + 1/2) > BASE
    # Δ > BASE / (1/(2·(1-2·BASE)) + 1/2)
    coeff = 1.0 / (2.0 * (1.0 - 2.0 * BASE_MISCLASS)) + 0.5
    threshold = BASE_MISCLASS / coeff
    print(f"\n  Analytic threshold: debiasing hurts B when Δ_misclass > {threshold:.4f}")
    print(f"  (Independent of Δτ as long as Δτ > 0)")

    # Constructive example
    print(f"\n  Constructive example (K=2):")
    pi_A, pi_B = 0.25, 0.05
    tau_z0_ex, tau_z1_ex = 0.1, 0.5
    pi_bar_ex = 0.5 * (pi_A + pi_B)
    delta_tau_ex = tau_z1_ex - tau_z0_ex

    example_rows = []
    print(f"    π_A={pi_A}, π_B={pi_B}, π̄={pi_bar_ex}, τ(0)={tau_z0_ex}, τ(1)={tau_z1_ex}")
    for s_val, pi_s in [('A', pi_A), ('B', pi_B)]:
        bg0, bg1 = analytic_bias_global(pi_s, pi_bar_ex, delta_tau_ex)
        bn0, bn1 = analytic_bias_naive(pi_s, delta_tau_ex)
        ratio = abs(bg0) / abs(bn0) if abs(bn0) > 1e-12 else np.nan
        hurts = "|bias_global| > |bias_naive|" if ratio > 1.0 else "|bias_global| < |bias_naive|"
        print(f"    Subgroup {s_val}: bias_naive(z=0)={bn0:+.4f}, bias_global(z=0)={bg0:+.4f}, "
              f"ratio={ratio:.3f} → {hurts}")

        for z, bg, bn in [(0, bg0, bn0), (1, bg1, bn1)]:
            r = abs(bg) / abs(bn) if abs(bn) > 1e-12 else np.nan
            example_rows.append({
                'k_subgroups': 2, 'example_name': 'extreme_k2',
                'pi_values': f'[{pi_A},{pi_B}]',
                'tau_z0': tau_z0_ex, 'tau_z1': tau_z1_ex,
                'subgroup': s_val, 'z_value': z,
                'bias_naive': bn, 'bias_global': bg, 'ratio': r,
            })

    df_ex = pd.DataFrame(example_rows)
    df_ex.to_csv(f'{RESULTS_DIR}/exp002_debiasing_hurts.csv', index=False)

    return df_heat, df_ex


# ============================================================
# Part 4: K=4 extension
# ============================================================

def run_k4_validation():
    """Validate the bias formula for K=4 subgroups."""
    PI_BASE = [0.25, 0.15, 0.10, 0.05]
    pi_bar_base = np.mean(PI_BASE)
    delta_base = np.array(PI_BASE) - pi_bar_base

    # Spread factors: scale deviations from mean
    spread_factors = [0.0, 0.25, 0.50, 0.75, 1.0]
    dc_vals = np.arange(0, 0.525, 0.05)

    rows = []
    k4_example_rows = []
    t0 = time.time()

    for fi, f in enumerate(spread_factors):
        pi_values = list(pi_bar_base + f * delta_base)
        pi_bar = np.mean(pi_values)
        dm_label = round(f * (max(PI_BASE) - min(PI_BASE)), 4)
        print(f"  f={f:.2f}: π={[round(p,4) for p in pi_values]}, "
              f"π̄={pi_bar:.4f}, spread={dm_label}")

        for ji, dc in enumerate(dc_vals):
            tau_z1 = TAU_Z0 + dc
            seed = fi * 100000 + ji * 100
            mc = mc_simulate_k4(pi_values, TAU_Z0, tau_z1, N, N_MC, seed=seed)

            for s in range(4):
                pi_s = pi_values[s]
                ab0, ab1 = analytic_bias_global(pi_s, pi_bar, dc)
                mc_g = mc[s]['global']

                for z, ab in [(0, ab0), (1, ab1)]:
                    mc_mean = np.mean(mc_g[:, z])
                    mc_se = np.std(mc_g[:, z], ddof=1) / np.sqrt(N_MC)
                    tr = abs(mc_mean) / abs(ab) if abs(ab) > 1e-8 else np.nan

                    rows.append({
                        'delta_misclass': dm_label,
                        'delta_cate': round(dc, 4),
                        'subgroup': s,
                        'pi_s': round(pi_s, 4),
                        'analytic_bias': ab,
                        'mc_bias': mc_mean,
                        'mc_bias_se': mc_se,
                        'tightness_ratio': tr,
                    })

                # Debiasing hurts examples at dc=0.40, full spread
                if f == 1.0 and abs(dc - 0.40) < 0.001:
                    bn0, bn1 = analytic_bias_naive(pi_s, dc)
                    for z_val, bg, bn in [(0, ab0, bn0), (1, ab1, bn1)]:
                        r = abs(bg) / abs(bn) if abs(bn) > 1e-12 else np.nan
                        k4_example_rows.append({
                            'k_subgroups': 4, 'example_name': 'base_k4',
                            'pi_values': str([round(p, 4) for p in pi_values]),
                            'tau_z0': TAU_Z0, 'tau_z1': TAU_Z0 + dc,
                            'subgroup': s, 'z_value': z_val,
                            'bias_naive': bn, 'bias_global': bg, 'ratio': r,
                        })

    elapsed = time.time() - t0
    print(f"  K=4 grid done in {elapsed:.1f}s")

    df_k4 = pd.DataFrame(rows)
    df_k4.to_csv(f'{RESULTS_DIR}/exp002_k4_validation.csv', index=False)

    # R² and tightness
    valid = df_k4[df_k4['analytic_bias'].abs() > 1e-8].copy()
    if len(valid) > 0:
        ss_res = ((valid['mc_bias'] - valid['analytic_bias']) ** 2).sum()
        ss_tot = ((valid['mc_bias'] - valid['mc_bias'].mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        tr = valid['tightness_ratio'].dropna()
        print(f"\n  R² (K=4): {r2:.6f}")
        print(f"  Tightness: mean={tr.mean():.4f} median={tr.median():.4f} "
              f"std={tr.std():.4f} [{tr.min():.4f}, {tr.max():.4f}]")

    # Debiasing hurts in K=4 at full spread
    pi_bar_full = np.mean(PI_BASE)
    denom_full = 1.0 - 2.0 * pi_bar_full
    threshold_k4 = pi_bar_full / (2.0 * (1.0 - pi_bar_full))
    print(f"\n  K=4 'debiasing hurts' condition: π_s < π̄/(2(1-π̄)) = {threshold_k4:.4f}")
    print(f"  K=4 base config: π = {PI_BASE}")
    for s, pi_s in enumerate(PI_BASE):
        r = bias_ratio(pi_s, pi_bar_full)
        hurts = "YES" if r > 1.0 else "no"
        print(f"    Subgroup {s} (π={pi_s:.2f}): ratio={r:.3f} → {hurts}")

    # Compare K=2 vs K=4 debiasing hurts fractions
    # K=2 at matched spread (Δ=0.20): hurts for B only (1/2 = 50%)
    # K=4 at full spread: hurts for subgroup 3 only (1/4 = 25%)
    print(f"\n  K=2 (Δ=0.20): 1/2 subgroups affected = 50%")
    n_hurts_k4 = sum(1 for pi_s in PI_BASE
                      if bias_ratio(pi_s, pi_bar_full) > 1.0)
    print(f"  K=4 (spread=0.20): {n_hurts_k4}/4 subgroups affected = {n_hurts_k4/4*100:.0f}%")
    print(f"  Remark: formula applies to binary Z ∈ {{0,1}} with arbitrary K subgroups.")

    # Append K=4 examples to debiasing_hurts.csv
    if k4_example_rows:
        df_k4_ex = pd.DataFrame(k4_example_rows)
        existing_path = f'{RESULTS_DIR}/exp002_debiasing_hurts.csv'
        if os.path.exists(existing_path):
            df_old = pd.read_csv(existing_path)
            pd.concat([df_old, df_k4_ex], ignore_index=True).to_csv(
                existing_path, index=False)
        else:
            df_k4_ex.to_csv(existing_path, index=False)

    return df_k4


# ============================================================
# Quick test (5x5 grid, 100 MC)
# ============================================================

def quick_test():
    """Validate code correctness on a small grid."""
    dm_vals = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    dc_vals = np.array([0.0, 0.10, 0.20, 0.30, 0.40])

    df = run_k2_grid(dm_vals, dc_vals, n_mc=100)
    valid = df[df['analytic_bias'].abs() > 1e-8]

    max_abs_err = (valid['mc_bias'] - valid['analytic_bias']).abs().max()
    mean_abs_err = (valid['mc_bias'] - valid['analytic_bias']).abs().mean()

    print(f"  Quick test: {len(valid)} non-zero-bias points")
    print(f"  Max |MC - analytic|  = {max_abs_err:.6f}")
    print(f"  Mean |MC - analytic| = {mean_abs_err:.6f}")

    # With 100 MC and N=5000, SE of MC mean ≈ 0.003.
    # Max error should be < ~0.02 (6 SE) for all points.
    if max_abs_err > 0.05:
        print(f"  WARNING: max error {max_abs_err:.4f} exceeds threshold 0.05")
    else:
        print(f"  Quick test passed (max error < 0.05).")
    return df


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 80)
    print("exp002_bias_bound: Formal Bias Bound Verification")
    print(f"N={N}, N_MC={N_MC}, base_misclass={BASE_MISCLASS}, τ(Z=0)={TAU_Z0}")
    print("=" * 80)

    # Step 1: Quick test
    print("\n--- Quick test (5x5, 100 MC) ---")
    quick_test()

    # Step 2: Full K=2 grid
    print("\n--- Part 2: K=2 full grid ---")
    dm_full = np.arange(0, 0.26, 0.01)
    dc_full = np.arange(0, 0.525, 0.025)
    df_k2 = run_k2_grid(dm_full, dc_full, n_mc=N_MC, label="K=2")
    df_k2.to_csv(f'{RESULTS_DIR}/exp002_bias_bound.csv', index=False)
    print()
    report_k2_results(df_k2)

    # Step 3: Debiasing hurts
    print("\n--- Part 3: 'Debiasing can hurt' ---")
    df_heat, df_ex = run_debiasing_hurts()

    # Step 4: K=4 extension
    print("\n--- Part 4: K=4 extension ---")
    df_k4 = run_k4_validation()

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_k2 = df_k2[df_k2['analytic_bias'].abs() > 1e-8]
    ss_res = ((valid_k2['mc_bias'] - valid_k2['analytic_bias']) ** 2).sum()
    ss_tot = ((valid_k2['mc_bias'] - valid_k2['mc_bias'].mean()) ** 2).sum()
    r2_k2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    tr_k2 = valid_k2['tightness_ratio'].dropna()

    print(f"  K=2 R²: {r2_k2:.6f}")
    print(f"  K=2 tightness: {tr_k2.mean():.4f} ± {tr_k2.std():.4f}")

    valid_k4 = df_k4[df_k4['analytic_bias'].abs() > 1e-8]
    ss_res4 = ((valid_k4['mc_bias'] - valid_k4['analytic_bias']) ** 2).sum()
    ss_tot4 = ((valid_k4['mc_bias'] - valid_k4['mc_bias'].mean()) ** 2).sum()
    r2_k4 = 1.0 - ss_res4 / ss_tot4 if ss_tot4 > 0 else np.nan
    tr_k4 = valid_k4['tightness_ratio'].dropna()

    print(f"  K=4 R²: {r2_k4:.6f}")
    print(f"  K=4 tightness: {tr_k4.mean():.4f} ± {tr_k4.std():.4f}")

    active = df_heat[df_heat['delta_cate'] > 0]
    n_hurts = (active['debiasing_hurts_A'] | active['debiasing_hurts_B']).sum()
    print(f"  'Debiasing hurts' fraction: {n_hurts}/{len(active)} = {n_hurts/max(len(active),1)*100:.1f}%")

    print(f"\nOutput files:")
    for f in ['exp002_bias_bound.csv', 'exp002_bias_heatmap.csv',
              'exp002_debiasing_hurts.csv', 'exp002_k4_validation.csv',
              'exp002_derivation.md']:
        path = f'{RESULTS_DIR}/{f}'
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {path} ({size:,} bytes)")
