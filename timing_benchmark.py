#!/usr/bin/env python3
"""timing_benchmark.py — Wall-clock timing for EC-HTE pipeline steps.

Measures:
  (a) Single DIM estimate (naive + global corrected)
  (b) HB Dirichlet-Multinomial: centered EB (optimization) + Gibbs MCMC
  (c) EC-HTE full pipeline (DIM + HB centered EB + CM inversion)
  (d) Bootstrap SE (B=100 replicates)

Config: K=4, N=5000, n_expert=250, B=100, 5 repetitions.
"""

import os
import platform
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

N = 5000
K = 4
N_EXPERT = 250
B_BOOTSTRAP = 100
N_REPEATS = 5
LABELS = ['A', 'B', 'C', 'D']
MISCLASS = [0.30, 0.20, 0.10, 0.03]
TAU_Z1 = 0.5
TAU_Z0 = 0.1

# ── DGP (from exp_centered_eb_table1.py) ────────────────────────────────────

def generate_dim_data(seed):
    rng = np.random.RandomState(seed)
    T = rng.binomial(1, 0.5, N)
    Z = rng.binomial(1, 0.5, N)
    S = rng.choice(LABELS, N, p=[1.0 / K] * K)
    X = rng.randn(N, 5)
    tau = np.where(Z == 1, TAU_Z1, TAU_Z0)
    Y = 1 + tau * T + 0.5 * X[:, 0] + rng.randn(N)
    mc_rates = np.zeros(N)
    for i, s in enumerate(LABELS):
        mc_rates[S == s] = MISCLASS[i]
    Z_hat = np.where(rng.rand(N) < mc_rates, 1 - Z, Z)
    idx = rng.choice(N, min(N_EXPERT, N), replace=False)
    expert_mask = np.zeros(N, dtype=bool)
    expert_mask[idx] = True
    return dict(T=T, Z=Z, Z_hat=Z_hat, S=S, X=X, Y=Y, tau=tau, expert_mask=expert_mask)


# ── Shared utilities (from exp_centered_eb_table1.py) ───────────────────────

def diff_in_means(Y, T, mask):
    t1, t0 = mask & (T == 1), mask & (T == 0)
    n1, n0 = t1.sum(), t0.sum()
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan
    y1, y0 = Y[t1], Y[t0]
    return y1.mean() - y0.mean(), np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)


def get_counts(z_true, z_hat, subgroups, labels):
    k = len(labels)
    counts = np.zeros((k, 2, 2), dtype=int)
    for i, s in enumerate(labels):
        ms = subgroups == s
        for z in [0, 1]:
            mz = ms & (z_true == z)
            counts[i, z, 0] = (z_hat[mz] == 0).sum()
            counts[i, z, 1] = (z_hat[mz] == 1).sum()
    return counts


def compute_global_cm(counts):
    cg = counts.sum(axis=0)
    C_gl = np.zeros((2, 2))
    for z in [0, 1]:
        n = cg[z, :].sum()
        for zh in [0, 1]:
            C_gl[z, zh] = (cg[z, zh] + 1) / (n + 2) if n > 0 else 0.5
    return C_gl


def build_mixing_matrix(C):
    M = np.zeros((2, 2))
    for zh in [0, 1]:
        cs = C[0, zh] + C[1, zh]
        if cs > 0:
            M[zh, 0] = C[0, zh] / cs
            M[zh, 1] = C[1, zh] / cs
        else:
            M[zh, :] = 0.5
    return M


def invert_mixing(M, tau_obs, se_obs):
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        return tau_obs.copy(), se_obs.copy()
    Mi = np.linalg.inv(M)
    tau_c = Mi @ tau_obs
    se_c = np.sqrt(np.maximum(np.diag(Mi @ np.diag(se_obs ** 2) @ Mi.T), 0))
    return tau_c, se_c


# ── CM estimators ────────────────────────────────────────────────────────────

def _dirmult_neg_ll_centered(alpha, counts_z, theta_g):
    if alpha < 1e-10:
        return 1e10
    a0 = alpha * theta_g[0]
    a1 = alpha * theta_g[1]
    if a0 < 1e-10 or a1 < 1e-10:
        return 1e10
    ll = 0.0
    for c0, c1 in counts_z:
        n = c0 + c1
        if n == 0:
            continue
        ll += gammaln(c0 + a0) + gammaln(c1 + a1) - gammaln(n + alpha)
        ll -= gammaln(a0) + gammaln(a1) - gammaln(alpha)
    return -ll


def estimate_cm_centered_eb(counts, labels, global_cm):
    k = len(labels)
    alpha_opt = np.zeros(2)
    for z in [0, 1]:
        theta_g = global_cm[z]
        cl = [(int(counts[i, z, 0]), int(counts[i, z, 1])) for i in range(k)]
        total = sum(c0 + c1 for c0, c1 in cl)
        if total == 0:
            alpha_opt[z] = 2.0
            continue
        res = minimize_scalar(
            lambda a, tg=theta_g: _dirmult_neg_ll_centered(a, cl, tg),
            bounds=(0.01, 5000), method='bounded')
        alpha_opt[z] = res.x
    C = {}
    for i, s in enumerate(labels):
        C_s = np.zeros((2, 2))
        for z in [0, 1]:
            a = alpha_opt[z]
            n = counts[i, z, :].sum()
            d = n + a
            C_s[z, 0] = (counts[i, z, 0] + a * global_cm[z, 0]) / d if d > 0 else 0.5
            C_s[z, 1] = (counts[i, z, 1] + a * global_cm[z, 1]) / d if d > 0 else 0.5
        C[s] = C_s
    return C, alpha_opt


# ── HB Gibbs MCMC (from exp001_hb_synthetic.py) ─────────────────────────────

def _log_target_alpha(alpha, theta_z, theta_g, hp_a0=3.0, hp_b0=1.0):
    if alpha < 1e-10:
        return -1e10
    a0 = alpha * theta_g[0]
    a1 = alpha * theta_g[1]
    if a0 < 1e-10 or a1 < 1e-10:
        return -1e10
    k = theta_z.shape[0]
    lp = (hp_a0 - 1.0) * np.log(alpha) - hp_b0 * alpha
    lp += k * (gammaln(alpha) - gammaln(a0) - gammaln(a1))
    lp += (a0 - 1.0) * np.sum(np.log(np.clip(theta_z[:, 0], 1e-300, None)))
    lp += (a1 - 1.0) * np.sum(np.log(np.clip(theta_z[:, 1], 1e-300, None)))
    return lp


def estimate_cm_hb_gibbs(counts, labels, global_cm, seed=42,
                         n_iter=4000, n_warmup=2000, n_chains=4,
                         hp_a0=3.0, hp_b0=1.0):
    k = len(labels)
    chain_inits = [np.array([1.0, 1.0]), np.array([5.0, 5.0]),
                   np.array([0.5, 0.5]), np.array([10.0, 10.0])][:n_chains]
    all_theta = []

    for chain in range(n_chains):
        rng = np.random.RandomState(seed * 31 + chain * 7919 + 1)
        alpha = chain_inits[chain].copy()
        theta = np.ones((k, 2, 2)) * 0.5
        mh_scale = np.array([0.3, 0.3])
        accept_ct = np.array([0, 0])
        attempt_ct = np.array([0, 0])

        theta_samples = np.zeros((n_iter, k, 2, 2))

        for it in range(n_warmup + n_iter):
            for z in [0, 1]:
                post_a = counts[:, z, :].astype(float) + alpha[z] * global_cm[z]
                post_a = np.maximum(post_a, 1e-6)
                g = rng.gamma(post_a)
                g_sum = g.sum(axis=1, keepdims=True)
                theta[:, z, :] = g / np.maximum(g_sum, 1e-300)

            for z in [0, 1]:
                x_prop = np.log(alpha[z]) + rng.normal(0, mh_scale[z])
                a_prop = np.exp(x_prop)
                log_r = (_log_target_alpha(a_prop, theta[:, z, :], global_cm[z], hp_a0, hp_b0) -
                         _log_target_alpha(alpha[z], theta[:, z, :], global_cm[z], hp_a0, hp_b0))
                accepted = np.log(rng.rand()) < log_r
                if accepted:
                    alpha[z] = a_prop
                if it < n_warmup:
                    accept_ct[z] += int(accepted)
                    attempt_ct[z] += 1

            if it < n_warmup and it > 0 and it % 100 == 0:
                for z in [0, 1]:
                    rate = accept_ct[z] / max(attempt_ct[z], 1)
                    if rate < 0.2:
                        mh_scale[z] *= 0.7
                    elif rate > 0.5:
                        mh_scale[z] *= 1.3
                    accept_ct[z] = 0
                    attempt_ct[z] = 0

            if it >= n_warmup:
                theta_samples[it - n_warmup] = theta.copy()

        all_theta.append(theta_samples)

    theta_arr = np.stack(all_theta)
    C = {}
    for i, s in enumerate(labels):
        C[s] = theta_arr[:, :, i, :, :].mean(axis=(0, 1))
    return C


# ── Timing functions ─────────────────────────────────────────────────────────

def time_dim_estimate(data):
    """(a) Single DIM estimate: naive + global corrected."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    em = data['expert_mask']
    Z = data['Z']

    counts = get_counts(Z[em], Z_hat[em], S[em], LABELS)
    global_cm = compute_global_cm(counts)
    M_gl = build_mixing_matrix(global_cm)

    for s in LABELS:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if not np.any(np.isnan(tau_obs)):
            invert_mixing(M_gl, tau_obs, se_obs)


def time_hb_centered_eb(data):
    """(b-1) HB centered EB (optimization-based)."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    em = data['expert_mask']
    Z = data['Z']

    counts = get_counts(Z[em], Z_hat[em], S[em], LABELS)
    global_cm = compute_global_cm(counts)
    estimate_cm_centered_eb(counts, LABELS, global_cm)


def time_hb_gibbs(data):
    """(b-2) HB Gibbs MCMC (4 chains, 2000 warmup + 4000 iterations)."""
    em = data['expert_mask']
    Z, Z_hat, S = data['Z'], data['Z_hat'], data['S']

    counts = get_counts(Z[em], Z_hat[em], S[em], LABELS)
    global_cm = compute_global_cm(counts)
    estimate_cm_hb_gibbs(counts, LABELS, global_cm, seed=42)


def time_echte_full(data):
    """(c) EC-HTE full pipeline: data → counts → HB EB → mixing → inversion."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    em = data['expert_mask']
    Z = data['Z']

    counts = get_counts(Z[em], Z_hat[em], S[em], LABELS)
    global_cm = compute_global_cm(counts)

    C_hb, _ = estimate_cm_centered_eb(counts, LABELS, global_cm)
    M_hb = {s: build_mixing_matrix(C_hb[s]) for s in LABELS}

    for s in LABELS:
        sm = S == s
        th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
        th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
        tau_obs = np.array([th0_n, th1_n])
        se_obs = np.array([se0_n, se1_n])
        if not np.any(np.isnan(tau_obs)):
            invert_mixing(M_hb[s], tau_obs, se_obs)


def time_bootstrap_se(data, B=B_BOOTSTRAP):
    """(d) Bootstrap SE: resample expert → re-estimate CM → re-correct."""
    Y, T, Z_hat, S = data['Y'], data['T'], data['Z_hat'], data['S']
    em = data['expert_mask']
    Z = data['Z']

    expert_idx = np.where(em)[0]
    n_exp = len(expert_idx)

    rng = np.random.RandomState(123)

    for b in range(B):
        boot_idx = rng.choice(expert_idx, n_exp, replace=True)
        z_boot = Z[boot_idx]
        zh_boot = Z_hat[boot_idx]
        s_boot = S[boot_idx]

        counts_b = get_counts(z_boot, zh_boot, s_boot, LABELS)
        global_cm_b = compute_global_cm(counts_b)
        C_b, _ = estimate_cm_centered_eb(counts_b, LABELS, global_cm_b)
        M_b = {s: build_mixing_matrix(C_b[s]) for s in LABELS}

        for s in LABELS:
            sm = S == s
            th0_n, se0_n = diff_in_means(Y, T, sm & (Z_hat == 0))
            th1_n, se1_n = diff_in_means(Y, T, sm & (Z_hat == 1))
            tau_obs = np.array([th0_n, th1_n])
            se_obs = np.array([se0_n, se1_n])
            if not np.any(np.isnan(tau_obs)):
                invert_mixing(M_b[s], tau_obs, se_obs)


# ── Machine info ─────────────────────────────────────────────────────────────

def get_machine_info():
    import subprocess
    cpu_model = "unknown"
    try:
        out = subprocess.check_output(
            "grep 'model name' /proc/cpuinfo | head -1", shell=True).decode().strip()
        cpu_model = out.split(":")[1].strip()
    except Exception:
        pass

    try:
        n_cores = int(subprocess.check_output("nproc", shell=True).decode().strip())
    except Exception:
        n_cores = os.cpu_count() or -1

    try:
        out = subprocess.check_output("free -h | head -2", shell=True).decode().strip()
        mem_line = out.split("\n")[1]
        total_mem = mem_line.split()[1]
    except Exception:
        total_mem = "unknown"

    return {
        'cpu_model': cpu_model,
        'n_cores': n_cores,
        'total_memory': total_mem,
        'platform': platform.platform(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_timing(func, data, label, n_repeats=N_REPEATS):
    times = []
    for r in range(n_repeats):
        t0 = time.time()
        func(data)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  {label} rep {r+1}/{n_repeats}: {elapsed:.3f}s")
    return np.array(times)


if __name__ == '__main__':
    machine = get_machine_info()
    print("=" * 70)
    print("EC-HTE Timing Benchmark")
    print("=" * 70)
    print(f"CPU:    {machine['cpu_model']}")
    print(f"Cores:  {machine['n_cores']}")
    print(f"Memory: {machine['total_memory']}")
    print(f"Config: K={K}, N={N}, n_expert={N_EXPERT}, B={B_BOOTSTRAP}")
    print(f"Repeats: {N_REPEATS}")
    print()

    data = generate_dim_data(seed=42)

    print("[1/5] Single DIM estimate (naive + global corrected)")
    t_dim = run_timing(time_dim_estimate, data, "DIM")

    print("\n[2/5] HB Centered EB (optimization)")
    t_eb = run_timing(time_hb_centered_eb, data, "HB-EB")

    print("\n[3/5] HB Gibbs MCMC (4 chains × 6000 iter)")
    t_gibbs = run_timing(time_hb_gibbs, data, "HB-Gibbs")

    print("\n[4/5] EC-HTE full pipeline (DIM + HB-EB + inversion)")
    t_echte = run_timing(time_echte_full, data, "EC-HTE")

    print(f"\n[5/5] Bootstrap SE (B={B_BOOTSTRAP} replicates)")
    t_boot = run_timing(time_bootstrap_se, data, "Bootstrap")

    # ── Results table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TIMING RESULTS")
    print("=" * 70)
    print(f"{'Step':<40s} | {'Time (s)':>12s} | {'Per-rep':>12s}")
    print("-" * 40 + "-+-" + "-" * 12 + "-+-" + "-" * 12)

    rows = []

    def fmt_row(name, times, per_rep=False):
        m, s = times.mean(), times.std()
        pr = ""
        pr_val = np.nan
        if per_rep:
            pr_val = m / B_BOOTSTRAP
            pr = f"{pr_val:.4f}"
        print(f"{name:<40s} | {m:>5.3f} ± {s:.3f} | {pr:>12s}")
        rows.append({
            'step': name,
            'mean_seconds': round(m, 4),
            'std_seconds': round(s, 4),
            'per_replicate_seconds': round(pr_val, 4) if not np.isnan(pr_val) else None,
            'n_repeats': len(times),
        })

    fmt_row("Single DIM estimate", t_dim)
    fmt_row("HB Centered EB (optimization)", t_eb)
    fmt_row("HB Gibbs MCMC (4ch×6k iter)", t_gibbs)
    fmt_row("EC-HTE full pipeline", t_echte)
    fmt_row(f"Bootstrap SE (B={B_BOOTSTRAP})", t_boot, per_rep=True)

    print()
    print(f"CPU: {machine['cpu_model']}")
    print(f"Cores: {machine['n_cores']}, Memory: {machine['total_memory']}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    artifacts_dir = os.path.expanduser(
        '~/.agent-ml-research-idea_gen_0520/projects/ec_hte_llm_annotation/artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, 'timing_benchmark.csv')

    df = pd.DataFrame(rows)
    df['cpu_model'] = machine['cpu_model']
    df['n_cores'] = machine['n_cores']
    df['total_memory'] = machine['total_memory']
    df['config'] = f"K={K}, N={N}, n_expert={N_EXPERT}, B={B_BOOTSTRAP}"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
