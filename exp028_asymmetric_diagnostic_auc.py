"""
exp-028: Asymmetric Diagnostic AUC

Compares symmetric vs asymmetric diagnostic scores for predicting
when global MC correction hurts subgroup-level CATE estimation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
import time

warnings.filterwarnings("ignore")

# ============================================================
# Data (from exp025_positive_scenario.py COMBOS)
# ============================================================

TWEET_P_Z1 = 0.652
CIVIL_P_Z1 = 0.3

COMBOS = [
    {
        "llm": "Llama-3.1-8B", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.386, "fpr": 0.235, "fnr": 0.319},
            {"name": "False_short", "weight": 0.218, "fpr": 0.310, "fnr": 0.212},
            {"name": "True_medium", "weight": 0.258, "fpr": 0.219, "fnr": 0.258},
            {"name": "True_short", "weight": 0.138, "fpr": 0.091, "fnr": 0.190},
        ],
    },
    {
        "llm": "Qwen2.5-7B", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.1800, "fnr": 0.2874},
            {"name": "False_short", "weight": 0.222, "fpr": 0.2414, "fnr": 0.1750},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.2333, "fnr": 0.2105},
            {"name": "True_short", "weight": 0.141, "fpr": 0.0000, "fnr": 0.1379},
        ],
    },
    {
        "llm": "Mistral-7B", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.180, "fnr": 0.299},
            {"name": "False_short", "weight": 0.222, "fpr": 0.241, "fnr": 0.188},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.233, "fnr": 0.168},
            {"name": "True_short", "weight": 0.141, "fpr": 0.091, "fnr": 0.121},
        ],
    },
    {
        "llm": "Gemma-2-9B", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.110, "fnr": 0.437},
            {"name": "False_short", "weight": 0.222, "fpr": 0.138, "fnr": 0.300},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.033, "fnr": 0.263},
            {"name": "True_short", "weight": 0.141, "fpr": 0.091, "fnr": 0.207},
        ],
    },
    {
        "llm": "gpt-4.1", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.0536, "fnr": 0.5263},
            {"name": "False_short", "weight": 0.222, "fpr": 0.1011, "fnr": 0.3000},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.1939, "fnr": 0.2222},
            {"name": "True_short", "weight": 0.141, "fpr": 0.1429, "fnr": 0.3077},
        ],
    },
    {
        "llm": "gpt-4o", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.0417, "fnr": 0.5789},
            {"name": "False_short", "weight": 0.222, "fpr": 0.0449, "fnr": 0.5000},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.0918, "fnr": 0.4815},
            {"name": "True_short", "weight": 0.141, "fpr": 0.0714, "fnr": 0.3077},
        ],
    },
    {
        "llm": "gpt-4-turbo", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.0417, "fnr": 0.3684},
            {"name": "False_short", "weight": 0.222, "fpr": 0.0899, "fnr": 0.5500},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.2041, "fnr": 0.2963},
            {"name": "True_short", "weight": 0.141, "fpr": 0.1071, "fnr": 0.1538},
        ],
    },
    {
        "llm": "gpt-4o-mini", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.0417, "fnr": 0.4211},
            {"name": "False_short", "weight": 0.222, "fpr": 0.0562, "fnr": 0.4000},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.1939, "fnr": 0.2963},
            {"name": "True_short", "weight": 0.141, "fpr": 0.1071, "fnr": 0.2308},
        ],
    },
    {
        "llm": "gpt-3.5-turbo", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.1429, "fnr": 0.2105},
            {"name": "False_short", "weight": 0.222, "fpr": 0.1798, "fnr": 0.1500},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.3776, "fnr": 0.1481},
            {"name": "True_short", "weight": 0.141, "fpr": 0.3929, "fnr": 0.1538},
        ],
    },
    {
        "llm": "gpt-4o", "dataset": "CivilComments", "K": 4, "p_z1": CIVIL_P_Z1,
        "subgroups": [
            {"name": "S0", "weight": 0.3877, "fpr": 0.2633, "fnr": 0.1823},
            {"name": "S1", "weight": 0.1137, "fpr": 0.2411, "fnr": 0.1477},
            {"name": "S2", "weight": 0.3373, "fpr": 0.2455, "fnr": 0.2862},
            {"name": "S3", "weight": 0.1613, "fpr": 0.3323, "fnr": 0.2949},
        ],
    },
    {
        "llm": "gpt-4o-mini", "dataset": "CivilComments", "K": 4, "p_z1": CIVIL_P_Z1,
        "subgroups": [
            {"name": "S0", "weight": 0.3877, "fpr": 0.4975, "fnr": 0.0563},
            {"name": "S1", "weight": 0.1137, "fpr": 0.5692, "fnr": 0.0455},
            {"name": "S2", "weight": 0.3373, "fpr": 0.4472, "fnr": 0.0989},
            {"name": "S3", "weight": 0.1613, "fpr": 0.5579, "fnr": 0.0769},
        ],
    },
    {
        "llm": "gpt-3.5-turbo", "dataset": "CivilComments", "K": 4, "p_z1": CIVIL_P_Z1,
        "subgroups": [
            {"name": "S0", "weight": 0.3877, "fpr": 0.3848, "fnr": 0.1233},
            {"name": "S1", "weight": 0.1137, "fpr": 0.4743, "fnr": 0.1023},
            {"name": "S2", "weight": 0.3373, "fpr": 0.4102, "fnr": 0.1519},
            {"name": "S3", "weight": 0.1613, "fpr": 0.4665, "fnr": 0.1987},
        ],
    },
    {
        "llm": "Llama-3.1-8B", "dataset": "TweetEval", "K": 2, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False", "weight": 0.604, "fpr": 0.252, "fnr": 0.269},
            {"name": "True", "weight": 0.396, "fpr": 0.186, "fnr": 0.233},
        ],
    },
    {
        "llm": "Qwen2.5-7B", "dataset": "TweetEval", "K": 2, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False", "weight": 0.604, "fpr": 0.1938, "fnr": 0.2335},
            {"name": "True", "weight": 0.396, "fpr": 0.1707, "fnr": 0.1830},
        ],
    },
    {
        "llm": "Gemma-2-9B", "dataset": "TweetEval", "K": 2, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False", "weight": 0.604, "fpr": 0.115, "fnr": 0.368},
            {"name": "True", "weight": 0.396, "fpr": 0.047, "fnr": 0.245},
        ],
    },
    {
        "llm": "Mistral-7B", "dataset": "TweetEval", "K": 2, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False", "weight": 0.604, "fpr": 0.191, "fnr": 0.246},
            {"name": "True", "weight": 0.396, "fpr": 0.209, "fnr": 0.155},
        ],
    },
]

# ============================================================
# Parameters
# ============================================================

N_SIM = 5000
N_MC = 100
SEED_BASE = 42


def build_mixing_matrix(fpr, fnr):
    """M[z_hat, z] = P(Ẑ=z_hat | Z=z). Column-stochastic."""
    return np.array([[1 - fpr, fnr], [fpr, 1 - fnr]])


def compute_scores_and_ground_truth(combo, n_mc=N_MC, seed_base=SEED_BASE, verbose=False):
    K = combo["K"]
    p_z1 = combo["p_z1"]
    subgroups = combo["subgroups"]

    weights = np.array([sg["weight"] for sg in subgroups])
    weights /= weights.sum()

    # --- Mixing matrices ---
    M_list = [build_mixing_matrix(sg["fpr"], sg["fnr"]) for sg in subgroups]
    M_global = sum(w * M for w, M in zip(weights, M_list))
    M_global_inv = np.linalg.inv(M_global)

    if verbose:
        print(f"  M_global:\n{M_global}")
        print(f"  M_global_inv:\n{M_global_inv}")

    # --- B_s and asymmetric scores ---
    B_list = []
    asym_scores = np.zeros((K, 2))
    for i, M_s in enumerate(M_list):
        B_s = M_global_inv @ M_s - np.eye(2)
        B_list.append(B_s)
        asym_scores[i, 0] = abs(B_s[0, 1])
        asym_scores[i, 1] = abs(B_s[1, 1])
        if verbose:
            print(f"  B_{subgroups[i]['name']}:\n{B_s}")
            print(f"    asym_score z=0: {asym_scores[i,0]:.4f}, z=1: {asym_scores[i,1]:.4f}")

    # --- Symmetric scores ---
    pi_list = np.array([(sg["fpr"] + sg["fnr"]) / 2 for sg in subgroups])
    pi_bar = weights @ pi_list
    delta_list = pi_list - pi_bar
    sym_scores = np.zeros(K)
    for i in range(K):
        denom = (1 - 2 * pi_bar) * pi_list[i]
        if abs(denom) < 1e-12:
            sym_scores[i] = 0.0
        else:
            sym_scores[i] = abs(delta_list[i]) / denom

    # --- True CATE ---
    if K == 4:
        tau_list = np.array([0.0, 0.1, 0.2, 0.3])
    elif K == 2:
        tau_list = np.array([0.0, 0.3])
    else:
        raise ValueError(f"Unexpected K={K}")

    # --- MC simulation ---
    bias_naive_acc = np.zeros(K)
    bias_global_acc = np.zeros(K)

    for mc in range(n_mc):
        rng = np.random.default_rng(seed_base + mc)
        D = rng.binomial(1, 0.5, N_SIM)
        S = rng.choice(K, N_SIM, p=weights)

        # True outcomes: P(Z=1|D=d,S=s) = p_z1 + tau_s * (d - 0.5)
        prob_z1 = np.zeros(N_SIM)
        for s in range(K):
            mask = S == s
            prob_z1[mask] = p_z1 + tau_list[s] * (D[mask] - 0.5)
        prob_z1 = np.clip(prob_z1, 0, 1)
        Z = rng.binomial(1, prob_z1)

        # Noisy labels
        Z_hat = Z.copy()
        for s in range(K):
            mask_s = S == s
            fpr_s = subgroups[s]["fpr"]
            fnr_s = subgroups[s]["fnr"]
            z0 = mask_s & (Z == 0)
            z1 = mask_s & (Z == 1)
            Z_hat[z0] = rng.binomial(1, fpr_s, z0.sum())
            Z_hat[z1] = 1 - rng.binomial(1, fnr_s, z1.sum())

        # CATE estimation per subgroup
        for s in range(K):
            mask_s = S == s
            d1 = mask_s & (D == 1)
            d0 = mask_s & (D == 0)
            if d1.sum() < 5 or d0.sum() < 5:
                continue

            p_hat_1 = Z_hat[d1].mean()
            p_hat_0 = Z_hat[d0].mean()
            tau_naive = p_hat_1 - p_hat_0

            obs_1 = np.array([1 - p_hat_1, p_hat_1])
            obs_0 = np.array([1 - p_hat_0, p_hat_0])
            corr_1 = M_global_inv @ obs_1
            corr_0 = M_global_inv @ obs_0
            tau_global = corr_1[1] - corr_0[1]

            bias_naive_acc[s] += tau_naive - tau_list[s]
            bias_global_acc[s] += tau_global - tau_list[s]

    bias_naive_avg = bias_naive_acc / n_mc
    bias_global_avg = bias_global_acc / n_mc
    ground_truth = (np.abs(bias_global_avg) > np.abs(bias_naive_avg)).astype(int)

    if verbose:
        for s in range(K):
            print(f"  {subgroups[s]['name']}: bias_naive={bias_naive_avg[s]:.4f}, "
                  f"bias_global={bias_global_avg[s]:.4f}, gt={ground_truth[s]}")

    return {
        "asym_scores": asym_scores,
        "sym_scores": sym_scores,
        "ground_truth": ground_truth,
        "bias_naive": bias_naive_avg,
        "bias_global": bias_global_avg,
    }


def main():
    t0 = time.time()

    # --- Dry run: first combo, 10 MC ---
    print("=" * 60)
    print("DRY RUN: Llama-3.1-8B / TweetEval / K=4, 10 MC")
    print("=" * 60)
    dry = compute_scores_and_ground_truth(COMBOS[0], n_mc=10, verbose=True)
    print(f"  Ground truth vector: {dry['ground_truth']}")
    print()

    # --- Full run ---
    print("=" * 60)
    print(f"FULL RUN: {len(COMBOS)} combos, {N_MC} MC each")
    print("=" * 60)

    rows = []
    for ci, combo in enumerate(COMBOS):
        label = f"{combo['llm']}/{combo['dataset']}/K={combo['K']}"
        print(f"  [{ci+1}/{len(COMBOS)}] {label} ...", end=" ", flush=True)
        res = compute_scores_and_ground_truth(combo)
        K = combo["K"]
        for s in range(K):
            for z in range(2):
                rows.append({
                    "model": combo["llm"],
                    "dataset": combo["dataset"],
                    "K": combo["K"],
                    "subgroup": combo["subgroups"][s]["name"],
                    "z_level": z,
                    "symmetric_score": res["sym_scores"][s],
                    "asymmetric_score": res["asym_scores"][s, z],
                    "ground_truth_global_hurts": res["ground_truth"][s],
                    "bias_naive": res["bias_naive"][s],
                    "bias_global": res["bias_global"][s],
                })
        n_hurts = res["ground_truth"].sum()
        print(f"done (hurts: {n_hurts}/{K})")

    df = pd.DataFrame(rows)
    df.to_csv("results/exp_asymmetric_diagnostic_auc.csv", index=False)
    print(f"\nSaved detailed results: results/exp_asymmetric_diagnostic_auc.csv ({len(df)} rows)")

    # --- AUC per (model, dataset, K) ---
    summary_rows = []
    for (model, dataset, K_val), grp in df.groupby(["model", "dataset", "K"]):
        # Use z_level=0 (scores are identical for z=0 and z=1 in binary case)
        sub = grp[grp["z_level"] == 0]
        gt = sub["ground_truth_global_hurts"].values
        if len(np.unique(gt)) < 2:
            sym_auc = float("nan")
            asym_auc = float("nan")
        else:
            sym_auc = roc_auc_score(gt, sub["symmetric_score"].values)
            asym_auc = roc_auc_score(gt, sub["asymmetric_score"].values)
        summary_rows.append({
            "model": model,
            "dataset": dataset,
            "K": K_val,
            "symmetric_auc": sym_auc,
            "asymmetric_auc": asym_auc,
            "n_subgroups": len(sub),
            "n_hurts": int(gt.sum()),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv("results/exp_asymmetric_diagnostic_auc_summary.csv", index=False)
    print(f"Saved summary: results/exp_asymmetric_diagnostic_auc_summary.csv\n")

    # --- Overall AUC ---
    all_z0 = df[df["z_level"] == 0]
    gt_all = all_z0["ground_truth_global_hurts"].values
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Total subgroup-z observations: {len(df)}")
    print(f"Unique (subgroup) observations: {len(all_z0)}")
    print(f"Global hurts count: {gt_all.sum()} / {len(gt_all)}")
    print()

    if len(np.unique(gt_all)) < 2:
        print("WARNING: ground truth is constant — cannot compute AUC")
    else:
        overall_sym = roc_auc_score(gt_all, all_z0["symmetric_score"].values)
        overall_asym = roc_auc_score(gt_all, all_z0["asymmetric_score"].values)
        print(f"Overall Symmetric  AUC: {overall_sym:.4f}")
        print(f"Overall Asymmetric AUC: {overall_asym:.4f}")
        print(f"Improvement:            {overall_asym - overall_sym:+.4f}")

    print()
    print("Per-(model, dataset, K) breakdown:")
    print("-" * 80)
    print(f"{'Model':<18} {'Dataset':<16} {'K':>2}  {'Sym AUC':>8}  {'Asym AUC':>9}  {'n_sg':>4}  {'n_hurt':>6}")
    print("-" * 80)
    for _, r in df_sum.iterrows():
        sym_str = f"{r['symmetric_auc']:.4f}" if not np.isnan(r["symmetric_auc"]) else "   N/A"
        asym_str = f"{r['asymmetric_auc']:.4f}" if not np.isnan(r["asymmetric_auc"]) else "    N/A"
        print(f"{r['model']:<18} {r['dataset']:<16} {r['K']:>2}  {sym_str:>8}  {asym_str:>9}  {r['n_subgroups']:>4}  {int(r['n_hurts']):>6}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
