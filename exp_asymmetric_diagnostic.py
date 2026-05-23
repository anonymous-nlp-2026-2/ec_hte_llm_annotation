"""
Asymmetric diagnostic for global MC correction harm detection.

Derives the correct non-symmetric diagnostic from B_s = M_global^{-1} M_s - I.
The CATE bias from global correction is tau_s * (B_s[1,1] - B_s[1,0]),
while naive bias is -tau_s * (fpr_s + fnr_s). The diagnostic ratio
R(s) = |B_s[1,1] - B_s[1,0]| / (fpr_s + fnr_s) is tau-free and
reduces to |delta_s|/((1-2pi_bar)*pi_s) under the symmetric assumption.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
import time
import os

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

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
            {"name": "True_medium", "weight": 0.255, "fpr": 0.0816, "fnr": 0.2593},
            {"name": "True_short", "weight": 0.141, "fpr": 0.0714, "fnr": 0.2308},
        ],
    },
    {
        "llm": "gpt-3.5-turbo", "dataset": "TweetEval", "K": 4, "p_z1": TWEET_P_Z1,
        "subgroups": [
            {"name": "False_medium", "weight": 0.382, "fpr": 0.0655, "fnr": 0.5263},
            {"name": "False_short", "weight": 0.222, "fpr": 0.1236, "fnr": 0.3500},
            {"name": "True_medium", "weight": 0.255, "fpr": 0.1020, "fnr": 0.3889},
            {"name": "True_short", "weight": 0.141, "fpr": 0.0714, "fnr": 0.3846},
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

N_SIM = 5000
N_MC = 200
SEED_BASE = 42


def build_cm(fpr, fnr):
    """M[z_hat, z] = P(Z_hat=z_hat | Z=z). Column-stochastic."""
    return np.array([[1 - fpr, fnr], [fpr, 1 - fnr]])


def compute_all_scores(combo):
    K = combo["K"]
    subgroups = combo["subgroups"]
    weights = np.array([sg["weight"] for sg in subgroups])
    weights /= weights.sum()

    M_list = [build_cm(sg["fpr"], sg["fnr"]) for sg in subgroups]
    M_global = sum(w * M for w, M in zip(weights, M_list))
    M_global_inv = np.linalg.inv(M_global)

    # Symmetric parameters
    pi_list = np.array([(sg["fpr"] + sg["fnr"]) / 2 for sg in subgroups])
    pi_bar = weights @ pi_list

    scores = {}
    for i in range(K):
        sg = subgroups[i]
        fpr_s, fnr_s = sg["fpr"], sg["fnr"]
        pi_s = pi_list[i]
        delta_s = pi_s - pi_bar

        B_s = M_global_inv @ M_list[i] - np.eye(2)

        # 1. Current symmetric: |delta_s| / ((1-2*pi_bar) * pi_s)
        denom_sym = (1 - 2 * pi_bar) * pi_s
        sym_score = abs(delta_s) / denom_sym if abs(denom_sym) > 1e-12 else 0.0

        # 2. Old exp028 asymmetric: |B_s[0,1]|
        asym_old = abs(B_s[0, 1])

        # 3. NEW: correct asymmetric ratio
        # bias_global ~ tau_s * (B_s[1,1] - B_s[1,0])
        # bias_naive  ~ -tau_s * (fpr_s + fnr_s)
        # Ratio = |B_s[1,1] - B_s[1,0]| / (fpr_s + fnr_s)
        cate_bias_coeff = B_s[1, 1] - B_s[1, 0]
        naive_attenuation = fpr_s + fnr_s
        asym_ratio = abs(cate_bias_coeff) / naive_attenuation if naive_attenuation > 1e-12 else 0.0

        # 4. Raw |B_s[1,1] - B_s[1,0]| (unnormalized)
        asym_raw = abs(cate_bias_coeff)

        scores[i] = {
            "sym_score": sym_score,
            "asym_old": asym_old,
            "asym_ratio": asym_ratio,
            "asym_raw": asym_raw,
            "B_s_11": B_s[1, 1],
            "B_s_10": B_s[1, 0],
            "cate_bias_coeff": cate_bias_coeff,
            "fpr": fpr_s,
            "fnr": fnr_s,
            "pi_s": pi_s,
            "delta_s": delta_s,
        }

    return scores, M_global_inv


def mc_ground_truth(combo, M_global_inv, n_mc=N_MC, seed_base=SEED_BASE):
    K = combo["K"]
    p_z1 = combo["p_z1"]
    subgroups = combo["subgroups"]
    weights = np.array([sg["weight"] for sg in subgroups])
    weights /= weights.sum()

    if K == 4:
        tau_list = np.array([0.0, 0.1, 0.2, 0.3])
    elif K == 2:
        tau_list = np.array([0.0, 0.3])
    else:
        raise ValueError(f"Unexpected K={K}")

    bias_naive_acc = np.zeros(K)
    bias_global_acc = np.zeros(K)

    for mc in range(n_mc):
        rng = np.random.default_rng(seed_base + mc)
        D = rng.binomial(1, 0.5, N_SIM)
        S = rng.choice(K, N_SIM, p=weights)

        prob_z1 = np.zeros(N_SIM)
        for s in range(K):
            mask = S == s
            prob_z1[mask] = p_z1 + tau_list[s] * (D[mask] - 0.5)
        prob_z1 = np.clip(prob_z1, 0, 1)
        Z = rng.binomial(1, prob_z1)

        Z_hat = Z.copy()
        for s in range(K):
            mask_s = S == s
            fpr_s = subgroups[s]["fpr"]
            fnr_s = subgroups[s]["fnr"]
            z0 = mask_s & (Z == 0)
            z1 = mask_s & (Z == 1)
            Z_hat[z0] = rng.binomial(1, fpr_s, z0.sum())
            Z_hat[z1] = 1 - rng.binomial(1, fnr_s, z1.sum())

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

    return ground_truth, bias_naive_avg, bias_global_avg, tau_list


def safe_auc(gt, scores):
    if len(np.unique(gt)) < 2:
        return float("nan")
    return roc_auc_score(gt, scores)


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for ci, combo in enumerate(COMBOS):
        label = f"{combo['llm']}/{combo['dataset']}/K={combo['K']}"
        print(f"  [{ci+1}/{len(COMBOS)}] {label} ...", end=" ", flush=True)

        scores, M_global_inv = compute_all_scores(combo)
        gt, bias_naive, bias_global, tau_list = mc_ground_truth(combo, M_global_inv)

        K = combo["K"]
        n_hurts = gt.sum()
        print(f"done (hurts: {n_hurts}/{K})")

        for s in range(K):
            sc = scores[s]
            rows.append({
                "model": combo["llm"],
                "dataset": combo["dataset"],
                "K": combo["K"],
                "subgroup": combo["subgroups"][s]["name"],
                "fpr": sc["fpr"],
                "fnr": sc["fnr"],
                "pi_s": sc["pi_s"],
                "delta_s": sc["delta_s"],
                "B_s_11": sc["B_s_11"],
                "B_s_10": sc["B_s_10"],
                "cate_bias_coeff": sc["cate_bias_coeff"],
                "sym_score": sc["sym_score"],
                "asym_old": sc["asym_old"],
                "asym_ratio": sc["asym_ratio"],
                "asym_raw": sc["asym_raw"],
                "bias_naive": bias_naive[s],
                "bias_global": bias_global[s],
                "tau_s": tau_list[s],
                "ground_truth": gt[s],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp_asym_diag_v2.csv"), index=False)

    # --- Overall AUC (pooled) ---
    # Exclude tau_s=0 subgroups (both biases ~0, ground truth is noise)
    df_nonzero = df[df["tau_s"] > 0]

    score_cols = ["sym_score", "asym_old", "asym_ratio", "asym_raw"]
    score_labels = [
        "Symmetric |d_s|/((1-2pi_bar)*pi_s)",
        "Old asym |B_s[0,1]|",
        "NEW asym ratio |B[1,1]-B[1,0]|/(fpr+fnr)",
        "NEW asym raw |B[1,1]-B[1,0]|",
    ]

    print("\n" + "=" * 70)
    print("OVERALL AUC (pooled across all combos, tau_s > 0 only)")
    print("=" * 70)
    print(f"  Subgroups with tau_s > 0: {len(df_nonzero)} / {len(df)}")
    print(f"  Ground truth hurts: {df_nonzero['ground_truth'].sum()} / {len(df_nonzero)}")
    print()

    gt_all = df_nonzero["ground_truth"].values
    for col, label in zip(score_cols, score_labels):
        auc_val = safe_auc(gt_all, df_nonzero[col].values)
        print(f"  {label}: AUC = {auc_val:.4f}")

    # Also with all subgroups (including tau_s=0)
    print("\n" + "=" * 70)
    print("OVERALL AUC (all subgroups including tau_s=0)")
    print("=" * 70)
    gt_all2 = df["ground_truth"].values
    print(f"  Total subgroups: {len(df)}")
    print(f"  Ground truth hurts: {df['ground_truth'].sum()} / {len(df)}")
    print()
    for col, label in zip(score_cols, score_labels):
        auc_val = safe_auc(gt_all2, df[col].values)
        print(f"  {label}: AUC = {auc_val:.4f}")

    # --- Per-combo AUC ---
    print("\n" + "=" * 70)
    print("PER-COMBO BREAKDOWN")
    print("=" * 70)
    summary_rows = []
    for (model, dataset, K_val), grp in df.groupby(["model", "dataset", "K"]):
        gt = grp["ground_truth"].values
        row = {"model": model, "dataset": dataset, "K": K_val,
               "n_subgroups": len(grp), "n_hurts": int(gt.sum())}
        for col in score_cols:
            row[f"auc_{col}"] = safe_auc(gt, grp[col].values)
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(RESULTS_DIR, "exp_asym_diag_v2_summary.csv"), index=False)

    header = f"{'Model':<18} {'Dataset':<15} {'K':>2}  {'Sym':>6}  {'Old':>6}  {'Ratio':>6}  {'Raw':>6}  {'n':>3}  {'hurt':>4}"
    print(header)
    print("-" * len(header))
    for _, r in df_sum.iterrows():
        vals = []
        for col in score_cols:
            v = r[f"auc_{col}"]
            vals.append(f"{v:.3f}" if not np.isnan(v) else "  N/A")
        print(f"{r['model']:<18} {r['dataset']:<15} {r['K']:>2}  "
              f"{vals[0]:>6}  {vals[1]:>6}  {vals[2]:>6}  {vals[3]:>6}  "
              f"{r['n_subgroups']:>3}  {int(r['n_hurts']):>4}")

    # --- Diagnostic: show B_s details for each combo ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC DETAILS (B_s entries per subgroup)")
    print("=" * 70)
    for ci, combo in enumerate(COMBOS):
        label = f"{combo['llm']}/{combo['dataset']}/K={combo['K']}"
        sub = df[(df["model"] == combo["llm"]) &
                 (df["dataset"] == combo["dataset"]) &
                 (df["K"] == combo["K"])]
        print(f"\n{label}:")
        print(f"  {'Subgroup':<15} {'FPR':>6} {'FNR':>6}  {'B[1,1]':>8} {'B[1,0]':>8}  "
              f"{'Sym':>6} {'Ratio':>6}  {'b_naiv':>7} {'b_glob':>7}  {'GT':>2}")
        for _, r in sub.iterrows():
            print(f"  {r['subgroup']:<15} {r['fpr']:>6.3f} {r['fnr']:>6.3f}  "
                  f"{r['B_s_11']:>8.4f} {r['B_s_10']:>8.4f}  "
                  f"{r['sym_score']:>6.3f} {r['asym_ratio']:>6.3f}  "
                  f"{r['bias_naive']:>7.4f} {r['bias_global']:>7.4f}  "
                  f"{int(r['ground_truth']):>2}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
