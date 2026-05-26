# When Debiasing Hurts: An Exact Bias Identity for LLM-Annotated Treatment Effects

This repository contains the code and data for the paper **"When Debiasing Hurts: An Exact Bias Identity for LLM-Annotated Treatment Effects"**.

Large language models (LLMs) increasingly replace human annotators in computational social science, yet their error rates can vary by up to 5x across demographic and text-feature subgroups. When these noisy labels serve as effect modifiers in heterogeneous treatment effect (HTE) analysis, the standard remedy -- estimating a single confusion matrix and applying its inverse uniformly -- is widely trusted to remove bias. We prove it can do the opposite. We derive an exact bias identity showing that global correction introduces a subgroup-level bias proportional to the interaction of error-rate deviation and effect heterogeneity, a product structure invisible to any pooled analysis. Guided by the identity, we propose **EC-HTE** (Error-Corrected HTE), a per-subgroup correction via hierarchical Bayesian confusion matrix estimation that reduces bias by 7.4x under extreme heterogeneity, validated across 11 LLM-dataset combinations spanning four NLP tasks.

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**API Configuration (for annotation scripts only):**

Scripts that call LLM APIs (exp004, exp008*, exp023*, pilot_*) require:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # or your API endpoint
```

## Project Structure

| File | Description |
|------|-------------|
| `exp001_hb_synthetic.py` | Main synthetic validation (hierarchical Bayesian EC-HTE) |
| `exp002_bias_bound.py` | Bias identity verification (Theorem 1) |
| `exp003_tweeteval.py` | TweetEval semi-synthetic benchmark (K=4) |
| `exp004_hate_speech.py` | UC Berkeley hate speech case study |
| `exp005_diagnostic_criterion.py` | Diagnostic criterion (Proposition 1) |
| `exp006_budget_sensitivity.py` | Expert label budget sensitivity |
| `exp007_ppci_dsl.py` | EC-HTE + PPCI composition |
| `exp008_multi_llm.py` | Multi-LLM bias identity validation (9 LLMs) |
| `exp011_causal_forest.py` | CausalForestDML integration |
| `exp014_asymmetric.py` | Asymmetric misclassification robustness |
| `exp017_rlearner.py` | R-learner validation |
| `exp023_civil_comments_*.py` | CivilComments demographic subgroup analysis |
| `exp029_a1_sensitivity.py` | Assumption A1 sensitivity analysis |
| `exp030_k8_subgroups.py` | K=8 scalability boundary |
| `exp034_fl_comparison.py` | Frazis-Loewenstein baseline comparison |
| `exp_stance_detection.py` | Stance detection cross-task validation (K=5) |
| `exp_treatment_dep_misclass.py` | Treatment-dependent misclassification robustness |
| `draw_fig2.py` | Figure 3: bias amplification heatmap |
| `draw_fig3.py` | Figure 4: budget sensitivity plot |
| `scripts/fig1_bias_mechanism.py` | Figure 1: bias mechanism schematic |
| `run_table1.sh` | Run full Table 1 reproduction |
| `data/` | Preprocessed data files (.parquet) |

## Reproducing Results

### Core Experiments

```bash
# Main synthetic experiment (Table 1 + supplements)
python exp001_hb_synthetic.py

# Full Table 1 (includes Frazis-Loewenstein baselines)
bash run_table1.sh

# TweetEval semi-synthetic benchmark
python exp003_tweeteval.py

# Hate speech case study (requires OPENAI_API_KEY for annotation phase)
python exp004_hate_speech.py --phase prep
python exp004_hate_speech.py --phase annotate
python exp004_hate_speech.py --phase analyze

# CausalForest integration
python exp011_causal_forest.py
```

### Figures

```bash
python draw_fig2.py                    # Figure 3: bias heatmap
python draw_fig3.py                    # Figure 4: budget sensitivity
python scripts/fig1_bias_mechanism.py  # Figure 1: bias mechanism
```

### Robustness and Extensions

```bash
python exp014_asymmetric.py            # Asymmetric misclassification
python exp029_a1_sensitivity.py        # Assumption A1 sensitivity
python exp030_k8_subgroups.py          # K=8 scalability
python exp034_fl_comparison.py         # Frazis-Loewenstein comparison
python exp_stance_detection.py         # Stance detection (K=5)
python exp_treatment_dep_misclass.py   # Treatment-dependent misclassification
```

### Multi-LLM Validation

```bash
python exp008_multi_llm.py             # 9 LLMs x 2 datasets
python exp024_realcm_bias_identity.py  # Real CM bias identity validation
```

## Data

- **Synthetic data**: Generated automatically by experiment scripts.
- **TweetEval**: Downloaded automatically via HuggingFace `datasets` library.
- **CivilComments**: Downloaded automatically via HuggingFace `datasets` library. Preprocessed subsets in `data/`.
- **UC Berkeley Hate Speech**: Preprocessed subsets in `data/`.
- **AG News**: Downloaded automatically via HuggingFace `datasets` library.

## License

MIT
