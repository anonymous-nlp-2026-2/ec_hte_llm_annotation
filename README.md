# EC-HTE: Error-Corrected Heterogeneous Treatment Effects

This repository contains the code for **EC-HTE**, a framework for correcting heterogeneous measurement error in treatment effect estimation when using LLM-based annotations. EC-HTE leverages subgroup-specific confusion matrices to debias LLM misclassification and recover accurate heterogeneous treatment effects without requiring ground-truth labels at inference time.

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
| `exp002_bias_bound.py` | Bias identity verification under A1-A4 |
| `exp003_tweeteval.py` | TweetEval semi-synthetic benchmark |
| `exp004_hate_speech.py` | Hate speech case study (Civil Comments) |
| `exp005_diagnostic_criterion.py` | Diagnostic criterion for subgroup heterogeneity |
| `exp006_budget_sensitivity.py` | Budget sensitivity analysis |
| `exp007_ppci_dsl.py` | PPI-DSL comparison |
| `exp008_multi_llm.py` | Multi-LLM misclassification comparison |
| `exp011_causal_forest.py` | CausalForestDML integration |
| `exp_centered_eb_table1.py` | Table 1 results (centered empirical Bayes) |
| `draw_fig2.py` | Figure 2: bias heatmap |
| `scripts/fig1_bias_mechanism.py` | Figure 1: bias mechanism schematic |
| `run_table1.sh` | Run full Table 1 reproduction |
| `data/` | Preprocessed data files (.parquet) |

## Reproducing Results

### Core Experiments

```bash
# Main synthetic experiment (Table 1 + supplements)
python exp001_hb_synthetic.py

# Full Table 1
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
python draw_fig2.py                    # Figure 2: bias heatmap
python scripts/fig1_bias_mechanism.py  # Figure 1: bias mechanism
```

### Additional Experiments

```bash
python exp002_bias_bound.py            # Bias identity verification
python exp005_diagnostic_criterion.py  # Diagnostic criterion
python exp006_budget_sensitivity.py    # Budget sensitivity
python exp007_ppci_dsl.py              # PPI-DSL comparison
```

## Data

- **Synthetic data**: Generated automatically by experiment scripts.
- **TweetEval**: Downloaded automatically via HuggingFace `datasets` library.
- **Civil Comments**: Downloaded automatically via HuggingFace `datasets` library. Preprocessed subsets are included in `data/`.

## License

MIT
