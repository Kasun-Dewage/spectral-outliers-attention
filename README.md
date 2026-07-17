<div align="center">

# Spectral Outliers Reveal Dominant Learned Structure in Transformer Attention

**A Marchenko–Pastur random matrix analysis of attention projection weights across 11 pretrained transformers, with causal validation via ablation.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://github.com/huggingface/transformers)
[![lm-eval](https://img.shields.io/badge/lm--evaluation--harness-v0.4-green.svg)](https://github.com/EleutherAI/lm-evaluation-harness)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

</div>

---

## Overview

We apply **Marchenko–Pastur (MP) random matrix theory** to decompose the attention projection weights (**Q, K, V, O**) of pretrained transformers into a random-like **bulk** and a structured **outlier** component. Beyond scalar spectral summaries, this repository contains code for:

- **Entry-level outlier heatmaps** — spatial positions of large-magnitude weights, revealing row-bands in Q and column-bands in O.
- **Cross-layer residual-stream alignment** — persistent "information highway" dimensions in K and O.
- **Causal ablation experiments** — zeroing MP-identified spectral outliers collapses performance to near random chance, while zeroing count-matched bulk singular values incurs far smaller degradation.

The models span encoder-only (BERT, RoBERTa) and decoder-only (OPT, LLaMA 1/2/3, Mistral, Qwen 2.5, Phi-3) designs, covering both Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).

## Key Findings

1. **Spectral outliers carry a dominant learned structure.** Zeroing all 112,458 spectral outliers in Mistral-7B drops HellaSwag `0.814 → 0.256`, MMLU `0.627 → 0.269`, and PIQA `0.823 → 0.508` — essentially random chance.
2. **Bulk singular values are largely expendable.** Removing count-matched below-threshold singular values causes only modest degradation (though knowledge-intensive MMLU is more sensitive).
3. **Q dominates spectrally; V is least structured under GQA** — V outlier counts drop 6–27× moving from MHA to GQA.
4. **Per-component criticality follows K > Q ≫ V** in the tested LLaMA-3-8B setting — K outliers are sparse but carry the highest per-parameter impact.
5. **Entry outliers form structured bands** — row-bands in Q, column-bands in O — and specific residual-stream dimensions persist as outliers across layers in K and O.

## Repository Structure

```
.
├── src/
│   ├── mp_analysis_pretrained.py   MP spectral analysis, entry heatmaps, cross-layer alignment, 3D stacks
│   ├── ablation_experiment.py      Spectral / entry / band ablations + zero-shot eval
│   └── aggregate_results.py        Collate result JSONs into a summary CSV
├── results/                        Raw ablation result JSON files (per experiment)
├── paper/                          The full paper (PDF)
├── docs/
│   └── TABLES.md                   Key tables reproduced from the paper
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/Kasun-Dewage/spectral-outliers-attention.git
cd spectral-outliers-attention

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Access-gated models (LLaMA, Mistral) require a Hugging Face token:

```bash
huggingface-cli login
```

## Usage

### 1. Marchenko–Pastur spectral analysis

Analyze one or more models, producing per-model JSON summaries, entry-level outlier heatmaps, 3D attention stacks, and persistence heatmaps.

```bash
python src/mp_analysis_pretrained.py --list

python src/mp_analysis_pretrained.py \
    --models llama2-7b llama3-8b mistral-7b \
    --output_dir ./output/mp_analysis
```

### 2. Ablation experiments

Apply an ablation strategy to the attention projections, then evaluate zero-shot on HellaSwag, MMLU, and PIQA.

```bash
python src/ablation_experiment.py \
    --model mistral-7b \
    --strategy spectral_outliers \
    --components Q K V O \
    --tasks hellaswag mmlu piqa \
    --output_dir ./output/ablation
```

**Strategies**

| Strategy | Description |
|---|---|
| `spectral_outliers` | Zero singular values above the MP upper edge (remove signal) |
| `spectral_random_bulk` | Zero a random subset of below-threshold singular values, count-matched to `f × N_outlier` |
| `spectral_bulk_top` / `spectral_bottom` | Zero the largest / smallest below-threshold singular values |
| `entry_outliers` | Zero all entries with \|w\| > z·σ |
| `entry_random` | Zero a count-matched random set of non-outlier entries |
| `band_outliers` / `band_random` | Zero whole rows/columns whose norm exceeds μ + z·σ |

### 3. Aggregate results

```bash
python src/aggregate_results.py \
    --input_dir ./output/ablation \
    --output_csv summary.csv
```

## Ablation Results

Zero-shot accuracy. `N_zero` = singular values / entries zeroed. Random chance: HS ≈ 0.250, MMLU = 0.250, PIQA ≈ 0.500.

| Model | Strategy | Comp. | f | N_zero | HS | ΔHS | MMLU | ΔMMLU | PIQA | ΔPIQA |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-1-7B | baseline | — | — | — | .761 | — | .352 | — | .793 | — |
| LLaMA-2-7B | baseline | — | — | — | .760 | — | .458 | — | .790 | — |
| LLaMA-3-8B | baseline | — | — | — | .821 | — | .660 | — | .812 | — |
| Mistral-7B | baseline | — | — | — | .814 | — | .627 | — | .823 | — |
| LLaMA-1-7B | random_bulk | QKVO | 0.75 | 139,423 | .754 | −.007 | .292 | −.060 | .791 | −.002 |
| LLaMA-2-7B | random_bulk | QKVO | 0.75 | 139,754 | .737 | −.023 | .358 | −.100 | .773 | −.017 |
| LLaMA-2-7B | random_bulk | QKVO | 1.00 | 186,273 | .710 | −.050 | .329 | −.129 | .752 | −.038 |
| **Mistral-7B** | **outliers** | **QKVO** | **1.00** | **112,458** | **.256** | **−.558** | **.269** | **−.358** | **.508** | **−.315** |
| LLaMA-3-8B | entry_outliers | Q | 1.00 | 883,337 | .757 | −.064 | .488 | −.172 | .792 | −.020 |
| LLaMA-3-8B | entry_outliers | K | 1.00 | 234,239 | .728 | −.093 | .442 | −.218 | .770 | −.042 |
| LLaMA-3-8B | entry_outliers | V | 1.00 | 98,866 | .793 | −.028 | .612 | −.048 | .799 | −.013 |
| LLaMA-3-8B | random_bulk | O | 1.00 | 47,676 | .789 | −.032 | .606 | −.054 | .797 | −.015 |
| LLaMA-3-8B | random_bulk | V | 1.00 | 7,013 | .649 | −.172 | .254 | −.406 | .779 | −.033 |

More tables (spectral outlier counts, cross-layer persistence, MMLU subcategories) are in [`docs/TABLES.md`](docs/TABLES.md).

## Paper

The full paper is available in [`paper/`](paper/). If you use this work, please cite:

```bibtex
@article{dewage2024spectral,
  title   = {Spectral Outliers Reveal Dominant Learned Structure in Transformer Attention},
  author  = {Dewage, Kasun and Pensky, Marianna and De Silva, Suranadi and Bandara, T. H.},
  year    = {2024}
}
```

## License

Released under the [MIT License](LICENSE).
