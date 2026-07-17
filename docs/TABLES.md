# Paper Tables

Key tables reproduced from *Spectral Outliers Reveal Dominant Learned Structure in Transformer Attention*.

---

## Table I — Models Analyzed

`d` = hidden size, `L` = layers, `n_h` = attention heads, `n_kv` = key/value heads. GQA = Grouped-Query Attention, MHA = Multi-Head Attention.

| Model | d | L | n_h | n_kv | Attn. |
|---|---:|---:|---:|---:|:--:|
| BERT-base | 768 | 12 | 12 | 12 | MHA |
| RoBERTa-base | 768 | 12 | 12 | 12 | MHA |
| OPT-125M | 768 | 12 | 12 | 12 | MHA |
| LLaMA-1-7B | 4096 | 32 | 32 | 32 | MHA |
| LLaMA-2-7B | 4096 | 32 | 32 | 32 | MHA |
| LLaMA-3-8B | 4096 | 32 | 32 | 8 | GQA |
| Mistral-7B | 4096 | 32 | 32 | 8 | GQA |
| Qwen2.5-0.5B | 896 | 24 | 14 | 2 | GQA |
| Qwen2.5-1.5B | 1536 | 28 | 12 | 2 | GQA |
| Qwen2.5-7B | 3584 | 28 | 28 | 4 | GQA |
| Phi-3-mini | 3072 | 32 | 32 | 32 | MHA |

---

## Table II — Average MP Spectral Outliers per Layer & Mean Energy Ratio (%)

| Model | Q avg | Q e% | K avg | K e% | V avg | V e% | O avg | O e% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Encoder-Only (MHA)* | | | | | | | | |
| BERT-base | 275 | 86.5 | 276 | 86.7 | 268 | 82.5 | 261 | 79.7 |
| RoBERTa-base | 282 | 87.5 | 282 | 87.6 | 271 | 83.6 | 253 | 79.6 |
| *Decoder-Only (MHA)* | | | | | | | | |
| OPT-125M | 299 | 91.8 | 296 | 90.5 | 271 | 83.5 | 267 | 82.7 |
| LLaMA-1-7B | 1509 | 88.7 | 1509 | 88.5 | 1391 | 79.9 | 1398 | 79.7 |
| LLaMA-2-7B | 1519 | 87.8 | 1517 | 87.5 | 1392 | 79.0 | 1393 | 79.1 |
| Phi-3-mini | — | — | — | — | — | — | 1051 | 79.8 |
| *Decoder-Only (GQA)* | | | | | | | | |
| LLaMA-3-8B | 1535 | 90.1 | 341 | 75.4 | 219 | 43.1 | 1490 | 86.2 |
| Mistral-7B | 1511 | 87.5 | 341 | 74.7 | 212 | 43.6 | 1450 | 84.6 |
| Qwen-0.5B | 350 | 93.6 | 44 | 71.1 | 13 | 15.6 | 349 | 91.9 |
| Qwen-1.5B | 587 | 92.7 | 86 | 71.2 | 27 | 18.2 | 570 | 88.4 |
| Qwen-7B | 1324 | 89.0 | 170 | 68.8 | 73 | 25.7 | 1290 | 85.6 |

---

## Table III — Total Entry-Level Outlier Counts (summed across layers), % share per model

| Model | Q | Q% | K | K% | V | V% | O | O% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *MHA models* | | | | | | | | |
| BERT-base | 3,305 | 25.5 | 3,306 | 25.5 | 3,213 | 24.8 | 3,127 | 24.1 |
| RoBERTa-base | 3,381 | 25.9 | 3,388 | 25.9 | 3,251 | 24.9 | 3,037 | 23.3 |
| OPT-125M | 3,593 | 26.4 | 3,547 | 26.1 | 3,257 | 23.9 | 3,209 | 23.6 |
| LLaMA-1-7B | 48,295 | 26.0 | 48,288 | 26.0 | 44,521 | 24.0 | 44,733 | 24.1 |
| LLaMA-2-7B | 48,599 | 26.1 | 48,542 | 26.1 | 44,553 | 23.9 | 44,583 | 23.9 |
| *GQA models* | | | | | | | | |
| LLaMA-3-8B | 49,133 | 42.8 | 10,921 | 9.5 | 7,012 | 6.1 | 47,687 | 41.6 |
| Mistral-7B | 48,365 | 43.0 | 10,915 | 9.7 | 6,785 | 6.0 | 46,406 | 41.3 |
| Qwen2.5-0.5B | 8,400 | 46.3 | 1,059 | 5.8 | 315 | 1.7 | 8,366 | 46.1 |
| Qwen2.5-7B | 37,080 | 46.4 | 4,758 | 5.9 | 2,046 | 2.6 | 36,107 | 45.1 |

---

## Table IV — Persistent Residual-Stream Dimensions (≥ 3 layers as band outlier) per Component

| Model | Q | K | V | O | Union |
|---|---:|---:|---:|---:|---:|
| BERT-base | 3 | 7 | 0 | 2 | 11 |
| RoBERTa-base | 8 | 10 | 0 | 1 | 11 |
| OPT-125M | 15 | 17 | 0 | 12 | 24 |
| LLaMA-1-7B | 21 | 51 | 2 | 15 | 52 |
| LLaMA-2-7B | 33 | 52 | 1 | 17 | 54 |
| LLaMA-3-8B | 87 | 98 | 44 | 43 | 164 |

---

## Table V — Top Cross-Component Persistent Dimensions

"Hits" = number of Q, K, V, O components in which the dimension appears at least once.

| Model | Dim | Hits | Total | Q | K | V | O |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT | 381 | 3 | 16 | 7 | 8 | 0 | 1 |
| BERT | 308 | 1 | 8 | 0 | 0 | 0 | 8 |
| OPT | 174 | 3 | 23 | 10 | 10 | 0 | 3 |
| OPT | 638 | 1 | 7 | 0 | 0 | 0 | 7 |
| LLaMA-2 | 2533 | 4 | 58 | 32 | 12 | 1 | 13 |
| LLaMA-2 | 3431 | 3 | 61 | 25 | 28 | 0 | 8 |
| LLaMA-2 | 1512 | 3 | 37 | 3 | 2 | 0 | 32 |
| LLaMA-3 | 2977 | 3 | 54 | 24 | 22 | 0 | 8 |
| LLaMA-3 | 4055 | 1 | 30 | 0 | 0 | 0 | 30 |
| LLaMA-3 | 373 | 3 | 48 | 16 | 23 | 0 | 9 |

---

## Table VI — Ablation Results

`N_zero` = singular values / entries zeroed. HS = HellaSwag, PIQA = acc_norm. Random chance: HS ≈ 0.250, MMLU = 0.250, PIQA ≈ 0.500.

| Model | Strategy | Comp. | f | N_zero | HS | ΔHS | MMLU | ΔMMLU | PIQA | ΔPIQA |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| *Baselines* | | | | | | | | | | |
| LLaMA-1-7B | — | — | — | — | .761 | — | .352 | — | .793 | — |
| LLaMA-2-7B | — | — | — | — | .760 | — | .458 | — | .790 | — |
| LLaMA-3-8B | — | — | — | — | .821 | — | .660 | — | .812 | — |
| Mistral-7B | — | — | — | — | .814 | — | .627 | — | .823 | — |
| *Bulk (below-threshold) removal* | | | | | | | | | | |
| LLaMA-1-7B | random_bulk | QKVO | 0.75 | 139,423 | .754 | −.007 | .292 | −.060 | .791 | −.002 |
| LLaMA-2-7B | random_bulk | QKVO | 0.75 | 139,754 | .737 | −.023 | .358 | −.100 | .773 | −.017 |
| LLaMA-2-7B | random_bulk | QKVO | 1.00 | 186,273 | .710 | −.050 | .329 | −.129 | .752 | −.038 |
| *Outlier (signal) removal* | | | | | | | | | | |
| Mistral-7B | outliers | QKVO | 1.00 | 112,458 | .256 | −.558 | .269 | −.358 | .508 | −.315 |
| *Per-component ablations (LLaMA-3-8B)* | | | | | | | | | | |
| LLaMA-3-8B | entry_outliers | Q | 1.00 | 883,337 | .757 | −.064 | .488 | −.172 | .792 | −.020 |
| LLaMA-3-8B | entry_outliers | K | 1.00 | 234,239 | .728 | −.093 | .442 | −.218 | .770 | −.042 |
| LLaMA-3-8B | entry_outliers | V | 1.00 | 98,866 | .793 | −.028 | .612 | −.048 | .799 | −.013 |
| LLaMA-3-8B | random_bulk | O | 1.00 | 47,676 | .789 | −.032 | .606 | −.054 | .797 | −.015 |
| LLaMA-3-8B | random_bulk | V | 1.00 | 7,013 | .649 | −.172 | .254 | −.406 | .779 | −.033 |

---

## Table VII — MMLU Subcategory Accuracy under Selected Ablations

| Experiment | STEM | Hum | SocSci | Other |
|---|---:|---:|---:|---:|
| Mistral outliers QKVO | .286 | .242 | .311 | .251 |
| LL2 random-bulk QKVO f=0.75 | .305 | .335 | .386 | .420 |
| LL3 entry K | .394 | .375 | .541 | .494 |
| LL3 entry Q | .439 | .403 | .599 | .558 |
| LL3 entry V | .536 | .538 | .717 | .698 |
| LL3 random-bulk V | .252 | .246 | .255 | .265 |
