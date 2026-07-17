# Ablation Result Files

Each JSON is one ablation run, named:

```
{model}__{strategy}__{components}__f{fraction}__s{seed}.json
```

Fields: `model`, `strategy`, `components`, `z`, `fraction`, `seed`,
`n_zeroed`, and `results` (raw lm-evaluation-harness output for
HellaSwag, MMLU with subcategories, and PIQA).

Regenerate the summary CSV with:

```bash
python ../src/aggregate_results.py --input_dir . --output_csv summary.csv
```
