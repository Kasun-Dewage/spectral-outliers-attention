import os
import json
import argparse


ACC_KEYS = ["acc,none", "acc_norm,none", "acc", "acc_norm"]
TASKS = ["hellaswag", "piqa", "mmlu", "arc_easy", "arc_challenge",
         "winogrande", "mmlu_stem", "mmlu_humanities", "mmlu_social_sciences",
         "mmlu_other"]


def extract_acc(res, task):
    if task not in res:
        return None
    tr = res[task]
    for k in ACC_KEYS:
        if k in tr:
            try:
                return float(tr[k])
            except Exception:
                continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_csv", type=str, default="summary.csv")
    args = p.parse_args()

    rows = []
    for fn in sorted(os.listdir(args.input_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(args.input_dir, fn)
        try:
            with open(path, "r") as f:
                d = json.load(f)
        except Exception:
            continue
        res = d.get("results", {}) or {}
        row = {
            "file": fn,
            "model": d.get("model", ""),
            "strategy": d.get("strategy", ""),
            "components": "_".join(d.get("components", []) or []),
            "fraction": d.get("fraction", ""),
            "z": d.get("z", ""),
            "seed": d.get("seed", ""),
            "n_zeroed": d.get("n_zeroed", ""),
        }
        for t in TASKS:
            row[t] = extract_acc(res, t)
        rows.append(row)

    if not rows:
        print("No results found in", args.input_dir)
        return

    keys = list(rows[0].keys())
    with open(args.output_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if v is None:
                    v = ""
                vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print("Wrote", args.output_csv, "rows=", len(rows))


if __name__ == "__main__":
    main()
