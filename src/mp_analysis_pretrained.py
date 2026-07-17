import math
import json
import os
import sys
import re
import gc
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def marchenko_pastur_threshold(W):
    W_2d = W.detach().float().view(W.shape[0], -1)
    m, n = W_2d.shape
    gamma = max(m, n) / min(m, n)
    S = torch.linalg.svdvals(W_2d)
    sq = S ** 2
    sigma_sq = float(sq.median()) / (1.0 + gamma)
    lambda_plus = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    return lambda_plus


def compute_spectral_energy_ratio(W):
    W_2d = W.detach().float().view(W.shape[0], -1)
    S = torch.linalg.svdvals(W_2d)
    threshold = marchenko_pastur_threshold(W)
    sq = S ** 2
    outlier_mask = sq > threshold
    n_outliers = int(outlier_mask.sum().item())
    total_energy = float(sq.sum())
    if total_energy < 1e-12:
        return n_outliers, 0.0, threshold
    outlier_energy = float(sq[outlier_mask].sum())
    ratio = outlier_energy / total_energy
    return n_outliers, ratio, float(threshold)


def compute_entry_outlier_mask(W, z_thresh=4.0):
    W_2d = W.detach().float().view(W.shape[0], -1).cpu().numpy()
    std = float(W_2d.std())
    if std < 1e-12:
        return np.zeros_like(W_2d, dtype=bool), W_2d.shape
    mask = np.abs(W_2d) > z_thresh * std
    return mask, W_2d.shape


def compute_axis_band_outliers(W, axis, z_thresh=4.0):
    W_2d = W.detach().float().view(W.shape[0], -1).cpu().numpy()
    norms = np.linalg.norm(W_2d, axis=axis)
    mean = float(norms.mean())
    std = float(norms.std())
    if std < 1e-12:
        return np.array([], dtype=int), norms, mean, std
    mask = norms > mean + z_thresh * std
    return np.where(mask)[0].astype(int), norms, mean, std


MODELS = OrderedDict([
    ("bert-base-uncased",  {"hf_name": "bert-base-uncased",                "arch": "bert"}),
    ("roberta-base",       {"hf_name": "roberta-base",                     "arch": "roberta"}),
    ("gpt2",               {"hf_name": "gpt2",                             "arch": "gpt2"}),
    ("gpt2-medium",        {"hf_name": "gpt2-medium",                      "arch": "gpt2"}),
    ("gpt2-large",         {"hf_name": "gpt2-large",                       "arch": "gpt2"}),
    ("gpt2-xl",            {"hf_name": "gpt2-xl",                          "arch": "gpt2"}),
    ("opt-125m",           {"hf_name": "facebook/opt-125m",                "arch": "opt"}),
    ("llama1-7b",          {"hf_name": "huggyllama/llama-7b",              "arch": "llama"}),
    ("llama2-7b",          {"hf_name": "meta-llama/Llama-2-7b-hf",         "arch": "llama"}),
    ("llama3-8b",          {"hf_name": "meta-llama/Meta-Llama-3-8B",       "arch": "llama"}),
    ("llama3.2-1b",        {"hf_name": "meta-llama/Llama-3.2-1B",          "arch": "llama"}),
    ("mistral-7b",         {"hf_name": "mistralai/Mistral-7B-v0.1",        "arch": "llama"}),
    ("gemma-2b",           {"hf_name": "google/gemma-2b",                  "arch": "llama"}),
    ("gemma-2-2b",         {"hf_name": "google/gemma-2-2b",                "arch": "llama"}),
    ("qwen2.5-0.5b",       {"hf_name": "Qwen/Qwen2.5-0.5B",                "arch": "llama"}),
    ("qwen2.5-1.5b",       {"hf_name": "Qwen/Qwen2.5-1.5B",                "arch": "llama"}),
    ("qwen2.5-7b",         {"hf_name": "Qwen/Qwen2.5-7B",                  "arch": "llama"}),
    ("phi-3-mini",         {"hf_name": "microsoft/Phi-3-mini-4k-instruct", "arch": "llama"}),
])


def parse_layer_index_generic(name):
    patterns = [
        r"layer\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"layers\.(\d+)\.",
        r"\.block\.(\d+)\.",
    ]
    for pat in patterns:
        match = re.search(pat, name)
        if match:
            return int(match.group(1))
    return -1


def extract_qkvo_bert_like(model):
    out = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index_generic(name)
        if layer_idx < 0:
            continue
        comp = None
        if name.endswith(".query"):
            comp = "Q"
        elif name.endswith(".key"):
            comp = "K"
        elif name.endswith(".value"):
            comp = "V"
        elif "attention.output.dense" in name:
            comp = "O"
        if comp is None:
            continue
        out.setdefault(layer_idx, {})[comp] = module.weight.detach().float().cpu()
    return out


def extract_qkvo_gpt2(model):
    out = OrderedDict()
    for name, module in model.named_modules():
        layer_idx = parse_layer_index_generic(name)
        if layer_idx < 0:
            continue
        if name.endswith(".attn.c_attn"):
            W = module.weight.detach().float().cpu()
            three_h = W.shape[1]
            head_dim = three_h // 3
            W_q = W[:, :head_dim].T.contiguous()
            W_k = W[:, head_dim:2 * head_dim].T.contiguous()
            W_v = W[:, 2 * head_dim:].T.contiguous()
            d = out.setdefault(layer_idx, {})
            d["Q"], d["K"], d["V"] = W_q, W_k, W_v
        elif name.endswith(".attn.c_proj"):
            W = module.weight.detach().float().cpu()
            out.setdefault(layer_idx, {})["O"] = W.T.contiguous()
    return out


def extract_qkvo_opt(model):
    out = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index_generic(name)
        if layer_idx < 0:
            continue
        comp = None
        if name.endswith(".q_proj"):
            comp = "Q"
        elif name.endswith(".k_proj"):
            comp = "K"
        elif name.endswith(".v_proj"):
            comp = "V"
        elif name.endswith(".out_proj"):
            comp = "O"
        if comp is None:
            continue
        out.setdefault(layer_idx, {})[comp] = module.weight.detach().float().cpu()
    return out


def extract_qkvo_llama(model):
    out = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = parse_layer_index_generic(name)
        if layer_idx < 0:
            continue
        comp = None
        if name.endswith(".q_proj"):
            comp = "Q"
        elif name.endswith(".k_proj"):
            comp = "K"
        elif name.endswith(".v_proj"):
            comp = "V"
        elif name.endswith(".o_proj"):
            comp = "O"
        if comp is None:
            continue
        out.setdefault(layer_idx, {})[comp] = module.weight.detach().float().cpu()
    return out


EXTRACTORS = {
    "bert":    extract_qkvo_bert_like,
    "roberta": extract_qkvo_bert_like,
    "gpt2":    extract_qkvo_gpt2,
    "opt":     extract_qkvo_opt,
    "llama":   extract_qkvo_llama,
}


def analyze_qkvo_dict(qkvo, band_z=4.0):
    results = OrderedDict()
    for layer_key, comps in qkvo.items():
        results[layer_key] = {}
        for comp_name in ["Q", "K", "V", "O"]:
            if comp_name not in comps:
                continue
            W = comps[comp_name]
            n_out, ratio, lam_plus = compute_spectral_energy_ratio(W)
            entry_mask, shape = compute_entry_outlier_mask(W, z_thresh=4.0)
            col_idx, col_norms, col_mean, col_std = compute_axis_band_outliers(W, axis=0, z_thresh=band_z)
            row_idx, row_norms, row_mean, row_std = compute_axis_band_outliers(W, axis=1, z_thresh=band_z)
            if comp_name == "O":
                residual_dims = row_idx
                residual_axis = "row"
            else:
                residual_dims = col_idx
                residual_axis = "col"
            results[layer_key][comp_name] = {
                "outliers": n_out,
                "energy_ratio": ratio,
                "threshold": lam_plus,
                "shape": list(shape),
                "entry_outlier_coords": np.argwhere(entry_mask),
                "entry_total": int(entry_mask.size),
                "entry_outlier_count": int(entry_mask.sum()),
                "col_outlier_idx": col_idx,
                "row_outlier_idx": row_idx,
                "col_norms_mean": col_mean,
                "col_norms_std": col_std,
                "row_norms_mean": row_mean,
                "row_norms_std": row_std,
                "residual_outlier_dims": residual_dims,
                "residual_axis": residual_axis,
            }
    return results


def compute_cross_layer_alignment(analysis):
    per_component_by_layer = {c: OrderedDict() for c in ["Q", "K", "V", "O"]}
    for layer_key, comps in analysis.items():
        lk_str = str(layer_key)
        for comp in ["Q", "K", "V", "O"]:
            if comp not in comps:
                continue
            dims = comps[comp]["residual_outlier_dims"]
            per_component_by_layer[comp][lk_str] = [int(d) for d in dims]

    per_component = {c: {} for c in ["Q", "K", "V", "O"]}
    for comp in ["Q", "K", "V", "O"]:
        agg = defaultdict(list)
        for lk_str, dims in per_component_by_layer[comp].items():
            for d in dims:
                agg[d].append(lk_str)
        per_component[comp] = dict(agg)

    union = defaultdict(dict)
    for comp, dim_layers in per_component.items():
        for d, layers in dim_layers.items():
            union[d][comp] = list(layers)

    return {
        "per_component": per_component,
        "per_component_by_layer": per_component_by_layer,
        "union": dict(union),
    }


def print_alignment_summary(model_name, alignment, min_persist=3, top_k=15):
    sep = "=" * 120
    print("\n" + sep)
    print("Cross-layer residual-stream outlier alignment: {}".format(model_name))
    print(sep)
    per_comp = alignment["per_component"]
    for comp in ["Q", "K", "V", "O"]:
        dim_layers = per_comp.get(comp, {})
        pairs = sorted(
            [(d, len(ls)) for d, ls in dim_layers.items() if len(ls) >= min_persist],
            key=lambda x: -x[1],
        )
        if not pairs:
            print("  {}: no residual dims appearing as band outliers in >= {} layers".format(comp, min_persist))
            continue
        shown = pairs[:top_k]
        s = ", ".join(["{}(n={})".format(d, n) for d, n in shown])
        print("  {}: {}".format(comp, s))

    all_dims = set()
    for comp, dim_layers in per_comp.items():
        for d, layers in dim_layers.items():
            if len(layers) >= min_persist:
                all_dims.add(d)

    records = []
    for d in all_dims:
        counts = {c: len(per_comp.get(c, {}).get(d, [])) for c in ["Q", "K", "V", "O"]}
        hit = sum(1 for v in counts.values() if v > 0)
        total = sum(counts.values())
        records.append((d, hit, total, counts))
    records.sort(key=lambda r: (-r[1], -r[2]))

    print("\n  Top cross-component persistent residual dims (>= {} layers in at least one component):".format(min_persist))
    print("  {:>8} {:>10} {:>8}   {:>4} {:>4} {:>4} {:>4}".format(
        "dim", "comps_hit", "total", "Q", "K", "V", "O"))
    for d, hit, total, counts in records[:25]:
        print("  {:>8d} {:>10d} {:>8d}   {:>4d} {:>4d} {:>4d} {:>4d}".format(
            d, hit, total, counts["Q"], counts["K"], counts["V"], counts["O"]))
    print(sep + "\n")


def print_table(model_name, analysis):
    sep = "=" * 170
    mid = "-" * 170
    print("\n" + sep)
    print("Pretrained {}: MP Outlier Analysis (Q, K, V, O)".format(model_name))
    print(sep)
    header = (
        "{:>10} "
        "{:>16} {:>6} {:>8} "
        "{:>16} {:>6} {:>8} "
        "{:>16} {:>6} {:>8} "
        "{:>16} {:>6} {:>8}"
    ).format(
        "Layer",
        "Q shape", "Q out", "Q e%",
        "K shape", "K out", "K e%",
        "V shape", "V out", "V e%",
        "O shape", "O out", "O e%",
    )
    print(header)
    print(mid)

    for layer_key in analysis.keys():
        data = analysis[layer_key]
        row_parts = ["{:>10}".format(str(layer_key))]
        for comp in ["Q", "K", "V", "O"]:
            info = data.get(comp)
            if info is None:
                row_parts.append("{:>16} {:>6} {:>8}".format("-", "-", "-"))
                continue
            shape = info["shape"]
            shape_str = "{}x{}".format(shape[0], shape[1])
            row_parts.append("{:>16} {:>6d} {:>7.2f}%".format(
                shape_str, info["outliers"], info["energy_ratio"] * 100.0
            ))
        print(" ".join(row_parts))
    print(sep + "\n")


def plot_outlier_locations(model_name, analysis, out_path, max_points_per_matrix=20000):
    layer_keys = list(analysis.keys())
    n_layers = len(layer_keys)
    components = ["Q", "K", "V", "O"]

    if n_layers == 0:
        print("  No layers found for {}, skipping plot.".format(model_name))
        return

    fig_w = 4 * 2.2
    fig_h = max(2.0, n_layers * 1.1)
    fig, axes = plt.subplots(n_layers, 4, figsize=(fig_w, fig_h), squeeze=False)

    for r, layer_key in enumerate(layer_keys):
        for c, comp in enumerate(components):
            ax = axes[r][c]
            info = analysis[layer_key].get(comp)
            if info is None:
                ax.set_axis_off()
                continue
            shape = info["shape"]
            coords = info["entry_outlier_coords"]
            n_pts = coords.shape[0]
            if n_pts > max_points_per_matrix:
                idx = np.random.choice(n_pts, max_points_per_matrix, replace=False)
                coords = coords[idx]

            ax.add_patch(plt.Rectangle(
                (0, 0), shape[1], shape[0],
                facecolor="#f0f0f0", edgecolor="#b0b0b0", linewidth=0.5,
            ))
            if n_pts > 0:
                ax.scatter(coords[:, 1], coords[:, 0],
                           s=0.3, c="red", marker=".", alpha=0.6, linewidths=0)

            ax.set_xlim(0, shape[1])
            ax.set_ylim(shape[0], 0)
            ax.set_aspect("auto")
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(comp, fontsize=10)
            if c == 0:
                ax.set_ylabel(str(layer_key), fontsize=7, rotation=0, labelpad=18, va="center")

            ax.text(0.98, 0.02, "{}".format(info["entry_outlier_count"]),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5, color="#333333")

    fig.suptitle("{}: entry-level outlier locations (|w| > 4 std)".format(model_name), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved outlier-location plot to {}".format(out_path))


def plot_3d_outlier_stack(model_name, analysis, alignment, out_path,
                           min_persist_layers=3, max_highlight=20):
    layer_keys = list(analysis.keys())
    n_layers = len(layer_keys)
    components = ["Q", "K", "V", "O"]
    comp_idx = {c: i for i, c in enumerate(components)}
    if n_layers == 0:
        print("  No layers for {}, skipping 3D plot.".format(model_name))
        return

    layer_to_x = {str(lk): i for i, lk in enumerate(layer_keys)}

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fafafa")

    comp_colors = {"Q": "#4C72B0", "K": "#55A868", "V": "#C44E52", "O": "#8172B2"}

    for comp in components:
        y_val = comp_idx[comp]
        xs, zs = [], []
        for layer_key, comps in analysis.items():
            if comp not in comps:
                continue
            x = layer_to_x[str(layer_key)]
            dims = comps[comp]["residual_outlier_dims"]
            for d in dims:
                xs.append(x)
                zs.append(int(d))
        if xs:
            ax.scatter(
                xs, [y_val] * len(xs), zs,
                c=comp_colors[comp], s=9, alpha=0.32,
                edgecolors="none", depthshade=True,
            )

    per_comp = alignment["per_component"]
    persist_scores = []
    all_dims = set()
    for comp in components:
        for d in per_comp.get(comp, {}).keys():
            all_dims.add(d)
    for d in all_dims:
        counts = {c: len(per_comp.get(c, {}).get(d, [])) for c in components}
        hit_multi = sum(1 for v in counts.values() if v >= 2)
        max_single = max(counts.values())
        total = sum(counts.values())
        if max_single >= min_persist_layers or hit_multi >= 2:
            persist_scores.append((d, hit_multi, max_single, total, counts))
    persist_scores.sort(key=lambda r: (-r[1], -r[2], -r[3]))
    highlighted = persist_scores[:max_highlight]

    tab20 = plt.get_cmap("tab20")
    base_colors = [tab20(i) for i in range(20)]

    legend_handles = []
    legend_labels = []
    if highlighted:
        for i, (d, hit_multi, max_single, total, counts) in enumerate(highlighted):
            color = base_colors[i % len(base_colors)]
            drawn_any = False
            for comp in components:
                y_val = comp_idx[comp]
                layers_with_d = per_comp.get(comp, {}).get(d, [])
                if len(layers_with_d) < 2:
                    continue
                xs_line = sorted([layer_to_x[lk] for lk in layers_with_d if lk in layer_to_x])
                if len(xs_line) < 2:
                    continue
                ax.plot(
                    xs_line,
                    [y_val] * len(xs_line),
                    [d] * len(xs_line),
                    color=color, linewidth=2.1, alpha=0.92, zorder=5,
                )
                ax.scatter(
                    xs_line,
                    [y_val] * len(xs_line),
                    [d] * len(xs_line),
                    c=[color], s=30, edgecolors="black", linewidths=0.5,
                    alpha=0.95, zorder=6, depthshade=False,
                )
                drawn_any = True
            if drawn_any:
                legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.2))
                legend_labels.append(
                    "dim {} | Q={} K={} V={} O={}".format(
                        d, counts["Q"], counts["K"], counts["V"], counts["O"]
                    )
                )

    ax.set_xlabel("Layer index", fontsize=12, labelpad=12)
    ax.set_ylabel("Component", fontsize=12, labelpad=12)
    ax.set_zlabel("Residual stream dimension", fontsize=12, labelpad=12)
    ax.set_yticks(list(range(len(components))))
    ax.set_yticklabels(components, fontsize=11)

    step = max(1, n_layers // 12)
    tick_xs = list(range(0, n_layers, step))
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([str(layer_keys[i])[:10] for i in tick_xs], fontsize=8)

    ax.set_title(
        "{}\nCross-layer residual-stream outlier alignment (Q/K/V cols, O rows)".format(model_name),
        fontsize=13, pad=18,
    )
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=22, azim=-55)

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels,
            loc="upper left", bbox_to_anchor=(1.10, 1.0),
            fontsize=7.5, frameon=True,
            title="Persistent dims (layers per component)",
            title_fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 3D attention stack plot to {}".format(out_path))


def plot_persistence_heatmap(model_name, analysis, alignment, out_path, top_n=40):
    layer_keys = list(analysis.keys())
    n_layers = len(layer_keys)
    components = ["Q", "K", "V", "O"]
    if n_layers == 0:
        return

    per_comp = alignment["per_component"]
    all_dims = set()
    for comp in components:
        all_dims.update(per_comp.get(comp, {}).keys())

    scored = []
    for d in all_dims:
        counts = {c: len(per_comp.get(c, {}).get(d, [])) for c in components}
        hit = sum(1 for v in counts.values() if v > 0)
        total = sum(counts.values())
        scored.append((hit, total, d))
    scored.sort(key=lambda r: (-r[0], -r[1]))
    top_dims = [r[2] for r in scored[:top_n]]
    if not top_dims:
        print("  No persistent residual dims, skipping heatmap.")
        return

    layer_to_x = {str(lk): i for i, lk in enumerate(layer_keys)}

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, max(5.0, len(top_dims) * 0.28)),
        sharey=True,
    )

    for ci, comp in enumerate(components):
        ax = axes[ci]
        mat = np.zeros((len(top_dims), n_layers), dtype=np.float32)
        for di, d in enumerate(top_dims):
            for lk in per_comp.get(comp, {}).get(d, []):
                if lk in layer_to_x:
                    mat[di, layer_to_x[lk]] = 1.0
        ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(comp, fontsize=12)
        ax.set_xlabel("Layer index", fontsize=10)
        if ci == 0:
            ax.set_yticks(list(range(len(top_dims))))
            ax.set_yticklabels([str(d) for d in top_dims], fontsize=7)
            ax.set_ylabel("Residual dim (sorted by cross-component persistence)", fontsize=10)
        step = max(1, n_layers // 10)
        tick_xs = list(range(0, n_layers, step))
        ax.set_xticks(tick_xs)
        ax.set_xticklabels([str(i) for i in tick_xs], fontsize=8)

    fig.suptitle(
        "{}: residual-stream dim persistence across layers".format(model_name),
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  Saved persistence heatmap to {}".format(out_path))


def load_and_analyze(model_key, output_dir, band_z=4.0, min_persist_layers=3):
    cfg = MODELS[model_key]
    hf_name = cfg["hf_name"]
    arch = cfg["arch"]

    print("\n" + "#" * 80)
    print("Loading model: {} ({}) [arch={}]".format(model_key, hf_name, arch))
    print("#" * 80)

    model = None
    try:
        if arch in ("bert", "roberta"):
            from transformers import AutoModel
            model = AutoModel.from_pretrained(hf_name, torch_dtype=torch.float32)
        elif arch == "gpt2":
            from transformers import GPT2Model
            model = GPT2Model.from_pretrained(hf_name, torch_dtype=torch.float32)
        elif arch == "opt":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32)
        elif arch == "llama":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                torch_dtype=torch.float16,
                device_map={"": 0},
                trust_remote_code=True,
            )
        else:
            print("Unknown architecture: {}, skipping.".format(arch))
            return None, None, None

        extractor = EXTRACTORS[arch]
        qkvo = extractor(model)

        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not qkvo:
            print("  No Q/K/V/O weights extracted for {}, skipping.".format(model_key))
            return None, None, None

        analysis = analyze_qkvo_dict(qkvo, band_z=band_z)

        del qkvo
        gc.collect()

        print_table(model_key, analysis)

        alignment = compute_cross_layer_alignment(analysis)
        print_alignment_summary(model_key, alignment, min_persist=min_persist_layers)

        safe_name = model_key.replace("/", "_")

        plot_path = os.path.join(output_dir, "{}_outlier_locations.png".format(safe_name))
        plot_outlier_locations(model_key, analysis, plot_path)

        plot_3d_path = os.path.join(output_dir, "{}_3d_attention_stack.png".format(safe_name))
        plot_3d_outlier_stack(
            model_key, analysis, alignment, plot_3d_path,
            min_persist_layers=min_persist_layers,
        )

        heatmap_path = os.path.join(output_dir, "{}_persistence_heatmap.png".format(safe_name))
        plot_persistence_heatmap(model_key, analysis, alignment, heatmap_path)

        return model_key, analysis, alignment

    except Exception as e:
        import traceback
        print("SKIPPING {}: {}: {}".format(model_key, type(e).__name__, e))
        traceback.print_exc()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, None


def analysis_to_serializable(analysis):
    out = {}
    for layer_key, comps in analysis.items():
        out[str(layer_key)] = {}
        for comp, info in comps.items():
            out[str(layer_key)][comp] = {
                "outliers": info["outliers"],
                "energy_ratio": info["energy_ratio"],
                "threshold": info["threshold"],
                "shape": info["shape"],
                "entry_total": info["entry_total"],
                "entry_outlier_count": info["entry_outlier_count"],
                "col_outlier_idx": [int(x) for x in info.get("col_outlier_idx", [])],
                "row_outlier_idx": [int(x) for x in info.get("row_outlier_idx", [])],
                "col_norms_mean": info.get("col_norms_mean", 0.0),
                "col_norms_std": info.get("col_norms_std", 0.0),
                "row_norms_mean": info.get("row_norms_mean", 0.0),
                "row_norms_std": info.get("row_norms_std", 0.0),
                "residual_outlier_dims": [int(x) for x in info.get("residual_outlier_dims", [])],
                "residual_axis": info.get("residual_axis", ""),
            }
    return out


def alignment_to_serializable(alignment):
    out = {"per_component": {}, "per_component_by_layer": {}, "union": {}}
    for comp, dim_layers in alignment["per_component"].items():
        out["per_component"][comp] = {
            str(d): list(layers) for d, layers in dim_layers.items()
        }
    for comp, layer_dims in alignment["per_component_by_layer"].items():
        out["per_component_by_layer"][comp] = {
            str(lk): [int(x) for x in dims] for lk, dims in layer_dims.items()
        }
    for d, comp_map in alignment["union"].items():
        out["union"][str(d)] = {
            c: list(layers) for c, layers in comp_map.items()
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="./output/mp_analysis")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--band_z", type=float, default=4.0)
    parser.add_argument("--min_persist_layers", type=int, default=3)
    args = parser.parse_args()

    if args.list:
        for k, v in MODELS.items():
            print("{:25s}  ->  {}  [{}]".format(k, v["hf_name"], v["arch"]))
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.models is not None:
        model_keys = [m for m in args.models if m in MODELS]
        missing = [m for m in args.models if m not in MODELS]
        if missing:
            print("WARNING: unknown model keys ignored: {}".format(missing))
    else:
        model_keys = list(MODELS.keys())

    all_results = {}
    all_alignments = {}

    for mk in model_keys:
        name, analysis, alignment = load_and_analyze(
            mk, args.output_dir,
            band_z=args.band_z,
            min_persist_layers=args.min_persist_layers,
        )
        if name is None or analysis is None:
            continue
        serializable = analysis_to_serializable(analysis)
        align_ser = alignment_to_serializable(alignment) if alignment is not None else {}
        all_results[name] = serializable
        all_alignments[name] = align_ser

        safe_name = name.replace("/", "_")

        per_model_path = os.path.join(args.output_dir, "{}_mp_analysis.json".format(safe_name))
        with open(per_model_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print("Saved {} analysis to {}".format(name, per_model_path))

        per_model_align_path = os.path.join(args.output_dir, "{}_cross_layer_alignment.json".format(safe_name))
        with open(per_model_align_path, "w") as f:
            json.dump(align_ser, f, indent=2)
        print("Saved {} alignment to {}".format(name, per_model_align_path))

    combined_path = os.path.join(args.output_dir, "all_models_mp_analysis.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll MP results saved to {}".format(combined_path))

    combined_align_path = os.path.join(args.output_dir, "all_models_cross_layer_alignment.json")
    with open(combined_align_path, "w") as f:
        json.dump(all_alignments, f, indent=2)
    print("All alignment results saved to {}".format(combined_align_path))

    print("\n" + "=" * 100)
    print("SUMMARY: Average MP outliers per layer, per component")
    print("=" * 100)
    for mk in all_results:
        data = all_results[mk]
        layers = list(data.keys())
        if not layers:
            continue
        n_layers = len(layers)

        def avg(comp):
            vals = [data[l].get(comp, {}).get("outliers", 0) for l in layers]
            return sum(vals) / n_layers if n_layers else 0

        def total(comp):
            return sum(data[l].get(comp, {}).get("outliers", 0) for l in layers)

        print("{:>20s} | layers={:>3d} | Q_avg={:.1f} K_avg={:.1f} V_avg={:.1f} O_avg={:.1f} | Q_tot={} K_tot={} V_tot={} O_tot={}".format(
            mk, n_layers, avg("Q"), avg("K"), avg("V"), avg("O"),
            total("Q"), total("K"), total("V"), total("O")
        ))
    print("=" * 100)

    print("\n" + "=" * 100)
    print("SUMMARY: Cross-layer persistent residual-stream dimensions (>= {} layers)".format(args.min_persist_layers))
    print("=" * 100)
    for mk in all_alignments:
        align = all_alignments[mk]
        per_comp = align.get("per_component", {})
        totals = {}
        for comp in ["Q", "K", "V", "O"]:
            dim_layers = per_comp.get(comp, {})
            n_persistent = sum(1 for d, ls in dim_layers.items() if len(ls) >= args.min_persist_layers)
            totals[comp] = n_persistent
        all_dims_union = set()
        for comp in ["Q", "K", "V", "O"]:
            for d, ls in per_comp.get(comp, {}).items():
                if len(ls) >= args.min_persist_layers:
                    all_dims_union.add(d)
        print("{:>20s} | persistent: Q={:>3d} K={:>3d} V={:>3d} O={:>3d} | unique dims (union)={}".format(
            mk, totals["Q"], totals["K"], totals["V"], totals["O"], len(all_dims_union)
        ))
    print("=" * 100)


if __name__ == "__main__":
    main()
