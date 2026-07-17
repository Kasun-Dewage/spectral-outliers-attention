import math
import os
import json
import argparse
import gc

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "llama1-7b":   "huggyllama/llama-7b",
    "llama2-7b":   "meta-llama/Llama-2-7b-hf",
    "llama3-8b":   "meta-llama/Meta-Llama-3-8B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "mistral-7b":  "mistralai/Mistral-7B-v0.1",
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B",
}

ATTN_SUF = {".q_proj": "Q", ".k_proj": "K", ".v_proj": "V", ".o_proj": "O"}


def mp_mask(S, shape):
    m, n = shape
    gamma = max(m, n) / min(m, n)
    sq = S ** 2
    sigma_sq = float(sq.median()) / (1.0 + gamma)
    lam = sigma_sq * (1.0 + math.sqrt(gamma)) ** 2
    return sq > lam, lam


def ablate_spectral(W, mode, fraction, seed, dev):
    orig = W.dtype
    src_dev = W.device
    Wf = W.detach().float().to(dev)
    U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
    mask, _ = mp_mask(S, Wf.shape)
    out_i = torch.where(mask)[0].cpu().numpy()
    bulk_i = torch.where(~mask)[0].cpu().numpy()
    S_cpu = S.cpu().numpy()
    k_tot = len(out_i)
    k = int(math.ceil(k_tot * fraction))
    if k == 0:
        del U, S, Vt, Wf
        torch.cuda.empty_cache()
        return W, 0
    rng = np.random.RandomState(seed)
    if mode == "outliers":
        zero_i = out_i[np.argsort(-S_cpu[out_i])][:k]
    elif mode == "bulk_top":
        zero_i = bulk_i[np.argsort(-S_cpu[bulk_i])][:min(k, len(bulk_i))]
    elif mode == "random_bulk":
        zero_i = rng.choice(bulk_i, size=min(k, len(bulk_i)), replace=False)
    elif mode == "bottom":
        zero_i = bulk_i[np.argsort(S_cpu[bulk_i])][:min(k, len(bulk_i))]
    else:
        del U, S, Vt, Wf
        torch.cuda.empty_cache()
        return W, 0
    S_new = S.clone()
    S_new[torch.from_numpy(zero_i).to(dev)] = 0.0
    Wn = (U * S_new.unsqueeze(0)) @ Vt
    out = Wn.to(orig).to(src_dev)
    del U, S, Vt, S_new, Wn, Wf
    torch.cuda.empty_cache()
    return out, int(len(zero_i))


def ablate_entries(W, mode, z, seed):
    orig = W.dtype
    dev = W.device
    Wf = W.detach().float()
    std = float(Wf.std())
    if std < 1e-12:
        return W, 0
    om = Wf.abs() > z * std
    k = int(om.sum().item())
    if k == 0:
        return W, 0
    Wn = Wf.clone()
    if mode == "outliers":
        Wn[om] = 0.0
    elif mode == "random":
        flat = Wn.view(-1)
        om_flat = om.view(-1).cpu().numpy()
        cand = np.where(~om_flat)[0]
        rng = np.random.RandomState(seed)
        pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
        flat[torch.from_numpy(pick).to(dev)] = 0.0
        Wn = flat.view(Wf.shape)
    else:
        return W, 0
    return Wn.to(orig).to(dev), k


def ablate_band(W, comp, mode, z, seed):
    orig = W.dtype
    dev = W.device
    Wf = W.detach().float()
    ax = 1 if comp == "O" else 0
    norms = Wf.norm(dim=ax)
    mu = float(norms.mean())
    sd = float(norms.std())
    if sd < 1e-12:
        return W, 0
    om = norms > mu + z * sd
    k = int(om.sum().item())
    if k == 0:
        return W, 0
    if mode == "outliers":
        idx = torch.where(om)[0]
    elif mode == "random":
        cand = torch.where(~om)[0].cpu().numpy()
        rng = np.random.RandomState(seed)
        pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
        idx = torch.from_numpy(pick).to(Wf.device)
    else:
        return W, 0
    Wn = Wf.clone()
    if comp == "O":
        Wn[idx, :] = 0.0
    else:
        Wn[:, idx] = 0.0
    return Wn.to(orig).to(dev), k


def iter_attn(model):
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        for suf, comp in ATTN_SUF.items():
            if name.endswith(suf):
                yield name, mod, comp
                break


def apply_strategy(model, strategy, target, z, fraction, seed, dev):
    if strategy == "baseline":
        return 0
    total = 0
    for name, mod, comp in iter_attn(model):
        if comp not in target:
            continue
        W = mod.weight.data
        if strategy.startswith("spectral_"):
            mode = strategy[len("spectral_"):]
            new_W, k = ablate_spectral(W, mode, fraction, seed, dev)
        elif strategy.startswith("entry_"):
            mode = strategy[len("entry_"):]
            new_W, k = ablate_entries(W, mode, z, seed)
        elif strategy.startswith("band_"):
            mode = strategy[len("band_"):]
            new_W, k = ablate_band(W, comp, mode, z, seed)
        else:
            continue
        mod.weight.data.copy_(new_W)
        total += k
        del new_W
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total


def run_eval(model, tokenizer, tasks, batch_size):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    r = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, batch_size=batch_size)
    return r["results"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--strategy", type=str, required=True)
    p.add_argument("--components", nargs="+", default=["Q", "K", "V", "O"])
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "piqa", "mmlu"])
    p.add_argument("--z", type=float, default=4.0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="./output/ablation")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hf = MODELS.get(args.model, args.model)

    model = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    model.eval()

    n = apply_strategy(
        model, args.strategy, set(args.components),
        args.z, args.fraction, args.seed, "cuda"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res = run_eval(model, tokenizer, args.tasks, args.batch_size)

    ctag = "".join(sorted(args.components))
    fname = "{}__{}__{}__f{:.2f}__s{}.json".format(
        args.model, args.strategy, ctag, args.fraction, args.seed
    )
    path = os.path.join(args.output_dir, fname)
    with open(path, "w") as f:
        json.dump({
            "model": args.model,
            "strategy": args.strategy,
            "components": args.components,
            "z": args.z,
            "fraction": args.fraction,
            "seed": args.seed,
            "n_zeroed": n,
            "results": res,
        }, f, indent=2, default=str)
    print("SAVED", path, "n_zeroed=", n)


if __name__ == "__main__":
    main()
