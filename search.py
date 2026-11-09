"""Validation-based configuration search (no test-set tuning).

For every configuration: runs `--models` x `--rates` x `--seeds`, saves one JSON per run.
Then writes <out>/search_summary.csv. The SELECTED configuration is the one with the highest mean
VALIDATION accuracy over all its rates and seeds (rmm only). Test numbers are recorded for every
configuration but are never used to choose. Re-running skips runs whose JSON already exists.

Example:
  python search.py --out results_search
  python search.py --out results_search --configs lstm-adam --models baseline     # control for the chosen config
"""
import argparse, csv, glob, json, os, itertools, numpy as np
from train import run, parser, config_tag

CONFIGS = {                     # name: (av_enc, opt, bert_lr or None = optimiser default, other lr)
    "mean-paper": ("mean", "paper", None, 1e-4),
    "mean-adam": ("mean", "adam", None, 1e-4),
    "lstm-paper": ("lstm", "paper", None, 1e-4),
    "lstm-adam": ("lstm", "adam", None, 1e-4),            # BERT 2e-5, others 1e-4
    # lower learning rates, added on 30 Oct 2026 before any of them was run:
    "lstm-adam-blr1e-5": ("lstm", "adam", 1e-5, 1e-4),
    "lstm-adam-blr1e-5-lr5e-5": ("lstm", "adam", 1e-5, 5e-5),
}

p = argparse.ArgumentParser()
p.add_argument("--configs", nargs="+", default=list(CONFIGS))
p.add_argument("--models", nargs="+", default=["rmm"])
p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--unc", default="none")
p.add_argument("--out", default="results_search")
a = p.parse_args()

for name, mdl, r, s in itertools.product(a.configs, a.models, a.rates, a.seeds):
    av_enc, opt, blr, lr = CONFIGS[name]
    extra = (["--bert-lr", str(blr)] if blr is not None else []) + ["--lr", str(lr)]
    args = parser().parse_args(["--model", mdl, "--av-enc", av_enc, "--opt", opt, "--rate", str(r), "--seed", str(s),
                                "--unc", a.unc if mdl == "rmm" else "none", "--out", a.out] + extra)
    if os.path.exists(os.path.join(a.out, f"{config_tag(args)}_r{r}_s{s}.json")):
        continue
    run(args)

rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "*.json"))]
cfgs = sorted({r["config"] for r in rows})
val_mean = {c: np.mean([r["val_acc"] for r in rows if r["config"] == c]) for c in cfgs}
rmm_cfgs = [c for c in cfgs if c.startswith("rmm_")]
selected = max(rmm_cfgs, key=lambda c: val_mean[c]) if rmm_cfgs else None
cols = ["config", "rate", "n", "val_acc", "test_acc", "test_acc_std", "test_waf", "test_acc_non0", "test_acc_non0_std",
        "test_waf_non0", "params", "gflops", "latency_ms", "train_s", "best_epoch", "selected_by_val"]
with open(os.path.join(a.out, "search_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for c in cfgs:
        for rate in sorted({r["rate"] for r in rows if r["config"] == c}):
            g = [r for r in rows if r["config"] == c and r["rate"] == rate]
            m = lambda k: np.mean([r[k] for r in g]); sd = lambda k: np.std([r[k] for r in g])
            w.writerow([c, rate, len(g), round(m("val_acc"), 2), round(m("test_acc"), 2), round(sd("test_acc"), 2),
                        round(m("test_waf"), 2), round(m("test_acc_non0"), 2), round(sd("test_acc_non0"), 2),
                        round(m("test_waf_non0"), 2), g[0]["params"], round(g[0]["gflops"], 4), round(m("latency_ms"), 2),
                        round(m("train_seconds"), 1), round(m("best_epoch"), 1),
                        "yes" if c == selected or (selected and c == selected.replace("rmm_none", "baseline_none")) else ""])
print(open(os.path.join(a.out, "search_summary.csv")).read())
print("Selected by mean validation accuracy:", selected, {c: round(v, 2) for c, v in val_mean.items()})
