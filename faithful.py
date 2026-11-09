"""Paper-faithful model vs pseudocode version vs plain-fusion control over the paper's missing-rate grid.

Defined 7 Nov 2026, before any run of the paper-faithful model.
  Setting: fixed in advance to the one chosen earlier by validation (lstm-adam-blr1e-5: LSTM audio/visual encoders,
           Adam, BERT lr 1e-5, other layers 1e-4). No new search; nothing is chosen on test.
  Models:  paper    = model_paper.PaperRMM   (paper-faithful)
           rmm      = model.RMMTCVAE         (authors' pseudocode)
           baseline = model.ConcatBaseline   (plain fusion, same BERT and LSTMs)
  Rates:   0.0-0.5 in steps of 0.1 (paper Table 1). Seeds 0, 1, 2. Paper masking protocol (element).
Runs that already exist in results_search are copied instead of re-run (same code; runs are deterministic).
Writes <out>/faithful_summary.csv, including each model's drop from r = 0.0. Re-running skips finished runs.

  python faithful.py                       # everything (paper model first)
  python faithful.py --models paper --rates 0.0 0.5     # the two headline rates only
"""
import argparse, csv, glob, json, os, shutil, itertools, numpy as np
from train import run, parser, config_tag

SETTING = ["--av-enc", "lstm", "--opt", "adam", "--bert-lr", "1e-05", "--lr", "0.0001", "--mask", "element"]
PUBLISHED = {0.0: 84.7, 0.1: 83.2, 0.2: 82.7, 0.3: 80.6, 0.4: 80.1, 0.5: 78.1}   # paper Table 1, MOSI, RMM-TCVAE Acc

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["paper", "rmm", "baseline"])
p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--out", default="results_faithful")
p.add_argument("--reuse-dir", default="results_search")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

for mdl, r, s in itertools.product(a.models, a.rates, a.seeds):
    args = parser().parse_args(["--model", mdl, "--rate", str(r), "--seed", str(s), "--out", a.out] + SETTING)
    fn = f"{config_tag(args)}_r{r}_s{s}.json"
    if os.path.exists(os.path.join(a.out, fn)):
        continue
    if os.path.exists(os.path.join(a.reuse_dir, fn)):
        shutil.copy(os.path.join(a.reuse_dir, fn), os.path.join(a.out, fn))
        continue
    run(args)

rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "*.json"))]
cols = ["model", "rate", "n", "val_acc", "test_acc_non0", "test_acc_non0_std", "test_waf_non0", "drop_from_r0",
        "published_rmm_tcvae_acc", "test_acc", "test_acc_std", "params", "gflops", "latency_ms", "train_s",
        "best_epoch", "peak_gpu_mb"]
with open(os.path.join(a.out, "faithful_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for mdl in ["paper", "rmm", "baseline"]:
        g_all = [r for r in rows if r["model"] == mdl]
        clean = [r["test_acc_non0"] for r in g_all if r["rate"] == 0.0]
        for rate in sorted({r["rate"] for r in g_all}):
            g = [r for r in g_all if r["rate"] == rate]
            m = lambda k: round(float(np.mean([r[k] for r in g])), 2)
            sd = lambda k: round(float(np.std([r[k] for r in g])), 2)
            drop = round(m("test_acc_non0") - float(np.mean(clean)), 2) if clean else ""
            w.writerow([mdl, rate, len(g), m("val_acc"), m("test_acc_non0"), sd("test_acc_non0"), m("test_waf_non0"),
                        drop, PUBLISHED.get(rate, ""), m("test_acc"), sd("test_acc"), g[0]["params"],
                        round(g[0]["gflops"], 4), m("latency_ms"), m("train_seconds"), m("best_epoch"),
                        m("peak_gpu_mb")])
print(open(os.path.join(a.out, "faithful_summary.csv")).read())
