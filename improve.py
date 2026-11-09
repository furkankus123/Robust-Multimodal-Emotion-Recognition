"""Robustness experiment at r = 0.5 on the selected configuration (LSTM, Adam, BERT 1e-5).

Arms (all defined on 30 Oct 2026 before any of them was run):
  base        the selected RMM-TCVAE (copied from results_search; runs are deterministic)
  hideunk     missing words hidden from BERT attention instead of shown as [UNK]
  cleanview   also trained on the unmasked version of each segment (CE on both views)   <- control for 'consist'
  consist     cleanview + KL(clean prediction || masked prediction), weight 1
  auxreg      auxiliary L1 regression on the sentiment score, weight 0.5
Each arm: rates 0.0 and 0.5 x seeds 0,1,2. Test is evaluated once per run; nothing is chosen on test.
Writes <out>/improve_summary.csv. Re-running skips finished runs.
"""
import argparse, csv, glob, json, os, shutil, itertools, numpy as np
from train import run, parser, config_tag

BASE = ["--av-enc", "lstm", "--opt", "adam", "--bert-lr", "1e-05", "--lr", "0.0001"]
ARMS = {
    "base": [],
    "hideunk": ["--hide-unk"],
    "cleanview": ["--clean-view"],
    "consist": ["--clean-view", "--consist", "1.0"],
    "auxreg": ["--aux-reg", "0.5"],
    # added 30 Oct 2026 after the five arms above: their combination, reported only if validation favours it
    "combo": ["--hide-unk", "--clean-view", "--consist", "1.0", "--aux-reg", "0.5"],
}
p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", default=list(ARMS))
p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--model", default="rmm")
p.add_argument("--out", default="results_improve")
p.add_argument("--base-dir", default="results_search")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
for arm, r, s in itertools.product(a.arms, a.rates, a.seeds):
    args = parser().parse_args(["--model", a.model, "--rate", str(r), "--seed", str(s), "--out", a.out] + BASE + ARMS[arm])
    fn = f"{config_tag(args)}_r{r}_s{s}.json"
    if os.path.exists(os.path.join(a.out, fn)):
        continue
    if os.path.exists(os.path.join(a.base_dir, fn)):          # identical run already done in the search
        shutil.copy(os.path.join(a.base_dir, fn), os.path.join(a.out, fn))
        continue
    run(args)

rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "*.json"))]
cols = ["config", "rate", "n", "val_acc", "test_acc_non0", "test_acc_non0_std", "test_waf_non0", "test_acc", "test_acc_std",
        "latency_ms", "train_s", "best_epoch"]
with open(os.path.join(a.out, "improve_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for c in sorted({r["config"] for r in rows}):
        for rate in sorted({r["rate"] for r in rows if r["config"] == c}):
            g = [r for r in rows if r["config"] == c and r["rate"] == rate]
            m = lambda k: round(float(np.mean([r[k] for r in g])), 2); sd = lambda k: round(float(np.std([r[k] for r in g])), 2)
            w.writerow([c, rate, len(g), m("val_acc"), m("test_acc_non0"), sd("test_acc_non0"), m("test_waf_non0"),
                        m("test_acc"), sd("test_acc"), m("latency_ms"), m("train_seconds"), m("best_epoch")])
print(open(os.path.join(a.out, "improve_summary.csv")).read())
