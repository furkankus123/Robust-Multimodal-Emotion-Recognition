"""Runs {rmm, baseline} x rates x seeds, then writes <out>/summary.csv (mean +- std over seeds)."""
import argparse, glob, json, itertools, csv, numpy as np
from train import run

p = argparse.ArgumentParser()
p.add_argument("--data", default="../mosi_raw.pkl")
p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5])
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--models", nargs="+", default=["rmm", "baseline"])
p.add_argument("--gate", default="keep")
p.add_argument("--mask", default="element")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--bert", default=None)
p.add_argument("--bert-pool", dest="bert_pool", default="cls")
p.add_argument("--texts", default=None)
p.add_argument("--bert-lr", dest="bert_lr", type=float, default=5e-5)
p.add_argument("--out", default="results")
a = p.parse_args()
for mdl, r, s in itertools.product(a.models, a.rates, a.seeds):
    run(argparse.Namespace(data=a.data, model=mdl, gate=a.gate, mask=a.mask, rate=r, seed=s, epochs=a.epochs,
                           patience=a.patience, warmup=a.warmup, bs=32, lr=1e-4, bert_lr=a.bert_lr, wd=0.01, hidden=128, latent=64,
                           out=a.out, bert=a.bert, bert_pool=a.bert_pool, texts=a.texts))
rows = [json.load(open(f)) for f in glob.glob(f"{a.out}/*.json")]
key = lambda r: (r["model"], r["gate"], r.get("text", "glove"), r.get("mask", "modality"), r["rate"])
with open(f"{a.out}/summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "gate", "text", "mask", "rate", "n_seeds", "acc_mean", "acc_std", "waf_mean", "waf_std",
                "params", "gflops", "latency_ms", "train_s", "best_epoch_mean"])
    for k in sorted({key(r) for r in rows}):
        g = [r for r in rows if key(r) == k]
        A = [r["test_acc"] for r in g]; W = [r["test_waf"] for r in g]
        w.writerow([*k, len(g), np.mean(A), np.std(A), np.mean(W), np.std(W), g[0]["params"], g[0]["gflops"],
                    np.mean([r["latency_ms"] for r in g]), np.mean([r["train_seconds"] for r in g]),
                    np.mean([r.get("best_epoch", np.nan) for r in g])])
print(open(f"{a.out}/summary.csv").read())
