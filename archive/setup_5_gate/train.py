import argparse, json, math, os, time, random
import numpy as np, torch
from torch.utils.data import DataLoader
from data import load_mosi, MosiSet, MODS
from model import RMMTCVAE, ConcatBaseline, loss_fn
from metrics import acc_waf
from resources import measure


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(d, dev):
    return {m: v.to(dev) for m, v in d.items()} if d else None


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); P, Y = [], []
    for x, xc, y, av in dl:
        P.append(model(to_dev(x, dev), av.to(dev))[0].argmax(1).cpu()); Y.append(y)
    return acc_waf(torch.cat(P).numpy(), torch.cat(Y).numpy())


def run(a):
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, stats = load_mosi(a.data, a.bert, a.bert_pool, a.texts)
    dims = {m: d["train"][0][m].shape[1] for m in MODS}
    if a.texts: dims["text"] = 768
    elem = a.mask == "element"
    tr = MosiSet(*d["train"], stats, a.rate, a.seed, fixed=False, mask=a.mask)
    va = MosiSet(*d["valid"], stats, a.rate, 1000 + a.seed, mask=a.mask)   # fixed masks; model selection uses VALID only
    te = MosiSet(*d["test"], stats, a.rate, 2000 + a.seed, mask=a.mask)    # fixed masks; test touched once at the end
    dl_tr = DataLoader(tr, a.bs, shuffle=True)
    dl_va, dl_te = DataLoader(va, 128), DataLoader(te, 128)
    ft = bool(a.texts)
    model = (ConcatBaseline(dims, a.hidden, finetune=ft, elem=elem) if a.model == "baseline"
             else RMMTCVAE(dims, a.hidden, a.latent, gate=a.gate, finetune=ft, elem=elem, unc=a.unc)).to(dev)
    bp = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    op = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    groups = [{"params": op, "lr": a.lr}] + ([{"params": bp, "lr": a.bert_lr}] if bp else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    total = a.epochs * len(dl_tr); warm = a.warmup                                        # warm-up (fixed steps) + cosine annealing (paper)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(a.epochs):
        model.train(); tr.resample()
        for x, xc, y, av in dl_tr:
            out = model(to_dev(x, dev), av.to(dev), to_dev(xc, dev))
            loss = loss_fn(out, y.to(dev))[0]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        v_acc, v_waf = evaluate(model, dl_va, dev)
        if v_acc > best:
            best, bad, best_ep = v_acc, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience: break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    t_acc, t_waf = evaluate(model, dl_te, dev)
    res = measure(model, dims, dev)
    res.update(dict(model=a.model, gate=a.gate, unc=a.unc if a.model == "rmm" else "none", rate=a.rate, seed=a.seed, mask=a.mask, epochs_run=ep + 1, best_epoch=best_ep,
                    val_acc=best, test_acc=t_acc, test_waf=t_waf, train_seconds=train_s,
                    peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None),
                    device=str(dev), lr=a.lr, bert_lr=a.bert_lr, bs=a.bs, hidden=a.hidden, latent=a.latent,
                    text="bert-ft" if a.texts else (("bert-" + a.bert_pool) if a.bert else "glove")))
    os.makedirs(a.out, exist_ok=True)
    fn = f"{a.model}_{a.gate}_{res['text']}_{a.mask}_{res['unc']}_r{a.rate}_s{a.seed}.json"
    json.dump(res, open(os.path.join(a.out, fn), "w"), indent=1)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../mosi_raw.pkl")
    p.add_argument("--model", choices=["rmm", "baseline"], default="rmm")
    p.add_argument("--gate", choices=["keep", "zero"], default="keep")
    p.add_argument("--unc", choices=["none", "gate", "shuffle", "fixedinv"], default="none")
    p.add_argument("--mask", choices=["modality", "element"], default="element")
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup", type=int, default=100, help="warm-up steps (run A used 5%% of 200 epochs = 400 steps)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bert-lr", dest="bert_lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--bert", default=None)
    p.add_argument("--bert-pool", dest="bert_pool", choices=["cls", "mean"], default="cls")
    p.add_argument("--texts", default=None, help="texts.pkl -> BERT fine-tuned inside the model")
    p.add_argument("--out", default="results")
    run(p.parse_args())
