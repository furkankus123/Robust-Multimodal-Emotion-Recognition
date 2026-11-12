import argparse, json, math, os, time, random
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import load_mosi, MosiSet, MODS
from model import RMMTCVAE, ConcatBaseline, loss_fn
from model_paper import PaperRMM, paper_loss
from metrics import acc_waf
from resources import measure

OPT_DEFAULT_BERT_LR = {"paper": 5e-5, "adam": 2e-5}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_dev(d, dev):
    return {m: v.to(dev) for m, v in d.items()} if d else None


@torch.no_grad()
def evaluate(model, dl, dev):
    """Acc/WAF on all segments (score 0 counted as negative) and on non-zero segments only (paper convention)."""
    model.eval(); P, Y = [], []
    for x, xc, y, av, sc in dl:
        P.append(model(to_dev(x, dev), av.to(dev))[0][:, :2].argmax(1).cpu()); Y.append(y)
    p, y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    nz = dl.dataset.score != 0
    return acc_waf(p, y) + acc_waf(p[nz], y[nz])


def config_tag(a):
    tag = f"{a.model}_{a.unc}_{a.mask}_{a.av_enc}_{a.opt}"
    if a.bert_lr is not None: tag += f"_blr{a.bert_lr:g}"      # non-default learning rates get their own name
    if a.lr != 1e-4: tag += f"_lr{a.lr:g}"
    if a.hide_unk: tag += "_hideunk"
    if a.clean_view: tag += "_cleanview"
    if a.consist > 0: tag += f"_kl{a.consist:g}"
    if a.aux_reg > 0: tag += f"_reg{a.aux_reg:g}"
    return tag


def run(a):
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, stats = load_mosi(a.data, a.bert, a.bert_pool, a.texts)
    dims = {m: d["train"][0][m].shape[1] for m in MODS}
    if a.texts: dims["text"] = 768
    av_seq = a.av_enc == "lstm"
    raw_dims = None
    if av_seq:
        raw_dims = {m: d["train"][2][m].shape[-1] for m in ("audio", "visual")}
        dims["audio"] = dims["visual"] = a.lstm_hidden
    elem = a.mask == "element"
    bert_lr = a.bert_lr if a.bert_lr is not None else OPT_DEFAULT_BERT_LR[a.opt]
    tr = MosiSet(*d["train"], stats, a.rate, a.seed, fixed=False, mask=a.mask, av_seq=av_seq)
    va = MosiSet(*d["valid"], stats, a.rate, 1000 + a.seed, mask=a.mask, av_seq=av_seq)   # selection uses VALID only
    te = MosiSet(*d["test"], stats, a.rate, 2000 + a.seed, mask=a.mask, av_seq=av_seq)    # test touched once
    dl_tr = DataLoader(tr, a.bs, shuffle=True)
    dl_va, dl_te = DataLoader(va, 128), DataLoader(te, 128)
    ft = bool(a.texts)
    kw = dict(finetune=ft, elem=elem, raw_dims=raw_dims, lstm_hidden=a.lstm_hidden, hide_unk=a.hide_unk,
              aux_reg=a.aux_reg > 0)
    assert not ((a.clean_view or a.consist > 0) and not elem), "--clean-view/--consist need --mask element"
    assert not (a.model == "paper" and a.unc != "none"), "--unc is only implemented for --model rmm"
    if a.model == "baseline":
        model = ConcatBaseline(dims, a.hidden, **kw)
    elif a.model == "paper":
        model = PaperRMM(dims, a.hidden, a.latent, **kw)
    else:
        model = RMMTCVAE(dims, a.hidden, a.latent, gate=a.gate, unc=a.unc, **kw)
    model = model.to(dev)
    lossf = paper_loss if a.model == "paper" else loss_fn
    bp = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    op = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    groups = [{"params": op, "lr": a.lr}] + ([{"params": bp, "lr": bert_lr}] if bp else [])
    total = a.epochs * len(dl_tr)
    if a.opt == "paper":                      # AdamW + warm-up + cosine annealing (paper, Table 9 text)
        opt = torch.optim.AdamW(groups, weight_decay=a.wd)
        warm = a.warmup
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    else:                                     # setup-3 optimiser: Adam, constant learning rates
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
    best, best_state, bad, t0, best_ep = -1, None, 0, time.time(), 0
    for ep in range(a.epochs):
        model.train(); tr.resample()
        for x, xc, y, av, sc in dl_tr:
            y, sc = y.to(dev), sc.to(dev)
            out = model(to_dev(x, dev), av.to(dev), to_dev(xc, dev))
            loss = lossf(out, y, sc, a.aux_reg)[0]
            if a.clean_view or a.consist > 0:        # second view: the same segment without masking (training only)
                out_c = model(to_dev(xc, dev), torch.ones_like(av).to(dev))
                if a.clean_view:
                    loss = loss + lossf(out_c, y, sc, a.aux_reg)[0]
                if a.consist > 0:                    # masked prediction should match the clean one
                    loss = loss + a.consist * F.kl_div(F.log_softmax(out[0][:, :2], 1),
                                                       F.softmax(out_c[0][:, :2].detach(), 1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        v = evaluate(model, dl_va, dev)
        # early stopping and checkpoint selection use validation accuracy on ALL segments (score 0 counted as
        # negative); test accuracy is reported both on all segments and with score-0 segments excluded
        if v[0] > best:
            best, bad, best_ep, best_v = v[0], 0, ep + 1, v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience: break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    t = evaluate(model, dl_te, dev)
    res = measure(model, dims, dev)
    res.update(dict(model=a.model, gate=a.gate, unc=a.unc if a.model == "rmm" else "none", rate=a.rate, seed=a.seed,
                    mask=a.mask, av_enc=a.av_enc, opt=a.opt, epochs_run=ep + 1, best_epoch=best_ep,
                    val_acc=best_v[0], val_waf=best_v[1], val_acc_non0=best_v[2], val_waf_non0=best_v[3],
                    test_acc=t[0], test_waf=t[1], test_acc_non0=t[2], test_waf_non0=t[3], train_seconds=train_s,
                    peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None),
                    device=str(dev), lr=a.lr, bert_lr=bert_lr, bs=a.bs, hidden=a.hidden, latent=a.latent,
                    lstm_hidden=a.lstm_hidden, patience=a.patience, hide_unk=a.hide_unk, clean_view=a.clean_view,
                    consist=a.consist, aux_reg=a.aux_reg,
                    text="bert-ft" if a.texts else (("bert-" + a.bert_pool) if a.bert else "glove")))
    res["config"] = config_tag(a)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{res['config']}_r{a.rate}_s{a.seed}.json"), "w"), indent=1)
    print(json.dumps(res))
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mosi_raw.pkl")
    p.add_argument("--model", choices=["rmm", "baseline", "paper"], default="rmm",
                   help="rmm = pseudocode version, paper = paper-faithful (model_paper.py), baseline = plain fusion")
    p.add_argument("--gate", choices=["keep", "zero"], default="keep")
    p.add_argument("--unc", choices=["none", "gate", "shuffle", "fixedinv"], default="none")
    p.add_argument("--mask", choices=["modality", "element"], default="element")
    p.add_argument("--av-enc", dest="av_enc", choices=["mean", "lstm"], default="mean")
    p.add_argument("--opt", choices=["paper", "adam"], default="paper")
    p.add_argument("--rate", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bert-lr", dest="bert_lr", type=float, default=None, help="default: 5e-5 (paper) / 2e-5 (adam)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--lstm-hidden", dest="lstm_hidden", type=int, default=64)
    p.add_argument("--hide-unk", dest="hide_unk", action="store_true", help="hide [UNK] tokens from BERT attention")
    p.add_argument("--clean-view", dest="clean_view", action="store_true", help="also train on the unmasked segment")
    p.add_argument("--consist", type=float, default=0.0, help="weight of KL(clean || masked) consistency")
    p.add_argument("--aux-reg", dest="aux_reg", type=float, default=0.0, help="weight of auxiliary L1 regression")
    p.add_argument("--bert", default=None)
    p.add_argument("--bert-pool", dest="bert_pool", choices=["cls", "mean"], default="cls")
    p.add_argument("--texts", default="data/texts.pkl")
    p.add_argument("--out", default="results")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
