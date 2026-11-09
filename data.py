"""MOSI loader (MultiBench mosi_raw.pkl + texts.pkl) with two missing-data protocols.

  mask="modality": each whole modality is dropped per sample with probability r
  mask="element" : PAPER PROTOCOL: a fraction r of the elements of every modality sequence is masked, in
                   train/val/test alike. Text tokens -> [UNK]; audio/visual frames -> zeros. Needs --texts.

av_seq=True (paper: single-layer LSTM for audio/visual): audio/visual are returned as standardised 50-step frame
sequences (masked and padded frames = 0) instead of one mean-pooled vector.
Scores are kept so that accuracy can also be reported with score-0 segments excluded ("non0", as in the paper).
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

MODS = ["text", "audio", "visual"]
MAXLEN = 64
KEYMAP = {"text": "text", "audio": "audio", "visual": "vision"}
UNK, CLS, SEP = 100, 101, 102


def _clean(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _valid(d):
    s = sum(np.abs(_clean(d[k])).sum(-1) for k in ("text", "audio", "vision"))
    return s > 0


def _pool(x, valid):
    cnt = np.maximum(valid.sum(1, keepdims=True), 1)
    return (x * valid[..., None]).sum(1) / cnt


def tokenize(texts_pkl, ids):
    from transformers import AutoTokenizer
    with open(texts_pkl, "rb") as f:
        texts = pickle.load(f)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    e = tok([texts[i] for i in ids], padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="np")
    return np.stack([e["input_ids"], e["attention_mask"]], 1).astype(np.int64)


def load_mosi(path, bert=None, bert_pool="cls", texts=None):
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if bert:
        with open(bert, "rb") as f:
            bf = pickle.load(f)[bert_pool]
    out, stats = {}, {}
    for split in ["train", "valid", "test"]:
        d = raw[split]
        valid = _valid(d)
        feats = {m: _pool(_clean(d[KEYMAP[m]]), valid) for m in MODS}
        extra = {"valid": valid,
                 "audio": _clean(d["audio"]).astype(np.float32),
                 "visual": _clean(d["vision"]).astype(np.float32),
                 "score": np.asarray(d["labels"], dtype=np.float64).reshape(-1)}
        if texts:
            feats["text"] = tokenize(texts, d["id"])
        elif bert:
            feats["text"] = np.stack([bf[i] for i in d["id"]]).astype(np.float64)
        y = (extra["score"] > 0).astype(np.int64)
        out[split] = [feats, y, extra]
    for m in MODS:
        if not (m == "text" and texts):
            mu = out["train"][0][m].mean(0, keepdims=True)
            sd = out["train"][0][m].std(0, keepdims=True) + 1e-6
            stats[m] = (mu.astype(np.float32), sd.astype(np.float32))
            for s in out:
                out[s][0][m] = ((out[s][0][m] - mu) / sd).astype(np.float32)
        if m != "text":                                    # frame-level statistics (train, real frames only)
            ex = out["train"][2]
            fr = ex[m][ex["valid"]]
            stats["frame_" + m] = (fr.mean(0).astype(np.float32), (fr.std(0) + 1e-6).astype(np.float32))
    return out, stats


def make_mask(n, rate, rng):
    keep = (rng.random((n, len(MODS))) >= rate)
    empty = ~keep.any(1)
    if empty.any():
        keep[empty, rng.integers(0, len(MODS), empty.sum())] = True
    return keep.astype(np.float32)


class MosiSet(Dataset):
    def __init__(self, feats, y, extra, stats, rate, seed, fixed=True, mask="modality", av_seq=False):
        self.mode, self.rate, self.fixed, self.av_seq = mask, rate, fixed, av_seq
        self.rng = np.random.default_rng(seed)
        self.y = torch.from_numpy(y)
        self.score = extra["score"]
        self.clean = {m: torch.from_numpy(feats[m]) for m in MODS}
        self.stats, self.extra = stats, extra
        if mask == "element":
            assert feats["text"].dtype == np.int64, "element masking needs --texts (token ids)"
        if av_seq:
            assert mask == "element", "--av-enc lstm is implemented for --mask element"
            v = extra["valid"][..., None]
            self.seq = {}
            for m in ("audio", "visual"):
                mu, sd = stats["frame_" + m]
                self.seq[m] = (((extra[m] - mu) / sd) * v).astype(np.float32)     # padded frames stay 0
            self.clean = dict(self.clean, **{m: torch.from_numpy(self.seq[m]) for m in ("audio", "visual")})
        self.resample()

    def resample(self):                       # training masks are redrawn every epoch
        n = len(self.y)
        if self.mode == "modality":
            self.x, self.av, self.xc = self.clean, torch.from_numpy(make_mask(n, self.rate, self.rng)), None
            return
        r, rng, ex = self.rate, self.rng, self.extra
        valid = ex["valid"]
        cnt = np.maximum(valid.sum(1), 1)
        x, cov = {}, np.ones((n, 3), np.float32)
        for k, m in ((1, "audio"), (2, "visual")):
            keep = (rng.random(valid.shape) >= r) & valid
            if self.av_seq:
                x[m] = torch.from_numpy(self.seq[m] * keep[..., None])
            else:
                pooled = (ex[m] * keep[..., None]).sum(1) / cnt[:, None]
                mu, sd = self.stats[m]
                x[m] = torch.from_numpy(((pooled - mu) / sd).astype(np.float32))
            cov[:, k] = keep.sum(1) / cnt
        ids = self.clean["text"].numpy().copy()
        real = (ids[:, 1] == 1) & (ids[:, 0] != CLS) & (ids[:, 0] != SEP)
        drop = (rng.random(real.shape) < r) & real
        ids[:, 0][drop] = UNK
        x["text"] = torch.from_numpy(ids)
        cov[:, 0] = 1 - drop.sum(1) / np.maximum(real.sum(1), 1)
        self.x, self.av, self.xc = x, torch.from_numpy(cov), self.clean

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        xc = {m: self.xc[m][i] for m in MODS} if self.xc is not None else {}
        return {m: self.x[m][i] for m in MODS}, xc, self.y[i], self.av[i], torch.tensor(self.score[i], dtype=torch.float32)
