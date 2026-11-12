"""RMM-TCVAE re-implementation from the paper text + the pseudocode file in the authors' repository
(github.com/WANGShuaiShine/RMM-TCVAE, commit d578635; the pseudocode file is not copied into this repository).
The authors' model files are not in their repo, so this is NOT their code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MODS, MAXLEN, UNK

SEQ_LEN = 50


def mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())


class BertText(nn.Module):
    """bert-base-uncased, fine-tuned end to end; returns the 768-d [CLS] vector."""
    def __init__(self, hide_unk=False):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.hide_unk = hide_unk

    def forward(self, t):                                   # t (B,2,L) long: ids, mask
        mask = t[:, 1]
        if self.hide_unk:                                   # missing words are hidden from attention instead of shown as [UNK]
            mask = mask * (t[:, 0] != UNK).long()
        return self.bert(input_ids=t[:, 0], attention_mask=mask).last_hidden_state[:, 0]


class SeqEnc(nn.Module):
    """Single-layer LSTM over the 50 frames (paper). Sequences are front-padded, so the last step is the
    segment's last real frame."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_out, batch_first=True)

    def forward(self, s):                                   # s (B,T,D)
        return self.lstm(s)[0][:, -1]


class _Base(nn.Module):
    def _setup_extractors(self, finetune, raw_dims, lstm_hidden, hide_unk=False):
        self.bert = BertText(hide_unk) if finetune else None
        self.seq = nn.ModuleDict({m: SeqEnc(raw_dims[m], lstm_hidden) for m in ("audio", "visual")}) if raw_dims else None
        self.raw_dims = raw_dims

    def _feats(self, x):
        x = dict(x)
        if self.bert is not None:
            x["text"] = self.bert(x["text"])
        if self.seq is not None:
            for m in ("audio", "visual"):
                x[m] = self.seq[m](x[m])
        return x

    def example_input(self, device):
        x = {m: torch.randn(1, self.dims[m], device=device) for m in MODS}
        if self.bert is not None:
            x["text"] = torch.ones(1, 2, MAXLEN, dtype=torch.long, device=device)
        if self.seq is not None:
            for m in ("audio", "visual"):
                x[m] = torch.randn(1, SEQ_LEN, self.raw_dims[m], device=device)
        return x


class GMFN(nn.Module):
    """Pseudocode 1.2: importance weights * (1-missing) -> 4-head attention over modality tokens -> concat -> MLP."""
    def __init__(self, h, M, gate="keep"):
        super().__init__()
        self.gate = gate
        self.imp = nn.Parameter(torch.ones(M))
        self.attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(h * M, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, feats, avail, mult=None):
        w = self.imp.unsqueeze(0).expand_as(avail)
        if mult is not None:
            w = w * mult
        if self.gate == "zero":
            w = w * avail
        x = feats * w.unsqueeze(-1)
        a, _ = self.attn(x, x, x)
        return self.fusion(a.flatten(1))


class RMMTCVAE(_Base):
    def __init__(self, dims, h=128, z=64, n_cls=2, gate="keep", layers=2, heads=4, drop=0.1, finetune=False,
                 elem=False, unc="none", K=8, raw_dims=None, lstm_hidden=64, hide_unk=False, aux_reg=False):
        super().__init__()
        self.dims, self.M, self.elem = dims, len(MODS), elem
        self.unc, self.K = unc, K
        self._setup_extractors(finetune, raw_dims, lstm_hidden, hide_unk)
        self.tau = nn.Parameter(torch.zeros(()), requires_grad=(unc in ("gate", "shuffle")))
        self.enc = nn.ModuleDict({m: mlp(dims[m], h) for m in MODS})
        self.mu = nn.Linear(h * self.M, z)
        self.logvar = nn.Linear(h * self.M, z)
        self.z2tok = nn.Linear(z, self.M * z)
        self.mod_emb = nn.Parameter(torch.randn(self.M, z) * 0.02)
        layer = nn.TransformerEncoderLayer(z, heads, 4 * z, drop, batch_first=True)
        self.trn = nn.TransformerEncoder(layer, layers)
        self.dec = nn.ModuleDict({m: nn.Linear(z, dims[m]) for m in MODS})
        self.gmfn = GMFN(h, self.M, gate)
        self.cls = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_cls + int(aux_reg)))   # +1 = regression output

    def encode_all(self, x):
        return torch.stack([self.enc[m](x[m]) for m in MODS], 1)

    def _target(self, x, xc):
        """reconstruction target = features of the COMPLETE (uncorrupted) input, no gradient"""
        if not xc:
            return {m: x[m].detach() for m in MODS}
        was = self.training
        self.eval()
        with torch.no_grad():
            t = self._feats(xc)
        self.train(was)
        return {m: t[m].detach() for m in MODS}

    @torch.no_grad()
    def _unc_mult(self, mu, logvar):
        B, K = mu.size(0), self.K
        std = torch.exp(0.5 * logvar)
        det = not self.training
        g = torch.Generator(device=mu.device).manual_seed(0) if det else None
        z = mu.unsqueeze(0) + torch.randn((K,) + tuple(mu.shape), device=mu.device, generator=g) * std.unsqueeze(0)
        tok = self.z2tok(z.reshape(K * B, -1)).view(K * B, self.M, -1) + self.mod_emb
        t = self.trn(tok)
        u = torch.stack([self.dec[m](t[:, i]).view(K, B, -1).var(0).mean(-1) for i, m in enumerate(MODS)], 1)
        u = torch.log(u + 1e-8)
        u = u - u.mean(1, keepdim=True)
        if self.unc == "shuffle":
            cg = torch.Generator().manual_seed(0) if det else None
            u = u[torch.randperm(B, generator=cg).to(u.device)]
        return u

    def forward(self, x, avail, xc=None):
        x = self._feats(x)
        tgt = self._target(x, xc)
        xin = x if self.elem else {m: x[m] * avail[:, i:i + 1] for i, m in enumerate(MODS)}
        f = self.encode_all(xin)
        c = f.flatten(1)
        mu, logvar = self.mu(c), self.logvar(c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        tok = self.z2tok(z).view(z.size(0), self.M, -1) + self.mod_emb
        t = self.trn(tok)
        rec = {m: self.dec[m](t[:, i]) for i, m in enumerate(MODS)}
        filled = {m: xin[m] + (1 - avail[:, i:i + 1]) * rec[m] for i, m in enumerate(MODS)}
        f2 = self.encode_all(filled)
        mult = None
        if self.unc != "none":
            u = self._unc_mult(mu, logvar)
            tau = 1.0 if self.unc == "fixedinv" else self.tau
            mult = self.M * torch.softmax(-tau * u, dim=1)
        fused = self.gmfn(f2, avail, mult)
        return self.cls(fused), mu, logvar, rec, tgt


class ConcatBaseline(_Base):
    """Control: same feature extractors (BERT, LSTMs) + encoders + classifier; no VAE / TRN / GMFN."""
    def __init__(self, dims, h=128, n_cls=2, finetune=False, elem=False, raw_dims=None, lstm_hidden=64,
                 hide_unk=False, aux_reg=False):
        super().__init__()
        self.dims, self.elem = dims, elem
        self._setup_extractors(finetune, raw_dims, lstm_hidden, hide_unk)
        self.enc = nn.ModuleDict({m: mlp(dims[m], h) for m in MODS})
        self.cls = nn.Sequential(nn.Linear(h * len(MODS), h), nn.ReLU(), nn.Linear(h, n_cls + int(aux_reg)))

    def forward(self, x, avail, xc=None):
        x = self._feats(x)
        f = [self.enc[m](x[m] if self.elem else x[m] * avail[:, i:i + 1]) for i, m in enumerate(MODS)]
        return self.cls(torch.cat(f, 1)), None, None, None, None


def loss_fn(out, y, score=None, reg_w=0.0, a_rec=0.01, b_kl=0.1):
    """Paper loss: CE + 0.1 KL + 0.01 Lrec.  Optional (ours): + reg_w * L1(regression output, sentiment score)."""
    logits, mu, logvar, rec, tgt = out
    ce = F.cross_entropy(logits[:, :2], y)
    if reg_w > 0:
        ce = ce + reg_w * F.l1_loss(logits[:, 2], score)
    if mu is None:
        return ce, ce, torch.zeros(()), torch.zeros(())
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    r = sum(F.mse_loss(rec[m], tgt[m]) for m in MODS)
    return ce + b_kl * kl + a_rec * r, ce, r.detach(), kl.detach()
