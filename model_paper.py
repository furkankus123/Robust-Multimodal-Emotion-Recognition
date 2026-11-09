"""Paper-faithful RMM-TCVAE, written 7 Nov 2026 from the paper's STAR Methods (Eq. 9-19, Algorithm 1, Table 9).

model.RMMTCVAE follows the authors' pseudocode file instead. Differences, each tied to the paper:
  * Conditional VAE (Eq. 11, 17): posterior q(z | x_o, x_m) and conditional prior p(z | x_o), KL between the two
    (the pseudocode uses a standard-normal prior).
      p(z | x_o)      Transformer encoder over the observed (masked) modality tokens   "available-modality TCVAE"
      q(z | x_o, x_m) Transformer encoder over observed + complete modality tokens     "reconstruction TCVAE"
    Training samples z from q (reparameterisation, Algorithm 1 l.10); evaluation uses the prior mean, because the
    complete input x_m is unknown at test time.
  * Encoder input P = modality tokens + modality-type embeddings (Eq. 9). Table 9: n = 4 layers, h = 2 heads, e = 128.
  * Decoder (Eq. 12): self-attention stack over [z; observed tokens]; its output h_m for each modality is
    (a) mapped back to the complete features for L_rec (Eq. 18) and (b) passed to the fusion network.
  * GMFN (Eq. 13-15): pairwise gates H_ij = z_p * tanh(W_i h_i) + (1 - z_p) * tanh(W_j h_j), z_p = sigmoid(W_z[h_i; h_j])
    for (a,v), (a,t), (v,t); multi-head attention over [H_av; H_at; H_vt]; a multimodal Transformer layer; MLP + softmax.
  * Loss (Eq. 19): CE + 0.1 KL + 0.01 L_rec.
Our reading where the paper is silent: latent size 64; one utterance-level token per modality, so positional
embeddings coincide with the modality-type embeddings and are not added separately; linear projections to e;
mean pooling over tokens before the latent heads and the classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MODS
from model import _Base, RMMTCVAE

PAIRS = [("audio", "visual"), ("audio", "text"), ("visual", "text")]


def transformer(e, heads, layers, drop):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(e, heads, 4 * e, drop, batch_first=True), layers)


class GatedPair(nn.Module):
    """Eq. 13"""
    def __init__(self, e):
        super().__init__()
        self.wi, self.wj, self.wz = nn.Linear(e, e), nn.Linear(e, e), nn.Linear(2 * e, e)

    def forward(self, hi, hj):
        zp = torch.sigmoid(self.wz(torch.cat([hi, hj], -1)))
        return zp * torch.tanh(self.wi(hi)) + (1 - zp) * torch.tanh(self.wj(hj))


class PaperRMM(_Base):
    _target = RMMTCVAE._target          # complete-input features as the reconstruction target (no gradient)

    def __init__(self, dims, e=128, z=64, n_cls=2, layers=4, heads=2, drop=0.1, finetune=False, elem=False,
                 raw_dims=None, lstm_hidden=64, hide_unk=False, aux_reg=False):
        super().__init__()
        self.dims, self.M, self.elem = dims, len(MODS), elem
        self._setup_extractors(finetune, raw_dims, lstm_hidden, hide_unk)
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], e) for m in MODS})
        self.type_emb = nn.Parameter(torch.randn(2 * self.M + 1, e) * 0.02)   # observed x3, complete x3, latent
        self.prior_enc, self.post_enc = transformer(e, heads, layers, drop), transformer(e, heads, layers, drop)
        self.prior_head, self.post_head = nn.Linear(e, 2 * z), nn.Linear(e, 2 * z)
        self.z_in = nn.Linear(z, e)
        self.dec = transformer(e, heads, layers, drop)
        self.rec = nn.ModuleDict({m: nn.Linear(e, dims[m]) for m in MODS})
        self.pairs = nn.ModuleList(GatedPair(e) for _ in PAIRS)
        self.mha = nn.MultiheadAttention(e, heads, dropout=drop, batch_first=True)
        self.fusion_tf = transformer(e, heads, 1, drop)
        self.cls = nn.Sequential(nn.Linear(e, e), nn.ReLU(), nn.Dropout(drop), nn.Linear(e, n_cls + int(aux_reg)))

    def _tokens(self, x, first_type):
        return torch.stack([self.proj[m](x[m]) for m in MODS], 1) + self.type_emb[first_type:first_type + self.M]

    def forward(self, x, avail, xc=None):
        x = self._feats(x)
        tgt = self._target(x, xc)
        xin = x if self.elem else {m: x[m] * avail[:, i:i + 1] for i, m in enumerate(MODS)}
        obs = self._tokens(xin, 0)
        mu_p, lv_p = self.prior_head(self.prior_enc(obs).mean(1)).chunk(2, -1)
        if self.training:
            both = torch.cat([obs, self._tokens(tgt, self.M)], 1)
            mu_q, lv_q = self.post_head(self.post_enc(both).mean(1)).chunk(2, -1)
            z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * lv_q)
        else:
            mu_q = lv_q = None
            z = mu_p
        z_tok = (self.z_in(z) + self.type_emb[-1]).unsqueeze(1)
        h = self.dec(torch.cat([z_tok, obs], 1))[:, 1:]
        rec = {m: self.rec[m](h[:, i]) for i, m in enumerate(MODS)}
        hm = dict(zip(MODS, h.unbind(1)))
        H = torch.stack([g(hm[i], hm[j]) for g, (i, j) in zip(self.pairs, PAIRS)], 1)
        H = self.fusion_tf(self.mha(H, H, H)[0])
        return self.cls(H.mean(1)), (mu_q, lv_q, mu_p, lv_p), None, rec, tgt


def paper_loss(out, y, score=None, reg_w=0.0, a_rec=0.01, b_kl=0.1):
    """Eq. 19: CE + 0.1 KL(q(z|x_o,x_m) || p(z|x_o)) + 0.01 L_rec. Same signature and returns as model.loss_fn."""
    logits, (mu_q, lv_q, mu_p, lv_p), _, rec, tgt = out
    ce = F.cross_entropy(logits[:, :2], y)
    if reg_w > 0:
        ce = ce + reg_w * F.l1_loss(logits[:, 2], score)
    kl = (0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp() - 1).sum(1)).mean()
    r = sum(F.mse_loss(rec[m], tgt[m]) for m in MODS)
    return ce + b_kl * kl + a_rec * r, ce, r.detach(), kl.detach()
