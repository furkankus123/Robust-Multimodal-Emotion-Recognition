"""Frozen bert-base-uncased features for every MOSI segment (words come from mosi.hdf5).
Saves {'cls': {id: 768}, 'mean': {id: 768}} to bert_feats.pkl.  Needs: pip install transformers h5py
Uses the HuggingFace download of bert-base-uncased (internet needed once)."""
import argparse, pickle, h5py, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

p = argparse.ArgumentParser()
p.add_argument("--hdf5", default="../mosi.hdf5")
p.add_argument("--out", default="../bert_feats.pkl")
a = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(dev).eval()
f = h5py.File(a.hdf5, "r")
ids = sorted(f["words"].keys())
def sent(k):
    w = [x[0].decode("utf-8", "ignore") for x in f["words"][k]["features"][()]]
    return " ".join(x for x in w if x.lower() != "sp")
texts = [sent(k) for k in ids]
cls, mean = {}, {}
with torch.no_grad():
    for i in range(0, len(ids), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        h = bert(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        for j, k in enumerate(ids[i:i + 64]):
            cls[k] = h[j, 0].cpu().numpy()
            mean[k] = ((h[j] * m[j]).sum(0) / m[j].sum()).cpu().numpy()
pickle.dump({"cls": cls, "mean": mean}, open(a.out, "wb"))
print(len(ids), "segments;", "example:", texts[0])
