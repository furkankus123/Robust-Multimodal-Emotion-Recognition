"""words of every MOSI segment from mosi.hdf5 -> texts.pkl {segment_id: sentence} (no network needed)."""
import argparse, pickle, h5py
p = argparse.ArgumentParser()
p.add_argument("--hdf5", default="data/mosi.hdf5"); p.add_argument("--out", default="data/texts.pkl")
a = p.parse_args()
f = h5py.File(a.hdf5, "r")
out = {}
for k in f["words"]:
    w = [x[0].decode("utf-8", "ignore") for x in f["words"][k]["features"][()]]
    out[k] = " ".join(x for x in w if x.lower() != "sp")
pickle.dump(out, open(a.out, "wb")); print(len(out), "segments; e.g.", next(iter(out.values())))
