import numpy as np


def acc_waf(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    acc = (pred == y).mean()
    f1s, ws = [], []
    for c in np.unique(y):
        tp = ((pred == c) & (y == c)).sum()
        p = tp / max((pred == c).sum(), 1)
        r = tp / max((y == c).sum(), 1)
        f1s.append(2 * p * r / max(p + r, 1e-12))
        ws.append((y == c).sum())
    return 100 * acc, 100 * float(np.average(f1s, weights=ws))     # weighted average F1 (WAF)
