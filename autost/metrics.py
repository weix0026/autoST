"""Scores, pooled over ALL the cells of a set of crops (not averaged batch by
batch: F1 is not linear, so an average of per-batch F1s is a different and
lower number -- the original training log did that; docs/CHANGES.md, item 4).

Only cells NOT burning on day t are scored (the spread region).

NumPy only (no scikit-learn), so the scores are the same on every machine.
"""
from __future__ import annotations

import numpy as np

N_BINS = 1000          # probability histogram resolution for threshold sweeps


class Counter:
    """Accumulates a histogram of predicted probabilities for burned and
    unburned cells; gives confusion counts at any threshold on the 0.001 grid,
    and approximate AUC / PR-AUC (exact versions: `auc_exact`)."""

    def __init__(self):
        self.pos = np.zeros(N_BINS, dtype=np.int64)
        self.neg = np.zeros(N_BINS, dtype=np.int64)

    def add(self, prob, target, mask=None):
        prob, target = np.asarray(prob, dtype=np.float64).ravel(), np.asarray(target).ravel() > 0.5
        if mask is not None:
            m = np.asarray(mask, dtype=bool).ravel()
            prob, target = prob[m], target[m]
        b = np.clip((prob * N_BINS).astype(np.int64), 0, N_BINS - 1)
        self.pos += np.bincount(b[target], minlength=N_BINS)
        self.neg += np.bincount(b[~target], minlength=N_BINS)

    def counts(self, threshold: float):
        """(tp, fp, fn, tn) for 'burns' = probability >= threshold."""
        k = int(round(threshold * N_BINS))
        tp, fn = int(self.pos[k:].sum()), int(self.pos[:k].sum())
        fp, tn = int(self.neg[k:].sum()), int(self.neg[:k].sum())
        return tp, fp, fn, tn

    def scores(self, threshold: float) -> dict:
        return scores_from_counts(*self.counts(threshold))

    def best_f1(self, thresholds):
        best = max(thresholds, key=lambda t: (self.scores(t)["f1"], -abs(t - 0.5)))
        return best, self.scores(best)["f1"]

    def auc(self) -> float:
        P, N = self.pos.sum(), self.neg.sum()
        if P == 0 or N == 0:
            return float("nan")
        neg_below = np.concatenate([[0], np.cumsum(self.neg)[:-1]])
        return float((self.pos * (neg_below + 0.5 * self.neg)).sum() / (P * N))

    def pr_auc(self) -> float:
        P = self.pos.sum()
        if P == 0:
            return float("nan")
        tp = np.cumsum(self.pos[::-1])
        fp = np.cumsum(self.neg[::-1])
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / P
        return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    @property
    def prevalence(self) -> float:
        n = self.pos.sum() + self.neg.sum()
        return float(self.pos.sum() / n) if n else float("nan")


def scores_from_counts(tp, fp, fn, tn) -> dict:
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else float("nan")
    denom = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "specificity": spec,
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "mcc": (tp * tn - fp * fn) / denom ** 0.5 if denom else 0.0,
        "tss": rec + spec - 1.0 if np.isfinite(spec) else float("nan"),
    }


def auc_exact(prob, target) -> float:
    """Exact ROC AUC (Mann-Whitney, ties counted half)."""
    from scipy.stats import rankdata
    prob, target = np.asarray(prob).ravel(), np.asarray(target).ravel() > 0.5
    P, N = target.sum(), (~target).sum()
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(prob)
    return float((r[target].sum() - P * (P + 1) / 2) / (P * N))


def front_band(fire_today, k: int = 3):
    """Cells within k cells of day t's fire that are not burning: the active
    front, where 'burns or not' is a real question. fire_today: (..., H, W)."""
    from scipy.ndimage import binary_dilation
    f = np.asarray(fire_today) > 0
    st = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    if f.ndim == 2:
        return binary_dilation(f, st) & ~f
    return np.stack([binary_dilation(x, st) & ~x for x in f])
