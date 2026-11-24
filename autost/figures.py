"""The paper's figures, remade from a finished run (after train and evaluate).

    python -m autost.figures                 # every figure
    python -m autost.figures --only 6,8      # some of them

Each figure follows the layout of the manuscript's figure with the same
number (Data Driven Fire Spread Manuscript, Aug 2026). Every prediction shown
comes from the model whose TEST fold holds that fire, cut at the threshold
chosen on that model's VALIDATION fold -- nothing here was tuned on the fires
it shows.

Figures 1, 2 and 10 of the manuscript (the fire map, every raw layer of one
fire-day, whole-fire predictions) need the full fire rasters, which are not
published; they are not made here.

Writes <data_root>/figures/figNN_<name>.png and .pdf:
    fig03_architecture     the network as a U, drawn at print size (needs no run)
    fig04_correlation      correlation of crop-mean channel values, static + day t
    fig05_crop_example     one test crop: every channel, target, probability, overlay
    fig06_training_curves  training loss, validation loss, F1, AUC per epoch and fold
    fig07_attribution      Integrated Gradients per channel (evaluate --attribution)
    fig08_thresholds       precision, recall, F1 against the threshold, test folds
    fig09_crop_grid        100 random test crops: hits, false alarms, misses
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import channels as CH
from . import config as C
from . import folds as FO
from .data import manifest

FOLD_COLORS = None   # matplotlib tab10, set when plotting

# Outcome colours: the manuscript's crop figures (5, 9).
CROP_RGB = {"unburned": (0.9, 0.9, 0.9), "today": (1.0, 0.5, 0.0), "tp": (0.0, 1.0, 0.0),
            "fp": (1.0, 0.0, 0.0), "fn": (0.0, 0.0, 1.0)}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight"})
    return plt


def _save(fig, out: Path, name: str):
    plt = _plt()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}.png / .pdf")
    return out / f"{name}.png"


def _outcome_rgb(today, target, pred, colors):
    rgb = np.empty(today.shape + (3,), dtype=np.float32)
    rgb[:] = colors["unburned"]
    rgb[today] = colors["today"]
    m = ~today
    rgb[m & pred & target] = colors["tp"]
    rgb[m & pred & ~target] = colors["fp"]
    rgb[m & ~pred & target] = colors["fn"]
    return rgb


# --------------------------------------------------------------- models
class Models:
    """Loads each fold's model once; predicts crops with the model that tests them."""

    def __init__(self, cfg):
        import torch
        from .evaluate import load_model
        self.torch, self._load = torch, load_model
        self.cfg = cfg
        self.dir = cfg.data_path(cfg.model["training"]["out_dir"])
        self.name = cfg.model["training"]["name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self._cache = {}

    def get(self, fold):
        if fold not in self._cache:
            self._cache = {fold: self._load(self.dir / f"{self.name}_fold{fold}.pt", self.device)}
        return self._cache[fold]

    def predict(self, fold, static, dynamic):
        model, ck = self.get(fold)
        t = self.torch
        with t.no_grad():
            xs = t.from_numpy(np.asarray(static, np.float32)).to(self.device)
            xd = t.from_numpy(np.asarray(dynamic, np.float32)).to(self.device)
            return t.sigmoid(model(xs, xd).float()).cpu().numpy()[:, 0], float(ck["threshold"])


def _test_crops(cfg, n, seed, min_new=0):
    """n random crops, each with the fold that tests its fire."""
    man = manifest(cfg)
    folds = FO.load(cfg).set_index("incident_c")["fold"]
    man = man.assign(fold=man["incident_c"].map(folds).astype(int))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(man))
    crop_dir = cfg.data_path(cfg.model["crops"]["dir"])
    picked = []
    for i in order:
        row = man.iloc[i]
        with np.load(crop_dir / row["file"]) as z:
            s, d, y = (z[k].astype(np.float32) for k in ("static", "dynamic", "target"))
        if min_new and ((y[0] > 0) & (d[-1, 0] == 0)).sum() < min_new:
            continue
        picked.append((row, s, d, y))
        if len(picked) >= n:
            break
    return picked


# --------------------------------------------------------------- figure 4
# ---- Figure 3: the network, drawn as a U at print size ----------------------------------
_ARCH_COL = {"dyn": ("#dbe9f6", "#2b6ca3"), "sta": ("#e3f1dc", "#3d7a2a"), "trf": ("#ece2f4", "#6a3d9a"),
             "att": ("#fdebd3", "#c46a00"), "out": ("#f6dcdc", "#a52a2a")}


def _arch_box(ax, x, y, w, h, title, sub, kind):
    from matplotlib.patches import FancyBboxPatch
    face, edge = _ARCH_COL[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=face, ec=edge, lw=1.1))
    ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=8.5, weight="bold", color="#222")
    ax.text(x, y - h * 0.22, sub, ha="center", va="center", fontsize=7.5, color="#333")


def _arch_arrow(ax, p, q, color="#333", label=None, loff=(0, 0)):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=0,
                                                    shrinkB=0, mutation_scale=9))
    if label:
        ax.text((p[0] + q[0]) / 2 + loff[0], (p[1] + q[1]) / 2 + loff[1], label, ha="center",
                va="center", fontsize=7.5, style="italic", color=color)


def fig_architecture(cfg, out):
    """The network (autost/model.py) as a U: encoder down the left, decoder up the
    right, attention on the skip links between them, the transformer and the map
    layers joining at the bottom. Drawn at its printed size (7.2 in wide), so the
    text is 7.5-8.5 pt on the page."""
    plt = _plt()
    from matplotlib.patches import Circle, FancyBboxPatch
    S = len(CH.static_names(bool(cfg.model["channels"]["firelines"])))
    D, T, N = len(CH.DYNAMIC_NAMES), int(cfg.model["crops"]["sequence"]), int(cfg.model["crops"]["size"])
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(0, 100); ax.set_ylim(-6.5, 66); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    W, H = 19, 7.0
    lv, ex, dx = [50, 39, 28, 17], [12, 16, 20, 24], [88, 84, 80, 76]
    ch, px = [64, 128, 256, 512], [N, N // 2, N // 4, N // 8]
    box, arrow, att = (lambda *a: _arch_box(ax, *a)), (lambda *a, **k: _arch_arrow(ax, *a, **k)), _ARCH_COL["att"][1]

    box(12, 61, 22, H, "Daily inputs", f"{D} layers × {T} days, {N}×{N}", "dyn")
    arrow((12, 61 - H / 2), (12, lv[0] + H / 2))
    box(88, 61, 22, H, "Burn probability", f"next perimeter, {N}×{N}", "out")
    arrow((dx[0], lv[0] + H / 2), (88, 61 - H / 2), label="mean over days,\n1×1 conv", loff=(-9.5, 0))
    for k in range(4):
        box(ex[k], lv[k], W, H, f"Encoder {k + 1}", f"{ch[k]} ch, {px[k]}×{px[k]}×{T}", "dyn")
        box(dx[k], lv[k], W, H, f"Decoder {k + 1}", f"{ch[k]} ch, {px[k]}×{px[k]}×{T}", "dyn")
        box(50, lv[k], 17, H, "Attention", "channel + spatial", "att")
        arrow((ex[k] + W / 2, lv[k]), (50 - 8.5, lv[k]), color=att)
        arrow((50 + 8.5, lv[k]), (dx[k] - W / 2, lv[k]), color=att)
        if k < 3:
            arrow((ex[k], lv[k] - H / 2), (ex[k], lv[k + 1] + H / 2), label="pool", loff=(-4.2, 0))
            arrow((dx[k + 1] + 4, lv[k + 1] + H / 2), (dx[k + 1] + 4, lv[k] - H / 2), label="up-conv",
                  loff=(5.6, 0))

    # bottom row: bottleneck, transformer, join (C), add the map layers (+)
    yb, ys, xb, xt, xc, xp, trf = 4.0, -3.2, 30, 51.5, 67.5, 76, _ARCH_COL["trf"][1]
    box(xb, yb, 17, H, "Bottleneck", f"512 ch, {N // 16}×{N // 16}×{T}", "dyn")
    arrow((xb, lv[3] - H / 2), (xb, yb + H / 2), label="pool", loff=(-3.6, 0))
    box(xt, yb, 19, H, "Transformer", "4 layers, 8 heads", "trf")
    arrow((xb + 8.5, yb), (xt - 9.5, yb), color=trf)
    for x, sym in ((xc, "C"), (xp, "+")):
        ax.add_patch(Circle((x, yb), 2.3, fc="white", ec="#333", lw=1.1, zorder=3))
        ax.text(x, yb, sym, ha="center", va="center", fontsize=10, weight="bold", zorder=4)
    arrow((xt + 9.5, yb), (xc - 2.3, yb), color=trf)
    ax.plot([xb, xb, xc], [yb - H / 2, -0.9, -0.9], color="#333", lw=1.1)   # the bottleneck joins too
    arrow((xc, -0.9), (xc, yb - 2.3))
    arrow((xc + 2.3, yb), (xp - 2.3, yb))
    ax.text(xc, yb + 4.2, "1024 ch", ha="center", va="center", fontsize=7.5, style="italic", color="#333")
    arrow((xp, yb + 2.3), (dx[3], lv[3] - H / 2), label="up-conv", loff=(4.4, 0))
    g = _ARCH_COL["sta"][1]
    box(10, yb, 17, H, "Map inputs", f"{S} layers, {N}×{N}", "sta")
    ax.plot([10, 10, xp], [yb - H / 2, ys, ys], color=g, lw=1.1)
    arrow((xp, ys), (xp, yb - 2.3), color=g)
    ax.text(43, ys - 1.9, f"four strided 2-D convolutions down to {N // 16}×{N // 16}, repeated for each of "
            f"the {T} days", ha="center", va="center", fontsize=7.5, style="italic", color=g)

    # key
    for i, (kind, txt) in enumerate([("dyn", "daily data (3-D conv.)"), ("sta", "map layers (2-D conv.)"),
                                     ("att", "skip link with attention"), ("trf", "transformer")]):
        x0, y0 = 27.5 + (i % 2) * 24, 63.2 - (i // 2) * 3.6
        face, edge = _ARCH_COL[kind]
        ax.add_patch(FancyBboxPatch((x0, y0 - 1.1), 2.6, 2.2, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=face, ec=edge, lw=1.0))
        ax.text(x0 + 3.6, y0, txt, ha="left", va="center", fontsize=7.5, color="#222")
    ax.text(50, 55.8, "C  join (concatenate)       +  add", ha="center", va="center", fontsize=7.5, color="#222")
    fig.savefig(out / "fig03_architecture.png", dpi=300)      # no tight bbox: keep the print size
    fig.savefig(out / "fig03_architecture.pdf")
    plt.close(fig)
    print("  fig03_architecture.png / .pdf")
    return out / "fig03_architecture.png"



def fig_correlation(cfg, out, n=5000, seed=7):
    plt = _plt()
    man = manifest(cfg)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(man), min(n, len(man)), replace=False)
    crop_dir = cfg.data_path(cfg.model["crops"]["dir"])
    vals = []
    for i in idx:
        with np.load(crop_dir / man.iloc[i]["file"]) as z:
            s = z["static"].astype(np.float32).mean(axis=(1, 2))
            d = z["dynamic"][-1].astype(np.float32).mean(axis=(1, 2))   # day t
        vals.append(np.concatenate([s, d]))
    n_s = len(s)
    names = CH.static_names(n_s == 33) + list(CH.DYNAMIC_NAMES)
    df = pd.DataFrame(vals, columns=names)
    corr = df.corr()
    corr.to_csv(out / "fig04_correlation.csv")
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for v in (n_s - 0.5,):
        ax.axhline(v, color="k", lw=1); ax.axvline(v, color="k", lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Pearson r")
    ax.set_title(f"Correlation of crop-mean channel values ({len(idx):,} crops; "
                 f"static above/left of the line, day t daily below/right)")
    return _save(fig, out, "fig04_correlation")


# --------------------------------------------------------------- figure 5
def _variety(s, d):
    """How much a crop shows: inputs that vary inside it, management layers present."""
    chans = np.concatenate([s, d[-1]])
    varying = int((chans.reshape(len(chans), -1).std(axis=1) > 0.02).sum())
    names = CH.static_names(s.shape[0] == 33)
    present = sum(bool(s[names.index(n)].any()) for n in
                  ("Hand Line", "Dozer Line", "Other Line", "Recent Burn", "Fuel Treatment", "Roads", "PODs")
                  if n in names)
    fuels = sum(bool(s[names.index(f"FM{k}")].mean() > 0.03) for k in range(1, 13))
    return varying + 2 * present + fuels


def _variety(s, d):
    """How much a crop shows: inputs that vary inside it, management layers present."""
    chans = np.concatenate([s, d[-1]])
    varying = int((chans.reshape(len(chans), -1).std(axis=1) > 0.02).sum())
    names = CH.static_names(s.shape[0] == 33)
    present = sum(bool(s[names.index(n)].any()) for n in
                  ("Hand Line", "Dozer Line", "Other Line", "Recent Burn", "Fuel Treatment", "Roads", "PODs")
                  if n in names)
    fuels = sum(bool(s[names.index(f"FM{k}")].mean() > 0.03) for k in range(1, 13))
    return varying + 2 * present + fuels


def fig_crop_example(cfg, models, out, seed=11, pool=4000, shortlist=300, min_f1=0.7):
    """One test crop that shows the inputs well and that the model gets mostly right:
    from `pool` random test crops with a clear band of new spread, the `shortlist`
    with the most varied inputs are predicted, and the most varied one with
    F1 >= min_f1 is drawn (else the best F1)."""
    plt = _plt()
    cands = []
    for row, s, d, y in _test_crops(cfg, pool, seed, min_new=150):
        today, target = d[-1, 0] > 0, y[0] > 0
        new = (target & ~today).sum() / max((~today).sum(), 1)
        if 0.15 <= today.mean() <= 0.7 and 0.05 <= new <= 0.5:
            cands.append((_variety(s, d), row, s, d, y))
    cands.sort(key=lambda c: -c[0])
    cands = cands[:shortlist]
    scored = []
    for fold in sorted({int(c[1]["fold"]) for c in cands}):
        sub = [c for c in cands if int(c[1]["fold"]) == fold]
        prob, thr = models.predict(fold, np.stack([c[2] for c in sub]), np.stack([c[3] for c in sub]))
        for (v, row, s, d, y), p in zip(sub, prob):
            today, target = d[-1, 0] > 0, y[0] > 0
            pred, m = p >= thr, ~(d[-1, 0] > 0)
            tp, fp, fn = (pred & target & m).sum(), (pred & ~target & m).sum(), (~pred & target & m).sum()
            scored.append((2 * tp / max(2 * tp + fp + fn, 1), v, row, s, d, y, p, thr))
    good = [c for c in scored if c[0] >= min_f1]
    f1, v, row, s, d, y, prob, thr = (max(good, key=lambda c: (c[1], c[0])) if good
                                      else max(scored, key=lambda c: c[0]))
    fold = int(row["fold"])
    today, target = d[-1, 0] > 0, y[0] > 0
    pred = prob >= thr
    names = CH.static_names(s.shape[0] == 33) + [f"{n} (day t)" for n in CH.DYNAMIC_NAMES]
    chans = np.concatenate([s, d[-1]])
    ncol = 9
    nrow = int(np.ceil(len(names) / ncol)) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.55, nrow * 1.7))
    for ax in axes.flat:
        ax.set_axis_off()
    for k, (nm, a) in enumerate(zip(names, chans)):
        ax = axes.flat[k]
        im = ax.imshow(a, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set_title(nm, fontsize=7)
    last = axes[-1]
    tgt = np.empty(today.shape + (3,), dtype=np.float32)
    tgt[:] = CROP_RGB["unburned"]
    tgt[target] = (0.35, 0.35, 0.35)
    tgt[target & ~today] = (0.85, 0.25, 0.1)
    last[0].imshow(tgt, interpolation="nearest")
    last[0].set_title("TARGET\nnext perimeter", fontsize=8, fontweight="bold")
    last[1].imshow(prob, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    last[1].set_title("PREDICTION\nprobability", fontsize=8, fontweight="bold")
    last[2].imshow(_outcome_rgb(today, target, pred, CROP_RGB), interpolation="nearest")
    last[2].set_title(f"OVERLAY (th = {thr:.2f})\nF1 (spread) = {f1:.2f}", fontsize=8, fontweight="bold")
    from matplotlib.patches import Patch
    last[3].legend(handles=[Patch(color=(0.35, 0.35, 0.35), label="target: burning day t"),
                            Patch(color=(0.85, 0.25, 0.1), label="target: new burn")],
                   loc="center left", fontsize=7, frameon=False)
    last[5].legend(handles=[Patch(color=CROP_RGB[k], label=l) for k, l in
                            [("tp", "TP (hit)"), ("fp", "FP (false alarm)"), ("fn", "FN (missed)"),
                             ("today", "day t fire"), ("unburned", "unburned")]],
                   loc="center left", fontsize=7, frameon=False)
    cax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cax, label="scaled input (0-1)")
    fig.subplots_adjust(right=0.9, top=0.95, hspace=0.45, wspace=0.08)
    fig.suptitle(f"Test crop {row['file']}  (fold {fold} model, day t {row['date']} -> {row['next_date']})",
                 fontsize=10, y=0.985)
    print(f"  example: {row['file']}  F1 {f1:.2f}  variety {v}  ({len(good)} of {len(scored)} shortlisted with F1 >= {min_f1})")
    return _save(fig, out, "fig05_crop_example")


# --------------------------------------------------------------- figure 6
def fig_training(cfg, out):
    plt = _plt()
    mdir = cfg.data_path(cfg.model["training"]["out_dir"])
    name, n = cfg.model["training"]["name"], int(cfg.model["folds"]["n"])
    panels = [("train_loss", "Train Loss", "loss"), ("val_loss", "Validation Loss", "loss"),
              ("val_f1_best", "Validation F1", "F1"), ("val_auc", "Validation AUC", "AUC")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True)
    cmap = plt.get_cmap("tab10")
    import torch
    for k in range(1, n + 1):
        p = mdir / f"{name}_fold{k}_log.csv"
        if not p.exists():
            continue
        log = pd.read_csv(p)
        best = int(torch.load(mdir / f"{name}_fold{k}.pt", map_location="cpu", weights_only=False)["epoch"])
        for ax, (col, title, yl) in zip(axes.flat, panels):
            ax.plot(log["epoch"], log[col], color=cmap((k - 1) % 10), lw=1.4, label=f"Fold {k}")
            b = log[log["epoch"] == best]
            ax.plot(b["epoch"], b[col], "o", color=cmap((k - 1) % 10), ms=5, mec="k", mew=0.6)
            ax.set_title(title); ax.set_ylabel(yl); ax.grid(True, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, int(log["epoch"].max()) + 1, 2))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.01, "Dots: the epoch kept for each fold (best F1 on its validation fold). "
             "Validation F1 is pooled over the validation fold at its best threshold.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, "fig06_training_curves")


# --------------------------------------------------------------- figure 7
def fig_attribution(cfg, out):
    plt = _plt()
    p = cfg.data_path(cfg.model["evaluation"]["out_dir"]) / "attribution_per_fold.csv"
    if not p.exists():
        print("  fig07: no attribution yet -- run `python -m autost.evaluate --attribution`")
        return None
    ad = pd.read_csv(p)
    g = ad.groupby("channel")["attribution"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.5, 12))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in g["mean"]]
    ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=colors, alpha=0.85,
            error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.grid(axis="x", alpha=0.4)
    ax.set_xlabel("Integrated Gradients attribution (spread-region mean, logit units)")
    ax.set_title(f"Per-channel attribution across {ad['fold'].nunique()} folds (test crops; bars = mean, "
                 "whiskers = 1 sd)", fontsize=10)
    return _save(fig, out, "fig07_attribution")


# --------------------------------------------------------------- figure 8
def fig_thresholds(cfg, out):
    plt = _plt()
    ev = cfg.data_path(cfg.model["evaluation"]["out_dir"])
    sw = pd.read_csv(ev / "sweep.csv")
    sw = sw[sw["region"] == "spread"]
    pf = pd.read_csv(ev / "per_fold.csv")
    pf = pf[pf["region"] == "spread"]
    fig, ax = plt.subplots(figsize=(9, 6))
    style = {"f1": ("F1", "#1f77b4"), "precision": ("Precision", "#2ca02c"), "recall": ("Recall", "#ff7f0e")}
    for m, (lab, col) in style.items():
        piv = sw.pivot(index="threshold", columns="fold", values=m)
        for f in piv.columns:
            ax.plot(piv.index, piv[f], color=col, lw=0.7, alpha=0.3)
        mu, sd = piv.mean(axis=1), piv.std(axis=1)
        ax.fill_between(piv.index, mu - sd, mu + sd, color=col, alpha=0.15)
        ax.plot(piv.index, mu, "o-", color=col, lw=2.5, ms=4, label=f"{lab} (mean)")
    thr = pf["threshold"]
    for t in thr:
        ax.plot([t, t], [0, 0.03], color="k", lw=1)
    ax.axvline(thr.mean(), color="k", ls="--", lw=1,
               label=f"thresholds chosen on validation (ticks; mean {thr.mean():.2f})\n"
                     f"test F1 at them = {pf['f1'].mean():.3f} +/- {pf['f1'].std():.3f}")
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"AutoST {pf['fold'].nunique()}-fold test: F1 / Precision / Recall vs threshold\n"
                 "(thin lines = per fold, thick = mean, band = +/- 1 sd)")
    return _save(fig, out, "fig08_thresholds")


# --------------------------------------------------------------- figure 9
def fig_crop_grid(cfg, models, out, n=100, seed=3):
    plt = _plt()
    crops = _test_crops(cfg, n, seed)
    crops.sort(key=lambda c: int(c[0]["fold"]))
    tiles, thrs = [], set()
    for fold in sorted({int(c[0]["fold"]) for c in crops}):
        sub = [c for c in crops if int(c[0]["fold"]) == fold]
        prob, thr = models.predict(fold, np.stack([c[1] for c in sub]), np.stack([c[2] for c in sub]))
        thrs.add(thr)
        for (row, s, d, y), p in zip(sub, prob):
            tiles.append(_outcome_rgb(d[-1, 0] > 0, y[0] > 0, p >= thr, CROP_RGB))
    side = int(np.ceil(np.sqrt(len(tiles))))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.4, side * 1.4 + 0.8))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))               # mix the folds back up
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, i in zip(axes.flat, order):
        ax.imshow(tiles[i], interpolation="nearest")
    fig.suptitle(f"{len(tiles)} random test crops (each fold's validation threshold, "
                 f"{min(thrs):.2f}-{max(thrs):.2f})\nGreen = TP (hit) | Red = FP (false alarm) | Blue = FN (missed) | "
                 "Orange = day t fire | Gray = unburned", fontsize=12)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.93)
    return _save(fig, out, "fig09_crop_grid")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 6,8")
    a = ap.parse_args(argv)
    cfg = C.load(a.data_config, a.config)
    out = cfg.data_path("figures")
    out.mkdir(parents=True, exist_ok=True)
    want = {int(x) for x in a.only.split(",")} if a.only else {3, 4, 5, 6, 7, 8, 9}
    models = Models(cfg) if want & {5, 9} else None
    steps = [(3, lambda: fig_architecture(cfg, out)),
             (4, lambda: fig_correlation(cfg, out)),
             (5, lambda: fig_crop_example(cfg, models, out)),
             (6, lambda: fig_training(cfg, out)),
             (7, lambda: fig_attribution(cfg, out)),
             (8, lambda: fig_thresholds(cfg, out)),
             (9, lambda: fig_crop_grid(cfg, models, out))]
    for k, fn in steps:
        if k in want:
            print(f"figure {k}")
            try:
                fn()
            except Exception as e:                     # one figure failing should not stop the rest
                print(f"  figure {k} FAILED: {type(e).__name__}: {e}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
