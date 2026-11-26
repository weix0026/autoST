"""Score every fold's model on its own test fires -- once.

    python -m autost.evaluate [--folds 1,2] [--attribution]

Each model is scored at the threshold chosen on its VALIDATION fold (saved
with the model). The sweep over thresholds is also written, for the curves,
but it is the test set looked at many times, so the headline numbers are the
ones at the chosen threshold.

Writes to <data_root>/evaluation/:
    per_fold.csv        each fold at its chosen threshold, spread region and front band
    pooled.csv          all test folds pooled cell by cell (every fire tested exactly once)
    summary.csv         mean and standard deviation over folds
    sweep.csv           every fold at every threshold
    attribution_*.csv   Integrated Gradients per channel (with --attribution)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from . import config as C
from . import folds as FO
from .data import Crops, manifest
from .metrics import Counter, front_band, scores_from_counts
from .model import FireSpreadAutoSTNet


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"]
    # checkpoints without the setting are the first (bottleneck) design
    fusion = ck["config"].get("model", {}).get("fusion", "bottleneck")
    n_s = len(ck["static_names"])
    n_in = sd["enc3d_1.0.weight"].shape[1]
    m = FireSpreadAutoSTNet(static_channels=n_s,
                            dynamic_channels=n_in - (n_s if fusion == "input" else 0),
                            t_sequence=int(ck["config"]["crops"]["sequence"]), fusion=fusion)
    m.load_state_dict(sd)
    return m.to(device).eval(), ck


def predict_counters(model, loader, device, band_k):
    spread_c, band_c = Counter(), Counter()
    with torch.no_grad():
        for (xs, xd), y in loader:
            p = torch.sigmoid(model(xs.to(device), xd.to(device)).float()).cpu().numpy()[:, 0]
            fire = xd[:, -1, 0].numpy() > 0
            t = y.numpy()[:, 0]
            spread_c.add(p, t, ~fire)
            band_c.add(p, t, front_band(fire, band_k))
    return spread_c, band_c


def integrated_gradients(model, loader, device, steps, max_crops):
    """Mean IG attribution per channel over spread-region cells (zero baseline)."""
    s_sum = d_sum = None
    count, seen = 0.0, 0
    for (xs, xd), _ in loader:
        if seen >= max_crops:
            break
        xs, xd = xs.to(device), xd.to(device)
        gs, gd = torch.zeros_like(xs), torch.zeros_like(xd)
        for a in torch.linspace(1.0 / steps, 1.0, steps, device=device):
            s = (a * xs).requires_grad_(True)
            d = (a * xd).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            model(s, d).sum().backward()
            gs += s.grad.detach(); gd += d.grad.detach()
        spread = (xd[:, -1, 0] == 0).float()                         # (B,H,W)
        a_s = (xs * gs / steps * spread[:, None]).sum(dim=(0, 2, 3))
        a_d = (xd * gd / steps * spread[:, None, None]).sum(dim=(0, 1, 3, 4))
        s_sum = a_s if s_sum is None else s_sum + a_s
        d_sum = a_d if d_sum is None else d_sum + a_d
        count += float(spread.sum()) * 1.0
        seen += xs.shape[0]
    return (s_sum / max(count, 1)).cpu().numpy(), (d_sum / max(count * xd.shape[1], 1)).cpu().numpy()


def _row(fold, region, counter, thr, extra=None):
    r = {"fold": fold, "region": region, "threshold": thr, **counter.scores(thr),
         "auc": counter.auc(), "pr_auc": counter.pr_auc(), "prevalence": counter.prevalence}
    r.update(extra or {})
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--attribution", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    cfg = C.load(a.data_config, a.config)
    ev = cfg.model["evaluation"]
    n = int(cfg.model["folds"]["n"])
    folds = range(1, n + 1) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    thresholds = [float(t) for t in ev["thresholds"]]
    out = cfg.data_path(ev["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    models = cfg.data_path(cfg.model["training"]["out_dir"])
    man = manifest(cfg)
    table = FO.load(cfg)

    per_fold, sweep, attr = [], [], []
    pooled_s, pooled_b = Counter(), Counter()
    for k in folds:
        path = models / f"{cfg.model['training']['name']}_fold{k}.pt"
        if not path.exists():
            print(f"fold {k}: no model ({path}) -- skipped")
            continue
        model, ck = load_model(path, device)
        _, _, test_f = FO.roles(table, k, n)
        ds = Crops(cfg, test_f, man)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
        sc, bc = predict_counters(model, dl, device, int(ev["band_k"]))
        thr = float(ck["threshold"])
        info = {"test_fires": len(test_f), "test_crops": len(ds), "epoch": ck["epoch"],
                "validation_fold": ck["validation_fold"], "validation_f1": ck["validation"]["f1"]}
        per_fold += [_row(k, "spread", sc, thr, info), _row(k, "front_band", bc, thr, info)]
        for t in thresholds:
            sweep += [_row(k, "spread", sc, t), _row(k, "front_band", bc, t)]
        pooled_s.pos += sc.pos; pooled_s.neg += sc.neg
        pooled_b.pos += bc.pos; pooled_b.neg += bc.neg
        print(f"fold {k}: {len(test_f)} fires, {len(ds)} crops, threshold {thr:.2f} -> "
              f"F1 {sc.scores(thr)['f1']:.4f}, AUC {sc.auc():.4f}", flush=True)
        if a.attribution or ev.get("attribution"):
            rng = np.random.default_rng(int(cfg.model["seed"]) + 1000 + k)
            idx = np.sort(rng.choice(len(ds), min(len(ds), int(ev["ig_crops"])), replace=False))
            ig_dl = DataLoader(Subset(ds, idx.tolist()), batch_size=8, shuffle=False)
            s_attr, d_attr = integrated_gradients(model, ig_dl, device, int(ev["ig_steps"]), len(idx))
            attr += [{"fold": k, "kind": "static", "channel": c, "attribution": float(v)}
                     for c, v in zip(ck["static_names"], s_attr)]
            attr += [{"fold": k, "kind": "dynamic", "channel": c, "attribution": float(v)}
                     for c, v in zip(ck["dynamic_names"], d_attr)]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not per_fold:
        print("no folds evaluated")
        return
    pf = pd.DataFrame(per_fold)
    pf.to_csv(out / "per_fold.csv", index=False)
    pd.DataFrame(sweep).to_csv(out / "sweep.csv", index=False)
    cols = ["f1", "precision", "recall", "specificity", "npv", "mcc", "tss", "auc", "pr_auc", "prevalence"]
    summ = pf.groupby("region")[cols].agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.reset_index().to_csv(out / "summary.csv", index=False)
    # Pooled: one threshold per fold was chosen on validation; pooling uses each
    # fold's own counts at its own threshold, summed.
    tot = {}
    for region in ("spread", "front_band"):
        r = pf[pf["region"] == region]
        tp, fp, fn, tn = (int(r[c].sum()) for c in ("tp", "fp", "fn", "tn"))
        tot[region] = {**scores_from_counts(tp, fp, fn, tn),
                       "auc": (pooled_s if region == "spread" else pooled_b).auc(),
                       "pr_auc": (pooled_s if region == "spread" else pooled_b).pr_auc(),
                       "prevalence": (pooled_s if region == "spread" else pooled_b).prevalence}
    pd.DataFrame(tot).T.reset_index(names="region").to_csv(out / "pooled.csv", index=False)
    if attr:
        ad = pd.DataFrame(attr)
        ad.to_csv(out / "attribution_per_fold.csv", index=False)
        ad.groupby(["kind", "channel"])["attribution"].agg(["mean", "std"]).reset_index() \
          .sort_values("mean", key=np.abs, ascending=False).to_csv(out / "attribution_summary.csv", index=False)
    print("\nmean over folds (spread region, chosen thresholds):")
    print(summ.loc[["spread"]].T.to_string())
    print("\npooled over all test fires:")
    print(json.dumps({k: {m: round(v, 4) if isinstance(v, float) else v for m, v in d.items()}
                      for k, d in tot.items()}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
