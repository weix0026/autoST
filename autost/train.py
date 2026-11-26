"""Train AutoST, one model per test fold.

    python -m autost.train                    # all 10 folds
    python -m autost.train --folds 3          # one fold
    python -m autost.train --folds 1 --epochs 1 --max-crops 200   # a quick check

For test fold k: train on 8 folds, validate on fold k+1, never touch fold k
(autost/folds.py). After every epoch the validation crops are scored, pooled
over all their cells, at every threshold in configs/autost.yaml; the epoch
with the best validation F1 is kept, together with the threshold that gave
it. Evaluation then scores the test fold once, at that threshold.

Each fold writes, under <data_root>/models/:
    <name>_fold<k>.pt         best epoch: weights + threshold + validation scores
    <name>_fold<k>_last.pt    the latest epoch, to resume an interrupted run
    <name>_fold<k>_log.csv    one row per epoch
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # repeatable cuBLAS

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import channels as CH
from . import config as C
from . import folds as FO
from .data import Crops, manifest
from .losses import combined_loss
from .metrics import Counter
from .model import FireSpreadAutoSTNet


def set_determinism(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        # Some CUDA kernels (e.g. 3-D max-pool backward) have no deterministic
        # version; PyTorch then warns rather than stops. Repeated GPU runs can
        # therefore differ in the last digits; docs/REPRODUCING.md says by how much.
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        # The fused attention kernels hang on the first batch on some CUDA builds.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def loaders(cfg, train_fires, val_fires, max_crops=None, seed=0):
    tr = cfg.model["training"]
    man = manifest(cfg)
    train_ds, val_ds = Crops(cfg, train_fires, man), Crops(cfg, val_fires, man)
    if max_crops:
        rng = np.random.default_rng(seed)
        for ds in (train_ds, val_ds):
            if len(ds) > max_crops:
                keep = np.sort(rng.choice(len(ds), max_crops, replace=False))
                ds.rows = ds.rows.iloc[keep].reset_index(drop=True)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=int(tr["batch_size"]), num_workers=int(tr["num_workers"]),
              worker_init_fn=_seed_worker, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, shuffle=True, generator=g, **kw),
            DataLoader(val_ds, shuffle=False, **kw))


def run_epoch(model, loader, los, device, optimizer=None, scaler=None, amp=False):
    """One pass. Returns (mean loss, Counter of spread-region cells)."""
    train = optimizer is not None
    model.train(train)
    total, n, counter = 0.0, 0, Counter()
    with torch.set_grad_enabled(train):
        for (xs, xd), y in loader:
            xs, xd, y = xs.to(device), xd.to(device), y.to(device)
            spread = (xd[:, -1, 0] == 0).unsqueeze(1).float()
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                out = model(xs, xd)
                loss, *_ = combined_loss(out.float(), y, spread, **los)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                counter.add(torch.sigmoid(out.float()).cpu().numpy(), y.cpu().numpy(),
                            spread.bool().cpu().numpy())
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
    return total / max(n, 1), counter


def train_fold(cfg, fold: int, epochs=None, max_crops=None, fresh=False, device=None):
    tr, los = cfg.model["training"], cfg.model["loss"]
    n_folds = int(cfg.model["folds"]["n"])
    seed = int(cfg.model["seed"]) + fold
    set_determinism(seed, bool(tr.get("deterministic", True)))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tr.get("amp", True))
    thresholds = [float(t) for t in cfg.model["evaluation"]["thresholds"]]
    epochs = int(epochs or tr["epochs"])

    train_f, val_f, test_f = FO.roles(FO.load(cfg), fold, n_folds)
    assert not (set(train_f) & set(test_f)) and not (set(val_f) & set(test_f))
    train_dl, val_dl = loaders(cfg, train_f, val_f, max_crops, seed)
    print(f"=== fold {fold}: train {len(train_f)} fires / {len(train_dl.dataset)} crops, "
          f"validation fold {fold % n_folds + 1}: {len(val_f)} fires / {len(val_dl.dataset)} crops, "
          f"test: {len(test_f)} fires (not used here) ===", flush=True)

    firelines = bool(cfg.model["channels"]["firelines"])
    (xs, xd), _ = train_dl.dataset[0]
    model = FireSpreadAutoSTNet(static_channels=xs.shape[0], dynamic_channels=xd.shape[1],
                                t_sequence=xd.shape[0],
                                fusion=cfg.model.get("model", {}).get("fusion", "input")).to(device)
    if xs.shape[0] != len(CH.static_names(firelines)):
        raise ValueError(f"crops have {xs.shape[0]} static channels; configs say "
                         f"{len(CH.static_names(firelines))} (firelines={firelines}). Set channels.firelines to match the crops.")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=float(tr["lr_factor"]),
                                                       patience=int(tr["lr_patience"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    out = cfg.data_path(tr["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tr['name']}_fold{fold}"
    best_path, last_path, log_path = out / f"{stem}.pt", out / f"{stem}_last.pt", out / f"{stem}_log.csv"

    start, best_f1, since_best = 0, -1.0, 0
    if last_path.exists() and not fresh:
        st = torch.load(last_path, map_location=device, weights_only=False)
        if (st.get("max_crops"), st.get("epochs")) != (max_crops, epochs):
            raise SystemExit(f"{last_path} was saved by a run with max_crops={st.get('max_crops')}, "
                             f"epochs={st.get('epochs')}; this run has max_crops={max_crops}, "
                             f"epochs={epochs}. Use --fresh, or delete the fold's files.")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        start, best_f1, since_best = st["epoch"] + 1, st["best_f1"], st["since_best"]
        torch.set_rng_state(st["rng_cpu"]); train_dl.generator.set_state(st["rng_loader"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"  resumed after epoch {start}", flush=True)
    elif fresh or not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_f1_best", "val_threshold",
                                    "val_f1_at_0.5", "val_auc", "lr", "seconds"])

    los_kw = {k: float(los[k]) for k in ("pos_weight", "bce_weight", "dice_weight", "focal_weight", "gamma")}
    for epoch in range(start, epochs):
        t0 = time.time()
        tl, _ = run_epoch(model, train_dl, los_kw, device, opt, scaler, amp)
        vl, counter = run_epoch(model, val_dl, los_kw, device, amp=False)   # float32, like evaluate
        sched.step(vl)
        thr, f1 = counter.best_f1(thresholds)
        lr = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{tl:.6f}", f"{vl:.6f}", f"{f1:.6f}", thr,
                                    f"{counter.scores(0.5)['f1']:.6f}", f"{counter.auc():.6f}",
                                    f"{lr:.3g}", f"{secs:.0f}"])
        print(f"  epoch {epoch + 1:3d}  train {tl:.4f}  val {vl:.4f}  val F1 {f1:.4f} at {thr:.2f}  "
              f"AUC {counter.auc():.4f}  {secs:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1, since_best = f1, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "threshold": thr,
                        "validation": {"f1": f1, **counter.scores(thr), "auc": counter.auc(),
                                       "pr_auc": counter.pr_auc(), "prevalence": counter.prevalence},
                        "fold": fold, "validation_fold": fold % n_folds + 1,
                        "static_names": CH.static_names(firelines), "dynamic_names": CH.DYNAMIC_NAMES,
                        "config": cfg.model}, best_path)
        else:
            since_best += 1
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "since_best": since_best,
                    "max_crops": max_crops, "epochs": epochs,
                    "rng_cpu": torch.get_rng_state(), "rng_loader": train_dl.generator.get_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   last_path)
        if since_best > int(tr["early_stop_patience"]):
            print(f"  no better validation F1 for {since_best} epochs: stopped", flush=True)
            break
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"  fold {fold}: best epoch {best['epoch']}, validation F1 {best['validation']['f1']:.4f} "
          f"at threshold {best['threshold']:.2f}", flush=True)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", default="all", help="'all' or comma-separated fold numbers")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-crops", type=int, default=None, help="cap train and validation crops (quick checks)")
    ap.add_argument("--fresh", action="store_true", help="ignore a saved _last.pt and start over")
    a = ap.parse_args(argv)
    cfg = C.load(a.data_config, a.config)
    n = int(cfg.model["folds"]["n"])
    folds = range(1, n + 1) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    summary = {}
    for k in folds:
        b = train_fold(cfg, k, a.epochs, a.max_crops, a.fresh)
        summary[k] = {"epoch": b["epoch"], "threshold": b["threshold"], "validation_f1": b["validation"]["f1"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
