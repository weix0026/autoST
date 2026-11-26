"""Attribution of groups of channels (run after `python -m autost.evaluate --attribution`).

    python -m autost.groups      -> <data_root>/evaluation/attribution_groups.csv

Integrated Gradients is additive: the attributions of all channels sum to the
difference between the predicted logit and the baseline logit. The attribution
of a group of channels is therefore the sum of its channels' attributions, and
that total does not depend on how the model divides credit among correlated
members (e.g. temperature and VPD, r = 0.89; canopy cover and bulk density,
r = 0.89). Each group holds channels whose zero (the IG baseline) means the same
thing, so the total has one meaning.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C

FM = [f"FM{i}" for i in range(1, 13)] + ["FM13+"]
ASPECT = [f"Aspect {d}" for d in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")]
WIND_DIR = [f"Wind {d}" for d in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")]
LINES = ["Hand Line", "Dozer Line", "Other Line"]
CANOPY = ["Canopy Cover", "Canopy Base Height", "Crown Bulk Density"]

# name: (channels, what zero means for every member)
GROUPS = {
    "fire weather": (["Wind Speed", "Burning Index", "Temperature", "VPD"],
                     "calm, cold, saturated air, no fire danger"),
    "fuel and soil wetness": (["Fuel Moisture 100hr", "Soil Anomaly", "Precipitation"], "driest state"),
    "firelines": (LINES, "no line"),
    "fire barriers": (LINES + ["Roads", "PODs"], "no barrier"),
    "past fuel disturbance": (["Fuel Treatment", "Recent Burn"], "none"),
    "canopy fuels": (CANOPY, "no canopy"),
    "surface fuel type": (FM, "no fuel class (not a real state: compare classes only)"),
    "terrain": (["DEM", "Slope"] + ASPECT, "flat ground at sea level"),
    "wind direction": (WIND_DIR, "no direction"),
    "current fire": (["Fire Today", "Distance to Fire"], "mixed: no fire / on the fire edge"),
}


def group_totals(per_fold: pd.DataFrame) -> pd.DataFrame:
    """per_fold: attribution_per_fold.csv (fold, kind, channel, attribution)."""
    w = per_fold.pivot_table(index="channel", columns="fold", values="attribution")
    rows = []
    for name, (chans, zero) in GROUPS.items():
        missing = [c for c in chans if c not in w.index]
        if missing:
            raise KeyError(f"group {name!r}: channels not in the attribution file: {missing}")
        s = w.loc[chans].sum()
        rows.append({"group": name, "channels": "; ".join(chans), "zero_means": zero,
                     "mean": s.mean(), "sd": s.std(), "median": s.median(),
                     "positive_folds": int((s > 0).sum()), "negative_folds": int((s < 0).sum()),
                     **{f"fold{k}": s[k] for k in s.index}})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = C.load(a.data_config, a.config)
    ev = cfg.data_path(cfg.model["evaluation"]["out_dir"])
    out = group_totals(pd.read_csv(ev / "attribution_per_fold.csv"))
    out.to_csv(ev / "attribution_groups.csv", index=False)
    for r in out.itertuples():
        print(f"{r.group:24s} {r.mean:+.3f} ± {r.sd:.3f}  (+{r.positive_folds} / -{r.negative_folds} folds)")
    print(f"-> {ev / 'attribution_groups.csv'}")


if __name__ == "__main__":
    main()
