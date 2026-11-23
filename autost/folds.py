"""Which fire is in which fold -- one fixed table, published with the crops.

The table is crops/folds.csv in the data record (columns incident_c, fold).
Whole fires go to one fold, so no fold is ever tested on a fire it trained
on. Fires were dealt out largest first (by number of perimeters), each to the
fold with the fewest perimeters so far; ties go to the lower fold and the
alphabetically first fire (`assign`).

For test fold k the model is trained on the other folds except k+1 (k = 10
uses fold 1), which is held back to choose the epoch and the threshold. The
test fold is used only once, at the end.
"""
from __future__ import annotations

import pandas as pd



def assign(fires: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    df = fires[["incident_c", "n_perimeters"]].copy()
    df["n_perimeters"] = df["n_perimeters"].astype(int)
    df = df.sort_values(["n_perimeters", "incident_c"], ascending=[False, True])
    load = [0] * n_folds
    out = []
    for inc, n in zip(df["incident_c"], df["n_perimeters"]):
        k = min(range(n_folds), key=lambda f: (load[f], f))
        load[k] += n
        out.append((inc, k + 1))
    return pd.DataFrame(out, columns=["incident_c", "fold"]).sort_values("incident_c").reset_index(drop=True)


def roles(fold_table: pd.DataFrame, test_fold: int, n_folds: int):
    """(train fires, validation fires, test fires) for one test fold."""
    val_fold = test_fold % n_folds + 1
    f = fold_table.set_index("incident_c")["fold"]
    test = sorted(f[f == test_fold].index)
    val = sorted(f[f == val_fold].index)
    train = sorted(f[(f != test_fold) & (f != val_fold)].index)
    return train, val, test


def load(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data_path(cfg.model["folds"]["csv"]), dtype={"incident_c": str, "fold": int})

