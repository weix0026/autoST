"""The crops as a PyTorch dataset, selected by fire."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def manifest(cfg) -> pd.DataFrame:
    p = cfg.data_path(cfg.model["crops"]["dir"]) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p}: unzip the published crops.zip into data_root "
                                f"(configs/data.yaml; docs/DATA.md)")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


class Crops(Dataset):
    def __init__(self, cfg, fires=None, man: pd.DataFrame = None):
        man = manifest(cfg) if man is None else man
        if fires is not None:
            man = man[man["incident_c"].isin(set(fires))]
        self.rows = man.reset_index(drop=True)
        self.dir = cfg.data_path(cfg.model["crops"]["dir"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with np.load(self.dir / self.rows.at[i, "file"]) as z:
            s = torch.from_numpy(z["static"].astype(np.float32))
            d = torch.from_numpy(z["dynamic"].astype(np.float32))
            y = torch.from_numpy(z["target"].astype(np.float32))
        return (s, d), y
