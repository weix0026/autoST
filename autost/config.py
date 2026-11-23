"""Reading the two YAML files and resolving paths.

Every data path is relative to ``data_root`` (configs/data.yaml, or the
environment variable AUTOST_DATA_ROOT): the folder where the published crops
and models were unzipped. Paths to files shipped with the code (the fire
list, the folds) are relative to the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CFG = REPO / "configs" / "data.yaml"
DEFAULT_MODEL_CFG = REPO / "configs" / "autost.yaml"


def _read(path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Config:
    data: Dict[str, Any]
    model: Dict[str, Any]

    @property
    def data_root(self) -> Path:
        root = os.environ.get("AUTOST_DATA_ROOT") or self.data["data_root"]
        p = Path(root)
        return p if p.is_absolute() else (REPO / p).resolve()

    def data_path(self, rel) -> Path:
        """A path under data_root (absolute paths are returned as they are)."""
        p = Path(rel)
        return p if p.is_absolute() else self.data_root / p

    def repo_path(self, rel) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO / p


def load(data_cfg=None, model_cfg=None) -> Config:
    return Config(_read(data_cfg or DEFAULT_DATA_CFG), _read(model_cfg or DEFAULT_MODEL_CFG))
