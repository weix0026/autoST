"""The whole chain on small made-up crops, no data download:
crops -> train (1 epoch) -> evaluate -> figures."""
import numpy as np
import pandas as pd
import pytest
import yaml

from autost import channels as CH
from autost import config as C

N_FIRES, CROPS_PER_FIRE, S = 4, 6, 64


def _make_crops(root, fires):
    """Crops in the published format: float16 static (33,64,64), dynamic
    (3,18,64,64), target (1,64,64); a fire disc that grows next day."""
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:S, 0:S]
    man = []
    for inc in fires:
        (root / "crops" / inc).mkdir(parents=True)
        for k in range(CROPS_PER_FIRE):
            static = rng.random((len(CH.static_names(True)), S, S)).astype(np.float16)
            dynamic = rng.random((3, len(CH.DYNAMIC_NAMES), S, S)).astype(np.float16)
            r = 8 + 2 * k
            for t in range(3):
                dynamic[t, 0] = ((rows - 32) ** 2 + (cols - 32) ** 2) < (r + t) ** 2
            target = (((rows - 32) ** 2 + (cols - 32) ** 2) < (r + 6) ** 2)[None].astype(np.float16)
            f = f"{inc}/2021-08-0{k + 1}_00.npz"
            np.savez_compressed(root / "crops" / f, static=static, dynamic=dynamic, target=target)
            man.append({"file": f, "incident_c": inc, "date": f"2021-08-0{k + 1}",
                        "next_date": f"2021-08-0{k + 2}", "row": 32, "col": 32, "filled": ""})
    pd.DataFrame(man).to_csv(root / "crops" / "manifest.csv", index=False)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data = yaml.safe_load(open(C.DEFAULT_DATA_CFG))
    model = yaml.safe_load(open(C.DEFAULT_MODEL_CFG))
    data["data_root"] = str(tmp / "data")
    model["folds"]["csv"] = str(tmp / "folds.csv")
    model["folds"]["n"] = N_FIRES
    model["training"].update(epochs=1, batch_size=4, amp=False, num_workers=0)
    fires = [f"XX-TST-{k:03d}_Fire{k}" for k in range(N_FIRES)]
    _make_crops(tmp / "data", fires)
    # 4 fires, 4 folds: fold k tests fire k, validates on fire k+1
    pd.DataFrame({"incident_c": fires, "fold": range(1, N_FIRES + 1)}).to_csv(model["folds"]["csv"], index=False)
    return C.Config(data, model)


def test_whole_chain(cfg, monkeypatch):
    import torch
    from autost import evaluate, figures, train
    from autost.data import Crops

    ds = Crops(cfg)
    (s, d), y = ds[0]
    assert s.shape == (33, S, S) and d.shape == (3, 18, S, S) and y.shape == (1, S, S)

    best = train.train_fold(cfg, 1, epochs=1, device=torch.device("cpu"))
    assert best["validation_fold"] == 2 and 0.0 <= best["threshold"] <= 1.0
    log = pd.read_csv(cfg.data_path("models") / "autost_fold1_log.csv")
    assert len(log) == 1

    monkeypatch.setattr(C, "load", lambda *a, **k: cfg)
    evaluate.main(["--folds", "1"])
    pf = pd.read_csv(cfg.data_path("evaluation") / "per_fold.csv")
    assert set(pf["region"]) == {"spread", "front_band"} and pf["test_fires"].iloc[0] == 1

    figures.main(["--only", "3,4,6,8"])
    out = cfg.data_path("figures")
    for name in ("fig03_architecture", "fig04_correlation", "fig06_training_curves", "fig08_thresholds"):
        assert (out / f"{name}.png").exists(), name
