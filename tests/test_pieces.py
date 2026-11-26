"""Checks of the pieces: channel scaling, folds, scores."""
import numpy as np
import pandas as pd
import pytest

from autost import channels as CH
from autost import folds as FO
from autost.metrics import Counter, auc_exact, front_band, scores_from_counts


def test_every_channel_is_between_0_and_1_at_landfire_maxima():
    for key, top in (("cc", 95), ("cbh", 100), ("cbd", 45)):
        assert CH.scale(np.array([top]), key).max() <= 1.0
    assert CH.scale(np.array([1e9]), "dem").max() == 1.0          # clipped


def test_soft_directions_sum_to_one_and_flat_ground_is_zero():
    d = CH.directions_soft(np.array([[0.0, 90.0], [200.0, 359.0]]), valid=np.array([[1, 1], [1, 0]]))
    assert d.shape == (8, 2, 2)
    assert np.allclose(d[:, 0, 0].sum(), 1) and d[:, 0, 0].argmax() == 0
    assert d[:, 0, 1].argmax() == 2                                # east
    assert np.allclose(d[:, 1, 1], 0)                              # not valid -> no direction


def test_fuel_one_hot_puts_13_and_non_burnable_together():
    f = CH.fuel_one_hot(np.array([[1, 12], [13, 98]]))
    assert f[0, 0, 0] == 1 and f[11, 0, 1] == 1 and f[12, 1, 0] == 1 and f[12, 1, 1] == 1
    assert np.allclose(f.sum(axis=0), 1)


def test_channel_counts():
    assert len(CH.static_names(False)) == 30
    assert len(CH.static_names(True)) == 33
    assert len(CH.DYNAMIC_NAMES) == 18


def test_folds_are_fixed_balanced_and_disjoint():
    fires = pd.DataFrame({"incident_c": [f"F{i:03d}" for i in range(53)],
                          "n_perimeters": [str(3 + (i * 7) % 11) for i in range(53)]})
    a, b = FO.assign(fires, 10), FO.assign(fires.sample(frac=1, random_state=1), 10)
    pd.testing.assert_frame_equal(a, b)                            # order of input does not matter
    load = a.merge(fires.astype({"n_perimeters": int})).groupby("fold")["n_perimeters"].sum()
    assert load.max() - load.min() <= 11
    for k in range(1, 11):
        tr, va, te = FO.roles(a, k, 10)
        assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
        assert len(tr) + len(va) + len(te) == 53


def test_pooled_counter_matches_exact_scores():
    rng = np.random.default_rng(0)
    t = rng.random(20000) < 0.1
    p = np.clip(t * 0.3 + rng.random(20000) * 0.7, 0, 0.999)
    c = Counter()
    for k in range(0, 20000, 1000):                                # in batches, pooled
        c.add(p[k:k + 1000], t[k:k + 1000])
    pred = p >= 0.5
    exact = scores_from_counts((pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum(), (~pred & ~t).sum())
    assert c.scores(0.5)["f1"] == pytest.approx(exact["f1"])
    assert c.auc() == pytest.approx(auc_exact(p, t), abs=2e-3)


def test_front_band():
    fire = np.zeros((9, 9), bool)
    fire[4, 4] = True
    b = front_band(fire, 1)
    assert b.sum() == 8 and not b[4, 4]


def test_group_totals_add_up_channel_attributions():
    """Every group's channels exist in the model, and a group total is the sum of its channels."""
    import pandas as pd
    from autost import channels as CH
    from autost.groups import GROUPS, group_totals
    names = CH.static_names(True) + list(CH.DYNAMIC_NAMES)
    rows = [{"fold": k, "kind": "x", "channel": c, "attribution": 0.01 * (i + 1) * k}
            for k in (1, 2) for i, c in enumerate(names)]
    out = group_totals(pd.DataFrame(rows)).set_index("group")
    for g, (chans, _) in GROUPS.items():
        expect = sum(0.01 * (names.index(c) + 1) for c in chans)
        assert abs(out.loc[g, "fold1"] - expect) < 1e-9
        assert abs(out.loc[g, "fold2"] - 2 * expect) < 1e-9
