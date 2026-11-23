"""The model's input channels: their order and how each is scaled.

The published crops were made with exactly these rules. The code that made
them is not part of this package; this file is kept so the values in the
crops can be read: every channel is a raw value divided by its divisor below
and clipped to 0-1. docs/DATA.md lists every channel.
"""
from __future__ import annotations

import numpy as np

# ---- scaling: value / divisor, then clipped to [0, 1] ----------------------
# Divisors are the physical ranges of the data (units in the comments).
DIVISORS = {
    "dem": 4343.0,          # m; highest point in the conterminous US is 4,421 m
    "slope": 90.0,          # degrees
    "cc": 100.0,            # canopy cover, percent (LANDFIRE class midpoints 15..95)
    "cbh": 100.0,           # canopy base height, m x 10 (LANDFIRE max 100 = 10 m)
    "cbd": 45.0,            # canopy bulk density, kg/m3 x 100 (LANDFIRE max 45)
    "wind_speed": 20.0,     # m/s
    "bi": 180.0,            # burning index (NFDRS, fuel model G)
    "tmmx": 320.0,          # daily maximum temperature, K
    "fm100": 30.0,          # 100-hour dead fuel moisture, percent
    "vpd": 8.0,             # mean vapour pressure deficit, kPa (gridMET; western fire days rarely pass 7)
    "precip": 50.0,         # mm/day
    "days_to_next": 10.0,   # days until the next perimeter
    "distance_cells": 100.0,  # distance to the fire, in 30 m cells (100 = 3 km)
}

# Soil-moisture anomaly (standard deviations): values beyond +-4 are fill,
# the rest maps [-3, 3] -> [0, 1].
SOIL_FILL_ABS = 4.0
SOIL_RANGE = (-3.0, 3.0)

# Values used where a required daily weather raster is present but has a
# hole (nodata), in the raw units above.
WEATHER_FILL = {"wind_speed": 5.0, "bi": 50.0, "tmmx": 293.15, "fm100": 20.0, "vpd": 2.0}

# Soft 8-direction encoding (N, NE, E, SE, S, SW, W, NW): a Gaussian of the
# angular distance to each 45-degree sector centre, sigma = 22.5 degrees,
# normalised to sum to 1. The original crops were made this way.
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DIRECTION_SIGMA_DEG = 22.5

# FBFM13 fuel models 1..12 get one channel each; the 13th channel holds fuel
# model 13 AND every non-burnable code (91-99) and nodata, as in the study.
N_FUEL_CHANNELS = 13


def static_names(firelines: bool) -> list:
    names = ["DEM", "Slope"] + [f"Aspect {d}" for d in DIRECTIONS]
    if firelines:
        names += ["Hand Line", "Dozer Line", "Other Line"]
    names += ["Recent Burn", "Fuel Treatment"]
    names += [f"FM{i}" for i in range(1, 13)] + ["FM13+"]
    names += ["Canopy Cover", "Canopy Base Height", "Crown Bulk Density", "Roads", "PODs"]
    return names


DYNAMIC_NAMES = (["Fire Today"] + [f"Wind {d}" for d in DIRECTIONS]
                 + ["Wind Speed", "Burning Index", "Temperature", "Fuel Moisture 100hr",
                    "VPD", "Soil Anomaly", "Precipitation", "Distance to Fire", "Days to Next"])


# ---- encoders --------------------------------------------------------------
def scale(arr, key: str) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32) / DIVISORS[key], 0.0, 1.0)


def soil(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(~np.isfinite(a) | (np.abs(a) > SOIL_FILL_ABS), 0.0, a)
    lo, hi = SOIL_RANGE
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def directions_soft(degrees, valid=None) -> np.ndarray:
    """(8, H, W) soft direction channels summing to 1; all zero where not `valid`."""
    d = np.mod(np.asarray(degrees, dtype=np.float64), 360.0)
    centres = (np.arange(8) * 45.0).reshape(8, *([1] * d.ndim))
    delta = np.abs((centres - d[None] + 180.0) % 360.0 - 180.0)
    w = np.exp(-0.5 * (delta / DIRECTION_SIGMA_DEG) ** 2)
    w /= w.sum(axis=0, keepdims=True)
    if valid is not None:
        w = w * np.asarray(valid, dtype=bool)[None]
    return w.astype(np.float32)


def fuel_one_hot(fbfm13) -> np.ndarray:
    f = np.nan_to_num(np.asarray(fbfm13, dtype=np.float32), nan=0.0).astype(np.int32)
    out = np.zeros((N_FUEL_CHANNELS,) + f.shape, dtype=np.float32)
    for m in range(1, 13):
        out[m - 1] = f == m
    out[12] = (f >= 13) | (f <= 0)
    return out


def distance_to_fire(fire_mask) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    fire = np.asarray(fire_mask) > 0
    if not fire.any():
        return np.ones(fire.shape, dtype=np.float32)
    return scale(distance_transform_edt(~fire), "distance_cells")
