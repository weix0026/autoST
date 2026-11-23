# The data: crops and trained models

The data record (Zenodo) has two zip files and this guide. Unzip both zip
files into one folder and set `data_root` in the code's `configs/data.yaml`
to that folder (code: GitHub, see `CITATION.cff`):

```
<data_root>/
  crops/manifest.csv                 one row per crop
  crops/folds.csv                    which fold each fire is in (columns incident_c, fold)
  crops/<incident>/<date>_<k>.npz    49,540 crops, 311 fires
  models/autost_fold<k>.pt           the 10 trained models (k = 1..10)
  models/autost_fold<k>_log.csv      loss and validation scores per epoch
  evaluation/*.csv                   the test scores in the paper
```

Each of the 311 fires is in exactly one of 10 folds, 31 or 32 fires per
fold. Fires were dealt out largest first (by number of perimeters), each to
the fold with the fewest perimeters so far.

## What a crop is

Each crop is a 64 x 64 window of 30 m cells (1.92 km on a side) on the fire's
own grid in its local UTM zone, centred on a random cell of the fire's edge on
day t. There are 10 crops per usable fire-day (4,954 fire-days). A fire-day is
usable when the two perimeters before it and the one after it exist. One extra
fire-day per fire repeats the last perimeter as "no spread" on the next day,
as in the original study; it is not an observation.

Each `.npz` file (NumPy, compressed) holds three float16 arrays, all in 0-1:

| array | shape | what it is |
|---|---|---|
| `static` | (33, 64, 64) | maps that do not change during the fire (channels below) |
| `dynamic` | (3, 18, 64, 64) | days t-2, t-1 and t, 18 channels each |
| `target` | (1, 64, 64) | 1 = inside the next perimeter (day t+1), else 0 |

The target includes the area already burning on day t. Scores count only
cells NOT burning on day t (dynamic channel 0 of day t is 0): those are the
cells where spread can be right or wrong.

`crops/manifest.csv` columns: `file` (path under `crops/`), `incident_c`
(NIROPS incident id), `date` (day t), `next_date` (day t+1), `row`, `col`
(the crop centre on the fire's 30 m grid), `filled` (weather layers that were
missing and filled with a typical value; empty for every published crop).

Dates are the UTC dates of the NIROPS infrared flights. Flights are at night,
so a perimeter dated t was mapped on the evening of t-1 local time; the weather
for day t is the mean of the days from t to the day before the next perimeter.

## Channels

Every value is the raw value divided by a fixed divisor and clipped to 0-1
(`autost/channels.py`). Yes/no layers are 1 inside, 0 outside. Line layers
(roads, POD boundaries, firelines) are lines buffered 30 m on each side.
Wind and aspect directions are spread over 8 compass channels that sum to 1
(a Gaussian of the angle to each 45-degree sector, sigma 22.5 degrees);
flat ground has all 8 aspect channels at 0.

**Static channels**

| # | name | source | raw value -> crop value |
|---|---|---|---|
| 0 | DEM | USGS 3DEP 1 arc-second | elevation m / 4343 |
| 1 | Slope | from the DEM (Horn 3x3, in metres on the UTM grid) | degrees / 90 |
| 2-9 | Aspect N, NE, E, SE, S, SW, W, NW | from the DEM | 8 soft direction channels |
| 10 | Hand Line | fireline engagement archive 2017-2024 | yes/no, line +-30 m |
| 11 | Dozer Line | same | yes/no, line +-30 m |
| 12 | Other Line | same | yes/no, line +-30 m |
| 13 | Recent Burn | NIFC InterAgency Fire Perimeter History | yes/no: burned in the 20 years before the fire, ending 30 days before it |
| 14 | Fuel Treatment | USFS FACTS hazardous fuel treatments (Forest Service land only) | yes/no: accomplished mechanical, fire or other treatment in the 20 years before the fire |
| 15-26 | FM1 ... FM12 | LANDFIRE FBFM13 | yes/no for fuel models 1-12 |
| 27 | FM13+ | LANDFIRE FBFM13 | yes/no: fuel model 13, non-burnable (91-99) or no data |
| 28 | Canopy Cover | LANDFIRE CC | percent / 100 |
| 29 | Canopy Base Height | LANDFIRE CBH | m x 10 / 100 |
| 30 | Crown Bulk Density | LANDFIRE CBD | kg/m3 x 100 / 45 |
| 31 | Roads | HERE road network | yes/no, line +-30 m |
| 32 | PODs | National Potential Operational Delineations | yes/no, boundary +-30 m |

Firelines: every line of the fire's year in the archive within 1 km of the
fire's final perimeter, as in the original study; the same on every day of
the fire. LANDFIRE: the newest release before the fire's year that the
LANDFIRE Product Service offers, LF2016 for fires of 2020-2022 and LF2022 for
2023.

**Dynamic channels** (for each of days t-2, t-1, t)

| # | name | source | raw value -> crop value |
|---|---|---|---|
| 0 | Fire Today | NIROPS perimeter of that day | yes/no: inside the perimeter |
| 1-8 | Wind N ... NW | gridMET wind direction (vector mean) | 8 soft direction channels, direction the wind blows FROM |
| 9 | Wind Speed | gridMET | m/s / 20 |
| 10 | Burning Index | gridMET (NFDRS) | BI / 180 |
| 11 | Temperature | gridMET daily maximum | K / 320 |
| 12 | Fuel Moisture 100hr | gridMET | percent / 30 |
| 13 | VPD | gridMET mean vapour pressure deficit | kPa / 8 |
| 14 | Soil Anomaly | TopoFire (University of Montana) | standard deviations: [-3, 3] -> [0, 1]; no data = 0.5 (normal) |
| 15 | Precipitation | gridMET | mm/day / 50 |
| 16 | Distance to Fire | from Fire Today | distance in 30 m cells / 100 (1 = 3 km or more) |
| 17 | Days to Next | NIROPS dates | days to the next perimeter / 10 |

Weather is the mean of the days from that perimeter's date to the day before
the next perimeter, put on the 30 m grid bilinearly. Holes in a weather layer
were filled with wind 5 m/s, BI 50, 293.15 K, 100-h moisture 20 % and VPD
2 kPa.

## Trained models

`models/autost_fold<k>.pt` is the model for test fold k: trained on eight
folds, with fold k+1 (fold 1 after fold 10) used to choose the epoch and the
probability threshold. Each file is a PyTorch dictionary with `model` (the
weights), `epoch`, `threshold`, `validation` (the validation scores),
`fold`, `validation_fold`, the channel names and the full `config`.
`autost.evaluate.load_model` loads one.

## Licences of the sources

The crops are derived from the sources above. NIROPS, 3DEP, LANDFIRE,
gridMET, TopoFire, NIFC fire history, FACTS and the national PODs are public.
The fireline engagement archive is published as [CITE THE FIRELINE DATASET].
The road channel is derived from the licensed HERE road network.
