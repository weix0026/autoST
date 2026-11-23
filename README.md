# AutoST: next-day wildfire spread

This is the code to train, test and apply AutoST from the published training
crops. AutoST is a deep-learning model that predicts where a wildfire will
have burned by its next mapped perimeter. It reads the last three daily
perimeters, terrain, fuels, roads, past fires and treatments, firelines, and
the weather in between.

- **Data.** 49,540 crops of 64 x 64 cells (30 m) from 311 western US fires of
  2020-2023 and their 5,616 daily infrared perimeters (NIROPS), with the 10
  trained models. They are a separate data record (Zenodo; `docs/DATA.md`
  lists every channel and its source).
- **Model.** A 3-D U-Net over the input days, with a transformer at the
  bottleneck and attention on the skip connections. It has 33 static channels
  and 18 daily channels.
- **Testing.** 10-fold cross-validation by fire, with whole fires in each fold
  (`crops/folds.csv` in the data record). For each test fold, a model is trained on eight folds,
  the epoch and the decision threshold are chosen on the next fold, and the
  test fold is scored once.

The crops were made from public sources (and two others, see `docs/DATA.md`)
as described in the paper's Methods. The code that made them, and whole-fire
prediction from overlapping crops, are not part of this package.

## Install

```
conda env create -f environment.yml
conda activate autost
pip install -e .
pytest            # about 20 s: the pieces, then the whole chain on made-up crops
```

## Run

```
# 0. Download crops.zip and models.zip from the data record and unzip both
#    into one folder. Set data_root in configs/data.yaml to that folder
#    (or set the environment variable AUTOST_DATA_ROOT).

# 1. Score the published models on their test folds (under 1 h on a GPU)
python -m autost.evaluate                    # --attribution adds Integrated Gradients (slow)
#    -> <data_root>/evaluation/per_fold.csv, pooled.csv, summary.csv, sweep.csv
#    Compare with the published evaluation/*.csv.

# 2. Or train the 10 models yourself (about 15 h on one RTX 5000 Ada; resumable)
#    Set training.out_dir in configs/autost.yaml to a new folder first, or the
#    published models are overwritten.
python -m autost.train
python -m autost.train --folds 3             # one fold

# 3. The paper's figures 3-9 (architecture, correlation, crop example,
#    training curves, attribution, threshold curves, 100-crop grid)
python -m autost.figures                     # --only 6,8 for some of them
#    -> <data_root>/figures
```

A quick check of the training loop on a few crops:
`python -m autost.train --folds 1 --epochs 1 --max-crops 200`.

## Reproducibility

Scoring the published models on the published crops gives the paper's test
scores (checked: fold 1 re-scored from the downloaded files matches exactly).
Retraining sets PyTorch's deterministic mode, but a few CUDA kernels have no
deterministic version and mixed precision is used, so retrained weights
differ slightly between runs and more between GPUs. Report test scores as the
mean and spread over folds. `environment.yml` gives the library versions
tested. Each model file holds the configuration it was trained with, its
epoch, its threshold and its validation scores.

## Code

| file | what it does |
|---|---|
| `autost/data.py` | the crops as a PyTorch dataset, selected by fire |
| `autost/channels.py` | channel names and how each was scaled in the crops |
| `autost/folds.py` | reads the fold table; train / validation / test roles |
| `autost/model.py`, `losses.py` | AutoST and its loss |
| `autost/train.py` | training, one model per test fold |
| `autost/metrics.py` | pooled scores (F1, precision, recall, AUC, PR-AUC, MCC, TSS) |
| `autost/evaluate.py` | test scores per fold and pooled; optional channel attribution |
| `autost/figures.py` | the paper's figures 3-9 from a finished run |

## Citation

See `CITATION.cff`. Please cite the paper, this software and the data record.

## Licence

MIT (`LICENSE`). The data sources keep their own terms (`docs/DATA.md`).
