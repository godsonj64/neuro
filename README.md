# NOD Neuroencoding: CORnet-S → fMRI Voxels and EEG TRF

Colab-ready research repository for neural encoding with the OpenNeuro NOD datasets:

- **fMRI:** `ds004496`
- **EEG:** `ds005811`
- **Base visual model:** pretrained **CORnet-S**, a biologically motivated recurrent model of the primate ventral visual stream.
- **Encoding models:**
  - fMRI: voxel-wise Ridge regression, `CORnet-S features → voxel activity`.
  - EEG: temporal response function model, `CORnet-S features over lags → millisecond EEG voltage`.

This repository is designed to run on Google Colab with minimal manual setup. It avoids downloading the full datasets by default. Use subject/session/task filters first, then scale up.

---

## 1. Quick Colab Run

Open `notebooks/colab_nod_encoding.ipynb`, or run the same commands in a Colab cell:

```bash
!git clone https://github.com/godsonj64/neuro.git
%cd neuro
!bash scripts/setup_colab.sh
```

Download a small subset first:

```bash
!python -m nod_encoding.download_openneuro --dataset ds004496 --target data/ds004496 --include "sub-01/**" --max-files 200
!python -m nod_encoding.download_openneuro --dataset ds005811 --target data/ds005811 --include "sub-01/**" --max-files 200
```

Extract CORnet-S features from stimulus images:

```bash
!python -m nod_encoding.extract_cornet_features \
  --bids-root data/ds004496 \
  --out features/ds004496_cornets.h5 \
  --layers V1 V2 V4 IT decoder \
  --batch-size 32
```

Train fMRI ridge encoding:

```bash
!python -m nod_encoding.train_fmri_ridge \
  --bids-root data/ds004496 \
  --features features/ds004496_cornets.h5 \
  --out results/fmri_ridge \
  --alpha 100.0
```

Train EEG TRF encoding:

```bash
!python -m nod_encoding.train_eeg_trf \
  --bids-root data/ds005811 \
  --features features/ds005811_cornets.h5 \
  --out results/eeg_trf \
  --tmin -0.1 \
  --tmax 0.6 \
  --alpha 10.0
```

---

## 2. Repository Layout

```text
neuro/
  configs/
    default.yaml
  notebooks/
    colab_nod_encoding.ipynb
  scripts/
    setup_colab.sh
  src/nod_encoding/
    download_openneuro.py
    cornet.py
    extract_cornet_features.py
    bids_utils.py
    fmri.py
    eeg.py
    train_fmri_ridge.py
    train_eeg_trf.py
    metrics.py
    utils.py
  tests/
    test_design_matrices.py
```

---

## 3. Scientific Pipeline

### fMRI encoding

For each stimulus image, CORnet-S generates a feature vector. Let `X ∈ R^{n×p}` be the image-feature matrix and `Y ∈ R^{n×v}` be trial-wise voxel responses. The voxel-wise encoding model solves

```math
\hat{B}=\arg\min_B \|Y-XB\|_F^2 + \alpha \|B\|_F^2.
```

The repository computes cross-validated Pearson correlation and coefficient of determination per voxel.

### EEG temporal response function

For millisecond EEG, the model constructs lagged feature matrices. If `x_t` is the feature vector at time `t`, the TRF uses delays `τ ∈ [tmin,tmax]`:

```math
\hat{y}_t = \sum_{\tau} x_{t-\tau} W_{\tau}.
```

The script fits multi-output Ridge regression to predict sensor voltage over time.

---

## 4. Practical Notes

OpenNeuro datasets can be large. The default downloader supports a subset-oriented workflow. Start with one subject. Increase `--max-files` or remove it after verifying that the pipeline works.

The scripts are intentionally defensive: they search for BIDS-compatible `events.tsv`, stimulus columns, image paths, fMRI NIfTI files, and EEG files. If a dataset-specific column name differs, use CLI overrides such as `--stimulus-column`.

---

## 5. Requirements

Main dependencies are in `requirements.txt`. Colab setup is handled by `scripts/setup_colab.sh`.

---

## 6. Citation Pointers

Please cite OpenNeuro datasets according to their dataset pages and cite CORnet-S / CORnet when using the pretrained model in publications.
