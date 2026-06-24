# SAR Land Cover Classification

Three scripts are included — run them in order.

---

## Setup

Install the required libraries:

```
pip install tensorflow rasterio scikit-image scikit-learn pandas numpy matplotlib
```

---

## Data Folder Structure

Organize your S-band SAR tiles like this before running anything:

```
<your_project_path>/
├── HH_tiles/     ← HH polarization .tif tiles
├── HV_tiles/     ← HV polarization .tif tiles
└── scripts/
    ├── find_thresholds.py
    ├── train_unet.py
    └── train_stacking.py
```

---

## Step 1 — Find Thresholds for Your Data

Since this is S-band data, the dB backscatter ranges will be different from L-band. Run this script first to automatically find the right classification thresholds for your dataset using K-Means clustering.

Open `find_thresholds.py` and update the folder paths at the top of `main()`:

```python
hh_folder = '<your_path>/HH_tiles'
hv_folder = '<your_path>/HV_tiles'
```

Then run:

```
python scripts/find_thresholds.py
```

It will print something like:

```
Classification Ranges (Copy these to train_unet.py):
Water      : [-np.inf, X.XXX]
Land       : [X.XXX, X.XXX]
Crop land  : [X.XXX, X.XXX]
Mountain   : [X.XXX, np.inf]
```

Copy those four threshold values — you'll need them in the next step.

---

## Step 2 — Update Thresholds in train_unet.py

Open `train_unet.py` and find the `generate_mask_lib` function (around line 28). Replace the threshold values with the ones you got from Step 1:

```python
threshold_water    = [-np.inf, X.XXX]
threshold_land     = [X.XXX, X.XXX]
threshold_cropland = [X.XXX, X.XXX]
threshold_mountain = [X.XXX, np.inf]
```

Also update the data folder paths in `prepare_data()`:

```python
hh_folder = '<your_path>/HH_tiles'
hv_folder = '<your_path>/HV_tiles'
```

---

## Step 3 — Run train_unet.py

This trains three models on your data — U-Net, CNN (FCN), and a Vision Transformer — and compares them.

```
python scripts/train_unet.py
```

At the end it prints an evaluation table (Accuracy, Mean IoU, F1-Score) and saves three model files:

```
model_u-net.h5
model_cnn.h5
model_vision.h5
```

---

## Step 4 — Run train_stacking.py (Optional)

This trains a stacking ensemble (SVM + MLP with Logistic Regression as meta-model). Update the folder paths the same way, then run:

```
python scripts/train_stacking.py
```

> This step takes a few minutes because SVM training is slow on large feature vectors. That's expected.

At the end it prints Accuracy, MCC, and F1-Score for the stacking model.

---

## Notes

- The thresholds in Step 1–2 are the most important thing to get right for a new dataset. The default values in the scripts were calibrated for L-band data and **will not be accurate for S-band**.
- Training is currently set to 2 epochs for speed. You can increase this in `train_unet.py` inside `main()` by changing `EPOCHS = 2` to a higher value.
