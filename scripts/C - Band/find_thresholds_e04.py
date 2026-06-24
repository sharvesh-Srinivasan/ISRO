import os
import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============================================================
# find_thresholds_e04.py
# ------------------------------------------------------------
# PURPOSE:
#   This script is the C-band (EOS-04) equivalent of find_thresholds.py.
#   It reads the EOS-04 HH/HV tiles, computes the gray-scale composite,
#   and outputs TWO things that differ from the L-band pipeline:
#
#   1. CLASSIFICATION THRESHOLDS — the dB boundary values between
#      Water / Land / Crop land / Mountain classes.  C-band backscatter
#      has different absolute magnitudes than L-band, so these must be
#      derived from the actual C-band data, NOT reused from L-band.
#
#   2. STRETCH RANGE (lower_threshold, upper_threshold) — the dB clipping
#      range used inside convert_to_db_and_stretch().  L-band used (-7, 3.5).
#      C-band composites have a wider/different dynamic range and the correct
#      values must come from the data's own 2nd–98th percentile.
#
# HOW TO USE:
#   python scripts/find_thresholds_e04.py
#
#   Copy the printed output values into train_unet_e04.py and
#   train_stacking_e04.py where marked with # <-- UPDATE FROM THIS SCRIPT
# ============================================================

# -----------------------------------
# C-BAND (EOS-04) TILE FOLDERS
# -----------------------------------
# These point to the EOS-04 tiles — different from L-band (HH_tiles / HV_tiles)
HH_FOLDER = 'D:/ISRO/Proj/E04_HH_tiles'
HV_FOLDER = 'D:/ISRO/Proj/E04_HV_tiles'


def convert_to_db_and_stretch(data, lower_threshold, upper_threshold):
    data = np.nan_to_num(data, nan=1e-9)
    data = np.where(data <= 0, 1e-9, data)
    data_db = 10 * np.log10(data)
    data_db = np.nan_to_num(data_db, nan=lower_threshold)
    data_db = (data_db - lower_threshold) / (upper_threshold - lower_threshold) * 255
    data_db = np.clip(data_db, 0, 255)
    return data_db.astype(np.uint8)


def convert_to_db(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        db_image = 10 * np.log10(image)
    return db_image


def resize_hv_data(hv_data, hv_transform, hh_transform, hh_shape, hv_crs, hh_crs):
    hv_data_resized = np.empty(hh_shape, dtype=hv_data.dtype)
    warp.reproject(
        source=hv_data,
        destination=hv_data_resized,
        src_transform=hv_transform,
        src_crs=hv_crs,
        dst_transform=hh_transform,
        dst_crs=hh_crs,
        resampling=Resampling.nearest
    )
    return hv_data_resized


def main():
    if not os.path.exists(HH_FOLDER) or not os.path.exists(HV_FOLDER):
        raise FileNotFoundError(
            f"C-band tile folders not found.\n"
            f"Expected:\n  {HH_FOLDER}\n  {HV_FOLDER}\n"
            f"Run aoi_sub_e04.py first to generate the tiles."
        )

    hh_files = sorted([f for f in os.listdir(HH_FOLDER) if f.endswith(('.tif', '.tiff'))])
    hv_files = sorted([f for f in os.listdir(HV_FOLDER) if f.endswith(('.tif', '.tiff'))])

    print(f"Found {len(hh_files)} C-band (EOS-04) HH tiles.")

    # ----------------------------------------------------------------
    # PASS 1 — Collect raw dB values from HV, ratio, and diff
    #          to determine the correct STRETCH RANGE for C-band.
    #          L-band used (-7, 3.5) — this may be completely wrong
    #          for C-band. We derive the correct range from data.
    # ----------------------------------------------------------------
    all_hv_db      = []
    all_ratio_db   = []
    all_diff_db    = []

    max_images = min(30, len(hh_files))  # use a subset for speed
    print(f"\nPass 1 of 2 — Collecting raw dB statistics from {max_images} tiles...")

    for i in range(max_images):
        hh_tif = os.path.join(HH_FOLDER, hh_files[i])
        hv_tif = os.path.join(HV_FOLDER, hv_files[i]) if i < len(hv_files) else None

        if hv_tif is None or not os.path.exists(hv_tif):
            continue

        with rasterio.open(hh_tif) as hh_ds:
            hh_data     = hh_ds.read(1).astype(np.float32)
            hh_transform = hh_ds.transform
            hh_crs      = hh_ds.crs

        with rasterio.open(hv_tif) as hv_ds:
            hv_data     = hv_ds.read(1).astype(np.float32)
            hv_transform = hv_ds.transform
            hv_crs      = hv_ds.crs

        hv_r = resize_hv_data(hv_data, hv_transform, hh_transform, hh_data.shape, hv_crs, hh_crs)
        hv_r[hv_r <= 0] = 1e-9
        hh_data[hh_data <= 0] = 1e-9

        ratio = hh_data / hv_r
        diff  = np.abs(hh_data - hv_r)   # abs so log10 is safe
        diff[diff <= 0] = 1e-9

        # Collect dB values (finite only, subsampled for speed)
        for arr, store in [(hv_r, all_hv_db), (ratio, all_ratio_db), (diff, all_diff_db)]:
            db_vals = 10 * np.log10(arr)
            db_vals = db_vals[np.isfinite(db_vals)]
            if len(db_vals) > 5000:
                db_vals = db_vals[np.random.choice(len(db_vals), 5000, replace=False)]
            store.extend(db_vals.tolist())

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{max_images} tiles...")

    # Determine stretch range = 2nd–98th percentile of the composite channels
    all_channels_db = np.array(all_hv_db + all_ratio_db + all_diff_db)
    all_channels_db = all_channels_db[np.isfinite(all_channels_db)]

    c_lower = float(np.percentile(all_channels_db, 2))
    c_upper = float(np.percentile(all_channels_db, 98))

    print(f"\n  C-band dB range (2nd–98th pct): [{c_lower:.3f}, {c_upper:.3f}]")
    print(f"  (L-band used: [-7, 3.5] — these are likely different for C-band)")

    # ----------------------------------------------------------------
    # PASS 2 — Build composites with the correct C-band stretch range
    #          and run K-Means to find classification thresholds.
    # ----------------------------------------------------------------
    print(f"\nPass 2 of 2 — Building composites and running K-Means clustering...")

    all_db_pixels = []

    for i in range(max_images):
        hh_tif = os.path.join(HH_FOLDER, hh_files[i])
        hv_tif = os.path.join(HV_FOLDER, hv_files[i]) if i < len(hv_files) else None

        if hv_tif is None or not os.path.exists(hv_tif):
            continue

        with rasterio.open(hh_tif) as hh_ds:
            hh_data     = hh_ds.read(1).astype(np.float32)
            hh_transform = hh_ds.transform
            hh_crs      = hh_ds.crs

        with rasterio.open(hv_tif) as hv_ds:
            hv_data     = hv_ds.read(1).astype(np.float32)
            hv_transform = hv_ds.transform
            hv_crs      = hv_ds.crs

        hv_r = resize_hv_data(hv_data, hv_transform, hh_transform, hh_data.shape, hv_crs, hh_crs)
        hv_r[hv_r <= 0] = 1e-9

        ratio = hh_data / hv_r
        diff  = hh_data - hv_r

        # Use the C-band derived stretch range (NOT the L-band -7 / 3.5)
        hh_hv_ratio_db = convert_to_db_and_stretch(ratio,  c_lower, c_upper)
        hh_hv_diff_db  = convert_to_db_and_stretch(diff,   c_lower, c_upper)
        hv_stretched   = convert_to_db_and_stretch(hv_r,   c_lower, c_upper)

        gray_composite = 0.33 * hv_stretched + 0.33 * hh_hv_ratio_db + 0.33 * hh_hv_diff_db
        image = resize(gray_composite, (256, 256), anti_aliasing=True, mode='reflect')
        db_image = convert_to_db(image)

        min_val = np.nanmin(db_image) if len(db_image[~np.isnan(db_image)]) > 0 else 0
        db_image[np.isnan(db_image)] = min_val
        finite_pixels = db_image[np.isfinite(db_image)]
        if len(finite_pixels) > 0:
            db_image[np.isinf(db_image)] = np.nanmax(finite_pixels) + 1

        all_db_pixels.extend(db_image.flatten())

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{max_images} tiles...")

    # ----------------------------------------------------------------
    # K-MEANS to find classification thresholds
    # ----------------------------------------------------------------
    all_db_pixels = np.array(all_db_pixels)
    p1, p99 = np.percentile(all_db_pixels, [1, 99])
    filtered = all_db_pixels[(all_db_pixels >= p1) & (all_db_pixels <= p99)]

    pixels = filtered.reshape(-1, 1)
    sample_size = min(len(pixels), 100000)
    np.random.seed(42)
    sampled = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(sampled)

    centers = np.sort(kmeans.cluster_centers_.flatten())
    thresholds = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers) - 1)]

    # ----------------------------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  C-BAND (EOS-04) CALIBRATION RESULTS")
    print("=" * 60)

    print("\n[1] STRETCH RANGE — copy into train_unet_e04.py and train_stacking_e04.py")
    print(f"    C_LOWER = {c_lower:.3f}")
    print(f"    C_UPPER = {c_upper:.3f}")
    print("    (These replace the L-band values of -7 and 3.5)")

    print("\n[2] CLASSIFICATION THRESHOLDS — copy into generate_mask_lib() in train_unet_e04.py")
    print(f"    threshold_water    = [-np.inf, {thresholds[0]:.3f}]")
    print(f"    threshold_land     = [{thresholds[0]:.3f}, {thresholds[1]:.3f}]")
    print(f"    threshold_cropland = [{thresholds[1]:.3f}, {thresholds[2]:.3f}]")
    print(f"    threshold_mountain = [{thresholds[2]:.3f}, np.inf]")

    print("\n[3] STACKING THRESHOLDS — copy into map_backscatter_to_label() in train_stacking_e04.py")
    print(f"    if backscatter <= {thresholds[0]:.3f}:   return 0  # Water")
    print(f"    elif backscatter <= {thresholds[1]:.3f}: return 1  # Land")
    print(f"    elif backscatter <= {thresholds[2]:.3f}: return 2  # Crop land")
    print(f"    else:                              return 3  # Mountain")
    print("=" * 60)
    print("\nDone. Paste the values above into the C-band training scripts.")


if __name__ == "__main__":
    main()
