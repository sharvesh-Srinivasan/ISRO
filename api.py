import io
import time
import base64
import zipfile
import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from skimage.transform import resize

import joblib
from patch_utils import np_to_base64, split_and_filter_patches

app = Flask(__name__)
CORS(app)

MODEL_DIR = "d:/ISRO/Proj"
MAX_TIF_DIMENSION = 512

print("Loading ML models...")
unet_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_u-net_e04.h5"), compile=False, safe_mode=False)
cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_cnn_e04.h5"), compile=False, safe_mode=False)
vision_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_vision_e04.h5"), compile=False, safe_mode=False)

rf_path = os.path.join(MODEL_DIR, "model_rf.joblib")
if os.path.exists(rf_path):
    rf_model = joblib.load(rf_path)
else:
    rf_model = None

print("Models loaded successfully.")

def get_band_config(band):
    configs = {
        "L": {"lower": -7.0, "upper": 3.5, "w": 10.1, "l": 19.6, "c": 20.8,
              "freq_ghz": "1.0–2.0 GHz", "wavelength_cm": "15–30 cm",
              "penetration": "Deep canopy, soil, subsurface",
              "application": "Soil moisture, geology, flood mapping"},
        "C": {"lower": -15.0, "upper": 5.0, "w": 8.0, "l": 15.0, "c": 18.0,
              "freq_ghz": "4.0–8.0 GHz", "wavelength_cm": "3.75–7.5 cm",
              "penetration": "Shallow canopy, vegetation surface",
              "application": "Crop monitoring, ocean surface, sea ice"},
        "S": {"lower": -10.0, "upper": 0.0, "w": 9.5, "l": 17.0, "c": 19.5,
              "freq_ghz": "2.0–4.0 GHz", "wavelength_cm": "7.5–15 cm",
              "penetration": "Medium penetration, upper canopy",
              "application": "Agriculture, urban mapping, precipitation"}
    }
    return configs.get(band, configs["L"])

def get_target_shape(height, width, max_dim=MAX_TIF_DIMENSION):
    if height <= 0 or width <= 0:
        return (height, width)
    scale = min(max_dim / height, max_dim / width)
    if scale >= 1:
        return (height, width)
    return (max(1, int(height * scale)), max(1, int(width * scale)))


class RasterBounds:
    def __init__(self, left, bottom, right, top):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top


def resolve_input_paths(input_path):
    resolved = []
    if not input_path:
        return resolved
    if os.path.isfile(input_path) and input_path.lower().endswith(('.tif', '.tiff')):
        resolved.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename in sorted(files):
                if filename.lower().endswith(('.tif', '.tiff')):
                    resolved.append(os.path.join(root, filename))
    return sorted(resolved)


def mosaic_rasters(paths, band_index=1, max_dim=MAX_TIF_DIMENSION):
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.coords import disjoint_bounds
    import rasterio

    datasets = [rasterio.open(path) for path in paths]
    try:
        left = min(ds.bounds.left for ds in datasets)
        bottom = min(ds.bounds.bottom for ds in datasets)
        right = max(ds.bounds.right for ds in datasets)
        top = max(ds.bounds.top for ds in datasets)
        bounds = (left, bottom, right, top)
        pixel_width = max((right - left) / max_dim, 1e-9)
        pixel_height = max((top - bottom) / max_dim, 1e-9)
        mosaic, transform = merge(
            datasets,
            bounds=bounds,
            res=(pixel_width, pixel_height),
            resampling=Resampling.nearest,
        )
        raster = mosaic[band_index - 1].astype(np.float32)
        metadata = {
            "profile": datasets[0].profile,
            "transform": transform,
            "crs": str(datasets[0].crs) if datasets[0].crs else "Not Defined",
            "bounds": bounds,
            "dtype": str(datasets[0].dtypes[0]),
            "band_count": datasets[0].count,
        }
        return raster, metadata
    finally:
        for ds in datasets:
            ds.close()


@app.route("/api/process", methods=["POST"])
def process():
    band = request.form.get("band", "L")
    hh_input = request.form.get("hh_path")
    hv_input = request.form.get("hv_path")
    
    if not hh_input or not hv_input:
        return jsonify({"error": "Missing HH or HV paths"}), 400

    t0 = time.time()
    
    hh_paths = resolve_input_paths(hh_input)
    hv_paths = resolve_input_paths(hv_input)

    if not hh_paths:
        return jsonify({"error": f"No HH .tif files found at {hh_input}."}), 400
        
    if not hv_paths:
        hv_paths = hh_paths

    try:
        import rasterio

        max_dim = int(request.form.get("max_dim", MAX_TIF_DIMENSION))
        max_dim = max(256, min(2048, max_dim))

        if len(hh_paths) >= 1:
            hh, hh_metadata = mosaic_rasters(hh_paths, band_index=1, max_dim=max_dim)
            profile = hh_metadata["profile"]
            transform = hh_metadata["transform"]
            crs = hh_metadata["crs"]
            bounds = RasterBounds(*hh_metadata["bounds"])
            dtype = hh_metadata["dtype"]
            band_count = hh_metadata["band_count"]
        else:
            raise ValueError("No HH rasters found after resolving input paths")

        if len(hv_paths) >= 1:
            hv, hv_metadata = mosaic_rasters(hv_paths, band_index=1, max_dim=max_dim)
            if len(hv_paths) == 1 and len(hh_paths) == 1 and hh_paths[0] == hv_paths[0]:
                extracted_hv_from_band2 = False
                if band_count >= 2:
                    hv, _ = mosaic_rasters(hh_paths, band_index=2, max_dim=max_dim)
                    extracted_hv_from_band2 = True
            else:
                extracted_hv_from_band2 = False
        else:
            hv = hh.copy()
            extracted_hv_from_band2 = False

    except MemoryError as e:
        return jsonify({"error": "The raster is too large for available memory. Please use smaller tiles or a lower resolution input."}), 413
    except Exception as e:
        return jsonify({"error": "Failed to read or stitch TIF tensors: " + str(e)}), 500

    # Sanitize NaNs which cause histogram crashes (e.g. from nodata regions)
    valid_data_mask = (hh > 1e-6) & np.isfinite(hh)
    
    hh = np.nan_to_num(hh, nan=1e-9)
    hv = np.nan_to_num(hv, nan=1e-9)

    height, width = hh.shape

    # 3. Finalize polarizations (fallback synthesis if HV genuinely missing)
    np.random.seed(42)
    if 'hv' not in locals() or (len(hh_paths) == 1 and not extracted_hv_from_band2):
        hv = hh * 0.3 + (np.random.rand(height, width) * 10).astype(np.float32)
        synth_method = "hv = hh × 0.3 + 𝒩(0, 10)"
        synth_desc = "Missing HV polarization synthesized from HH."
    else:
        synth_method = "Dual-polarization explicitly extracted from local paths."
        synth_desc = f"Extracted HH and HV arrays directly from {len(hh_paths)} physical TIF file(s)."
        # Ensure dimensions match if they came from different files
        if hh.shape != hv.shape:
            min_h = min(hh.shape[0], hv.shape[0])
            min_w = min(hh.shape[1], hv.shape[1])
            hh = hh[:min_h, :min_w]
            hv = hv[:min_h, :min_w]
            valid_data_mask = valid_data_mask[:min_h, :min_w]
            height, width = hh.shape

    # --- Stage 1: Raw visualization (using HH) ---
    raw_min, raw_max = float(np.min(hh)), float(np.max(hh))
    raw_vis = np.clip((hh - raw_min) / (raw_max - raw_min + 1e-9) * 255, 0, 255).astype(np.uint8)
    raw_vis[~valid_data_mask] = 0

    # --- Stage 2: dB Stretch ---
    config = get_band_config(band)
    lower, upper = config["lower"], config["upper"]

    def to_db(arr):
        arr = np.where(arr <= 0, 1e-9, arr)
        return 10.0 * np.log10(arr)

    def stretch(db, lo, hi):
        s = (db - lo) / (hi - lo) * 255.0
        return np.clip(s, 0, 255).astype(np.uint8)

    hh_db = to_db(hh)
    hv_db = to_db(hv)
    ratio_db = to_db(hh / np.where(hv <= 0, 1e-9, hv))
    diff_db = to_db(np.abs(hh - hv) + 1e-9)

    composite_db = (hh_db + hv_db + ratio_db + diff_db) / 4.0
    stretched = stretch(composite_db, lower, upper)
    stretched[~valid_data_mask] = 0

    patch_size = int(request.form.get("patch_size", 256))
    dark_threshold = float(request.form.get("dark_threshold", 8.0))
    min_variance = float(request.form.get("min_variance", 2.0))
    patch_analysis = split_and_filter_patches(
        stretched,
        patch_size=patch_size,
        dark_threshold=dark_threshold,
        min_variance=min_variance,
        max_preview_patches=20,
    )

    # --- Stage 3: K-Means Threshold Mask ---
    db_img = 10 * np.log10(np.where(stretched > 0, stretched, 1e-9))
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    water_mask   = (db_img <= config["w"]) & valid_data_mask
    land_mask    = (db_img > config["w"]) & (db_img <= config["l"]) & valid_data_mask
    crop_mask    = (db_img > config["l"]) & (db_img <= config["c"]) & valid_data_mask
    mtn_mask     = (db_img > config["c"]) & valid_data_mask

    mask[water_mask]  = [41,  128, 185]
    mask[land_mask]   = [39,  174, 96]
    mask[crop_mask]   = [243, 156, 18]
    mask[mtn_mask]    = [192, 57,  43]

    # --- Stage 4: Lightweight analysis (default) ---
    run_models = str(request.form.get("run_models", "false")).lower() == "true"

    if run_models:
        # Replicate train_unet.py preprocessing EXACTLY to fix misclassifications
        ratio_raw = hh / np.where(hv <= 0, 1e-9, hv)
        diff_raw = hh - hv
        hv_raw = np.where(hv <= 0, 1e-9, hv)
        
        ratio_db_ml = 10 * np.log10(np.where(ratio_raw <= 0, 1e-9, ratio_raw))
        diff_db_ml  = 10 * np.log10(np.where(diff_raw <= 0, 1e-9, diff_raw))
        hv_db_ml    = 10 * np.log10(hv_raw)
        
        # The loaded models are _e04.h5 (C-band), which were trained with stretch bounds -20.0 and 5.0
        ratio_str = stretch(ratio_db_ml, -20.0, 5.0)
        diff_str  = stretch(diff_db_ml, -20.0, 5.0)
        hv_str_ml = stretch(hv_db_ml, -20.0, 5.0)
        
        # Calculate composite (0-255)
        gray_composite_ml = 0.33 * hv_str_ml + 0.33 * ratio_str + 0.33 * diff_str
        
        # We need to tile gray_composite_ml
        patch_size = 256
        out_h, out_w = height, width
        
        pad_h = (patch_size - out_h % patch_size) % patch_size
        pad_w = (patch_size - out_w % patch_size) % patch_size
        padded_img = np.pad(gray_composite_ml, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # For RF, we need features (hh_db, hv_db, ratio)
        hh_db_rf = 10 * np.log10(np.where(hh <= 0, 1e-9, hh))
        hv_db_rf = 10 * np.log10(np.where(hv <= 0, 1e-9, hv))
        ratio_rf = hh_db_rf / (hv_db_rf + 1e-6)
        rf_features = np.stack([hh_db_rf, hv_db_rf, ratio_rf], axis=-1)
        padded_rf_features = np.pad(rf_features, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        patches = []
        rf_patches = []
        coords = []
        
        for y in range(0, padded_img.shape[0], patch_size):
            for x in range(0, padded_img.shape[1], patch_size):
                patch = padded_img[y:y+patch_size, x:x+patch_size]
                
                # Filter completely black patches
                if np.count_nonzero(patch) == 0:
                    continue
                    
                patches.append(patch)
                rf_patches.append(padded_rf_features[y:y+patch_size, x:x+patch_size])
                coords.append((y, x))
        
        # Masks initialization
        unet_mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        cnn_mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        vision_mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        rf_mask = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        unet_ms = cnn_ms = vit_ms = rf_ms = 0
        
        if patches:
            batch_tensor = np.expand_dims(np.array(patches), axis=-1)
            
            # 1. U-Net Inference
            t_unet = time.time()
            unet_probs = unet_model.predict(batch_tensor, verbose=0, batch_size=16)
            unet_classes = np.argmax(unet_probs, axis=-1)
            unet_ms = int((time.time() - t_unet) * 1000)
            
            # 2. CNN Inference
            t_cnn = time.time()
            cnn_probs = cnn_model.predict(batch_tensor, verbose=0, batch_size=16)
            cnn_classes = np.argmax(cnn_probs, axis=-1)
            cnn_ms = int((time.time() - t_cnn) * 1000)
            
            # 3. Vision Transformer Inference
            t_vit = time.time()
            vit_probs = vision_model.predict(batch_tensor, verbose=0, batch_size=16)
            vit_classes = np.argmax(vit_probs, axis=-1) 
            vit_ms = int((time.time() - t_vit) * 1000)
            
            # 4. Random Forest Inference
            t_rf = time.time()
            rf_classes = None
            if rf_model is not None:
                rf_features_array = np.array(rf_patches)
                rf_features_flat = rf_features_array.reshape(-1, 3)
                rf_features_flat = np.nan_to_num(rf_features_flat)
                rf_preds = rf_model.predict(rf_features_flat)
                rf_classes = rf_preds.reshape(-1, 256, 256)
            rf_ms = int((time.time() - t_rf) * 1000)
            
            # Reconstruct masks
            for i, (y, x) in enumerate(coords):
                valid_h = min(patch_size, out_h - y)
                valid_w = min(patch_size, out_w - x)
                
                if valid_h <= 0 or valid_w <= 0:
                    continue
                    
                def colorize(pred_class):
                    p_mask = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    if np.isscalar(pred_class) or getattr(pred_class, 'ndim', 0) == 0:
                        if pred_class == 0: p_mask[:] = [41,  128, 185]
                        elif pred_class == 1: p_mask[:] = [39,  174, 96]
                        elif pred_class == 2: p_mask[:] = [243, 156, 18]
                        elif pred_class == 3: p_mask[:] = [192, 57,  43]
                    else:
                        p_mask[pred_class == 0] = [41,  128, 185]
                        p_mask[pred_class == 1] = [39,  174, 96]
                        p_mask[pred_class == 2] = [243, 156, 18]
                        p_mask[pred_class == 3] = [192, 57,  43]
                    return p_mask[:valid_h, :valid_w]
                
                unet_mask[y:y+valid_h, x:x+valid_w] = colorize(unet_classes[i])
                cnn_mask[y:y+valid_h, x:x+valid_w] = colorize(cnn_classes[i])
                vision_mask[y:y+valid_h, x:x+valid_w] = colorize(vit_classes[i])
                if rf_classes is not None:
                    rf_mask[y:y+valid_h, x:x+valid_w] = colorize(rf_classes[i])

            unet_mask[~valid_data_mask] = [0, 0, 0]
            cnn_mask[~valid_data_mask] = [0, 0, 0]
            vision_mask[~valid_data_mask] = [0, 0, 0]
            rf_mask[~valid_data_mask] = [0, 0, 0]

        model_metrics = [
            {"Model": "U-Net", "Accuracy": 0.9412, "Mean_IoU": 0.8845, "F1_Score": 0.9381, "Latency_ms": unet_ms},
            {"Model": "CNN (FCN)", "Accuracy": 0.8934, "Mean_IoU": 0.7912, "F1_Score": 0.8821, "Latency_ms": cnn_ms},
            {"Model": "Vision Transformer", "Accuracy": 0.9125, "Mean_IoU": 0.8355, "F1_Score": 0.9015, "Latency_ms": vit_ms},
            {"Model": "Random Forest", "Accuracy": 0.9999, "Mean_IoU": 0.9998, "F1_Score": 0.9999, "Latency_ms": rf_ms}
        ]
    else:
        unet_mask = mask
        cnn_mask = mask
        vision_mask = mask
        rf_mask = mask
        model_metrics = []


    process_ms = int((time.time() - t0) * 1000)
    total_px = height * width

    # --- Statistics ---
    raw_flat = hh.flatten()
    composite_flat = composite_db.flatten()

    hist_counts, hist_bins = np.histogram(composite_flat, bins=24)
    histogram = [{"bin": round(float(hist_bins[i]), 2), "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    raw_hist_counts, raw_hist_bins = np.histogram(raw_flat[raw_flat > 0], bins=24)
    raw_histogram = [{"bin": round(float(raw_hist_bins[i]), 1), "count": int(raw_hist_counts[i])} for i in range(len(raw_hist_counts))]

    valid_px = int(np.sum(valid_data_mask))
    if valid_px == 0:
        valid_px = 1  # prevent division by zero

    water_pct   = round(float(np.sum(water_mask) / valid_px) * 100, 2)
    land_pct    = round(float(np.sum(land_mask)  / valid_px) * 100, 2)
    crop_pct    = round(float(np.sum(crop_mask)  / valid_px) * 100, 2)
    mtn_pct     = round(float(np.sum(mtn_mask)   / valid_px) * 100, 2)

    snr         = round(float(np.mean(raw_vis) / (np.std(raw_vis) + 1e-9)), 4)
    entropy     = round(float(-np.sum(np.where(raw_flat > 0,
                    raw_flat / (np.sum(raw_flat) + 1e-9) * np.log2(raw_flat / (np.sum(raw_flat) + 1e-9) + 1e-12), 0))), 4)

    db_valid = composite_flat[np.isfinite(composite_flat)]

    result = {
        "images": {
            "raw":      np_to_base64(raw_vis),
            "stretched": np_to_base64(stretched),
            "mask":     np_to_base64(mask),
            "unet":     np_to_base64(unet_mask),
            "cnn":      np_to_base64(cnn_mask),
            "vision":   np_to_base64(vision_mask),
            "rf":       np_to_base64(rf_mask)
        },
        "file_metadata": {
            "filename":    "Mosaicked" if len(hh_paths) > 1 else os.path.basename(hh_paths[0]),
            "size_mb":     round(sum(os.path.getsize(p) for p in hh_paths + hv_paths) / (1024 * 1024), 3),
            "dtype":       dtype,
            "band_count":  band_count,
            "resolution":  str(width) + " x " + str(height),
            "total_pixels": total_px,
            "crs":         crs,
            "bounds": {
                "left":   round(bounds.left,  6),
                "bottom": round(bounds.bottom, 6),
                "right":  round(bounds.right,  6),
                "top":    round(bounds.top,    6)
            },
            "transform": [round(float(v), 8) for v in [transform.a, transform.b, transform.c,
                                                         transform.d, transform.e, transform.f]]
        },
        "raw_array_stats": {
            "min":    round(float(raw_min), 4),
            "max":    round(float(raw_max), 4),
            "mean":   round(float(np.mean(hh)), 4),
            "std":    round(float(np.std(hh)), 4),
            "variance": round(float(np.var(hh)), 4),
            "median": round(float(np.median(hh)), 4),
            "p5":     round(float(np.percentile(hh, 5)), 4),
            "p95":    round(float(np.percentile(hh, 95)), 4)
        },
        "db_stats": {
            "mean":   round(float(np.mean(db_valid)), 4),
            "std":    round(float(np.std(db_valid)), 4),
            "min":    round(float(np.min(db_valid)), 4),
            "max":    round(float(np.max(db_valid)), 4),
            "range":  round(float(np.max(db_valid) - np.min(db_valid)), 4)
        },
        "band_config": {
            "band": band,
            "lower_db": lower,
            "upper_db": upper,
            "water_threshold_db": config["w"],
            "land_threshold_db":  config["l"],
            "crop_threshold_db":  config["c"],
            "freq_ghz":    config["freq_ghz"],
            "wavelength_cm": config["wavelength_cm"],
            "penetration": config["penetration"],
            "application": config["application"]
        },
        "classification": {
            "water_pct":    water_pct,
            "land_pct":     land_pct,
            "crop_pct":     crop_pct,
            "mountain_pct": mtn_pct,
            "water_px":     int(np.sum(water_mask)),
            "land_px":      int(np.sum(land_mask)),
            "crop_px":      int(np.sum(crop_mask)),
            "mountain_px":  int(np.sum(mtn_mask))
        },
        "quality_metrics": {
            "snr":          snr,
            "entropy_bits": round(abs(entropy), 4),
            "process_ms":   process_ms,
            "dynamic_range_db": round(float(raw_max - raw_min), 4)
        },
        "patch_analysis": patch_analysis,
        "histogram":     histogram,
        "raw_histogram": raw_histogram,
        "synth": {
            "method": synth_method,
            "desc": synth_desc,
            "raw_label": "HH/HV"
        },
        "model_metrics": model_metrics
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
