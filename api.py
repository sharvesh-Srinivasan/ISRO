import io
import time
import base64
import zipfile
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

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

def np_to_base64(img_array):
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/api/process", methods=["POST"])
def process():
    if 'file' not in request.files:
        return jsonify({"error": "Missing SAR file archive"}), 400

    band = request.form.get("band", "L")
    sar_file = request.files['file']
    file_bytes = sar_file.read()
    
    t0 = time.time()
    
    # 1. Unzip the file in memory
    tif_files = {}
    is_zip = sar_file.filename.lower().endswith('.zip')
    
    try:
        if is_zip:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for filename in z.namelist():
                    if filename.lower().endswith(('.tif', '.tiff')) and not filename.startswith('__MACOSX'):
                        tif_files[filename] = z.read(filename)
        else:
            # Fallback if they uploaded a single TIF directly by accident
            tif_files[sar_file.filename] = file_bytes
    except Exception as e:
        return jsonify({"error": "Failed to parse ZIP archive: " + str(e)}), 400

    if not tif_files:
        return jsonify({"error": "No .tif or .tiff files found inside the ZIP archive."}), 400

    # 2. Extract HH and HV polarizations
    hh_bytes = None
    hv_bytes = None
    
    # Strategy A: If there are multiple TIF files in the zip, sort by HH/HV in filename
    if len(tif_files) >= 2:
        for fname, b in tif_files.items():
            if 'hv' in fname.lower():
                hv_bytes = b
            elif 'hh' in fname.lower():
                hh_bytes = b
        
        # Fallback if names aren't clear
        if not hh_bytes: hh_bytes = list(tif_files.values())[0]
        if not hv_bytes: hv_bytes = list(tif_files.values())[1]
    else:
        # Strategy B: Single TIF file (might contain multiple bands)
        hh_bytes = list(tif_files.values())[0]
        hv_bytes = list(tif_files.values())[0]

    try:
        # Load HH
        with MemoryFile(hh_bytes) as memfile:
            with memfile.open() as ds:
                profile = ds.profile
                transform = ds.transform
                crs = str(ds.crs) if ds.crs else "Not Defined"
                bounds = ds.bounds
                dtype = str(ds.dtypes[0])
                band_count = ds.count
                hh = ds.read(1).astype(np.float32)
                
                # If the single file actually has multiple bands, read band 2 as HV
                if len(tif_files) == 1 and band_count >= 2:
                    hv = ds.read(2).astype(np.float32)
                    extracted_hv_from_band2 = True
                else:
                    extracted_hv_from_band2 = False
        
        # Load HV from separate file if needed
        if len(tif_files) >= 2:
            with MemoryFile(hv_bytes) as memfile:
                with memfile.open() as ds:
                    hv = ds.read(1).astype(np.float32)
    except Exception as e:
        return jsonify({"error": "Failed to read TIF tensors: " + str(e)}), 500

    if hh.shape[0] > 1024 or hh.shape[1] > 1024:
        hh = hh[:1024, :1024]
        if 'hv' in locals(): hv = hv[:1024, :1024]

    height, width = hh.shape

    # 3. Finalize polarizations (fallback synthesis if HV genuinely missing)
    np.random.seed(42)
    if 'hv' not in locals() or (len(tif_files) == 1 and not extracted_hv_from_band2):
        hv = hh * 0.3 + (np.random.rand(height, width) * 10).astype(np.float32)
        synth_method = "hv = hh × 0.3 + 𝒩(0, 10)"
        synth_desc = "Missing HV polarization synthesized from HH."
    else:
        synth_method = "Dual-polarization explicitly extracted from ZIP archive."
        synth_desc = f"Extracted HH and HV arrays directly from {len(tif_files)} physical TIF file(s)."
        # Ensure dimensions match if they came from different files
        if hh.shape != hv.shape:
            min_h = min(hh.shape[0], hv.shape[0])
            min_w = min(hh.shape[1], hv.shape[1])
            hh = hh[:min_h, :min_w]
            hv = hv[:min_h, :min_w]
            height, width = hh.shape

    # --- Stage 1: Raw visualization (using HH) ---
    raw_min, raw_max = float(np.min(hh)), float(np.max(hh))
    raw_vis = np.clip((hh - raw_min) / (raw_max - raw_min + 1e-9) * 255, 0, 255).astype(np.uint8)

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

    # --- Stage 3: K-Means Threshold Mask ---
    db_img = composite_db
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    water_mask   = db_img <= config["w"]
    land_mask    = (db_img > config["w"]) & (db_img <= config["l"])
    crop_mask    = (db_img > config["l"]) & (db_img <= config["c"])
    mtn_mask     = db_img > config["c"]

    mask[water_mask]  = [41,  128, 185]
    mask[land_mask]   = [39,  174, 96]
    mask[crop_mask]   = [243, 156, 18]
    mask[mtn_mask]    = [192, 57,  43]

    process_ms = int((time.time() - t0) * 1000)
    total_px = height * width

    # --- Statistics ---
    raw_flat = hh.flatten()
    composite_flat = composite_db.flatten()

    hist_counts, hist_bins = np.histogram(composite_flat, bins=24)
    histogram = [{"bin": round(float(hist_bins[i]), 2), "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    raw_hist_counts, raw_hist_bins = np.histogram(raw_flat[raw_flat > 0], bins=24)
    raw_histogram = [{"bin": round(float(raw_hist_bins[i]), 1), "count": int(raw_hist_counts[i])} for i in range(len(raw_hist_counts))]

    water_pct   = round(float(np.sum(water_mask) / total_px) * 100, 2)
    land_pct    = round(float(np.sum(land_mask)  / total_px) * 100, 2)
    crop_pct    = round(float(np.sum(crop_mask)  / total_px) * 100, 2)
    mtn_pct     = round(float(np.sum(mtn_mask)   / total_px) * 100, 2)

    snr         = round(float(np.mean(raw_vis) / (np.std(raw_vis) + 1e-9)), 4)
    entropy     = round(float(-np.sum(np.where(raw_flat > 0,
                    raw_flat / (np.sum(raw_flat) + 1e-9) * np.log2(raw_flat / (np.sum(raw_flat) + 1e-9) + 1e-12), 0))), 4)

    db_valid = composite_flat[np.isfinite(composite_flat)]

    result = {
        "images": {
            "raw":      np_to_base64(raw_vis),
            "stretched": np_to_base64(stretched),
            "mask":     np_to_base64(mask)
        },
        "file_metadata": {
            "filename":    sar_file.filename,
            "size_mb":     round(len(file_bytes) / (1024 * 1024), 3),
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
        "histogram":     histogram,
        "raw_histogram": raw_histogram,
        "synth": {
            "method": synth_method,
            "desc": synth_desc,
            "raw_label": "HH/HV"
        }
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
