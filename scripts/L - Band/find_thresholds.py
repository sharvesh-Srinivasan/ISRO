import os
import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
    hh_folder = 'd:/ISRO/Proj/HH_tiles'
    hv_folder = 'd:/ISRO/Proj/HV_tiles'
    
    if not os.path.exists(hh_folder) or not os.path.exists(hv_folder):
        raise FileNotFoundError(f"Ensure {hh_folder} and {hv_folder} exist.")

    hh_files = os.listdir(hh_folder)
    hv_files = os.listdir(hv_folder)
    
    print(f"Found {len(hh_files)} HH files. Processing images...")
    
    all_db_pixels = []
    
    max_images = min(50, len(hh_files))
    
    for i in range(max_images):
        hh_file = hh_files[i]
        hv_file = hv_files[i]
        
        if not hh_file.endswith(('.tif', '.tiff')):
            continue
            
        hh_tif_file = os.path.join(hh_folder, hh_file)
        hv_tif_file = os.path.join(hv_folder, hv_file)
        
        if not os.path.exists(hv_tif_file):
            continue
            
        with rasterio.open(hh_tif_file) as hh_ds:
            hh_data = hh_ds.read(1)
            hh_transform = hh_ds.transform
            hh_crs = hh_ds.crs
    
        with rasterio.open(hv_tif_file) as hv_ds:
            hv_data = hv_ds.read(1)
            hv_transform = hv_ds.transform
            hv_crs = hv_ds.crs
            
        hv_data_resized = resize_hv_data(hv_data, hv_transform, hh_transform, hh_data.shape, hv_crs, hh_crs)
        hv_data_resized[hv_data_resized <= 0] = 1e-9
    
        hh_hv_ratio = hh_data / hv_data_resized
        hh_hv_diff = hh_data - hv_data_resized
    
        hh_hv_ratio_db = convert_to_db_and_stretch(hh_hv_ratio, -7, 3.5)
        hh_hv_diff_db = convert_to_db_and_stretch(hh_hv_diff, -7, 3.5)
        hv = convert_to_db_and_stretch(hv_data_resized, -7, 3.5)
    
        gray_composite = 0.33*hv + 0.33 * hh_hv_ratio_db + 0.33 * hh_hv_diff_db
        image = resize(gray_composite, (256, 256), anti_aliasing=True, mode='reflect')
        db_image = convert_to_db(image)
        
        min_val = np.nanmin(db_image) if len(db_image[~np.isnan(db_image)]) > 0 else 0
        db_image[np.isnan(db_image)] = min_val
        
        finite_pixels = db_image[np.isfinite(db_image)]
        finite_max = np.nanmax(finite_pixels) if len(finite_pixels) > 0 else 0
        db_image[np.isinf(db_image)] = finite_max + 1
        
        all_db_pixels.extend(db_image.flatten())
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{max_images} images...")

    all_db_pixels = np.array(all_db_pixels)
    p1, p99 = np.percentile(all_db_pixels, [1, 99])
    filtered_pixels = all_db_pixels[(all_db_pixels >= p1) & (all_db_pixels <= p99)]
    
    print("\nRunning K-Means Clustering to find thresholds automatically...")
    
    # Reshape for KMeans
    pixels = filtered_pixels.reshape(-1, 1)

    # Subsample pixels for KMeans to avoid long computation times
    sample_size = min(len(pixels), 100000)
    np.random.seed(42)
    sampled_pixels = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(sampled_pixels)

    # Cluster centers
    centers = np.sort(kmeans.cluster_centers_.flatten())

    print("\n===== AUTOMATIC THRESHOLDS =====")
    print("Cluster Centers (dB):")
    for i, c in enumerate(centers):
        print(f"Cluster {i+1}: {c:.3f} dB")

    # Thresholds = midpoint between neighboring clusters
    thresholds = []
    for i in range(len(centers)-1):
        t = (centers[i] + centers[i+1]) / 2
        thresholds.append(t)

    print("\nClassification Ranges (Copy these to train_unet.py):")
    print(f"Water      : [-np.inf, {thresholds[0]:.3f}]")
    print(f"Land       : [{thresholds[0]:.3f}, {thresholds[1]:.3f}]")
    print(f"Crop land  : [{thresholds[1]:.3f}, {thresholds[2]:.3f}]")
    print(f"Mountain   : [{thresholds[2]:.3f}, np.inf]")

if __name__ == "__main__":
    main()