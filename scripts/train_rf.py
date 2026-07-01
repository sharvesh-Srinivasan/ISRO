import os
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import joblib

# Paths
hh_folder = r'D:\ISRO\Proj\E04_HH_tiles'
hv_folder = r'D:\ISRO\Proj\E04_HV_tiles'

# C-Band specific stretch
C_LOWER = -20.0
def convert_to_db(data):
    data = np.where(data <= 0, 1e-9, data)
    return 10 * np.log10(data)

def get_labels(composite_db, t_w, t_l, t_c):
    threshold_water    = [-np.inf, t_w]
    threshold_land     = [t_w,  t_l]
    threshold_cropland = [t_l,  t_c]
    threshold_mountain = [t_c,  np.inf]

    labels = np.zeros(composite_db.shape, dtype=np.uint8)
    labels[(composite_db >= threshold_land[0]) & (composite_db <= threshold_land[1])] = 1
    labels[(composite_db > threshold_cropland[0]) & (composite_db <= threshold_cropland[1])] = 2
    labels[(composite_db > threshold_mountain[0])] = 3
    # Water remains 0
    return labels

def train_rf_for_band(band_name, hh_folder, hv_folder, model_path, feat_min, feat_max, label_min, label_max, t_w, t_l, t_c):
    print(f"\n=============================================")
    print(f"Training Random Forest for {band_name}-Band")
    print(f"=============================================")
    
    hh_files = sorted([f for f in os.listdir(hh_folder) if f.endswith('.tif')])
    
    X_all = []
    Y_all = []
    
    print(f"Loading {len(hh_files)} tiles for feature extraction...")
    
    for i, hh_file in enumerate(hh_files):
        hv_file = hh_file.replace('HH', 'HV')
        hh_path = os.path.join(hh_folder, hh_file)
        hv_path = os.path.join(hv_folder, hv_file)
        
        if not os.path.exists(hv_path):
            continue
            
        with rasterio.open(hh_path) as src_hh:
            hh = src_hh.read(1).astype(np.float32)
        with rasterio.open(hv_path) as src_hv:
            hv = src_hv.read(1).astype(np.float32)
            
        # ML Feature Extraction exactly mapping to api.py logic
        hv_raw = np.where(hv <= 0, 1e-9, hv)
        ratio_raw = hh / hv_raw
        diff_raw = hh - hv
        
        ratio_db_ml = 10 * np.log10(np.where(ratio_raw <= 0, 1e-9, ratio_raw))
        diff_db_ml  = 10 * np.log10(np.where(diff_raw <= 0, 1e-9, diff_raw))
        hv_db_ml    = 10 * np.log10(hv_raw)
        
        # Use appropriate stretch depending on the band passed via arguments
        
        def stretch_local(data_db, s_min, s_max):
            data_db = np.nan_to_num(data_db, nan=s_min)
            data_db = (data_db - s_min) / (s_max - s_min) * 255
            data_db = np.clip(data_db, 0, 255)
            return data_db.astype(np.float32)

        ratio_str = stretch_local(ratio_db_ml, feat_min, feat_max)
        diff_str  = stretch_local(diff_db_ml, feat_min, feat_max)
        hv_str_ml = stretch_local(hv_db_ml, feat_min, feat_max)
        
        # Ground Truth Generation: Match api.py's math exactly using label bounds
        ratio_str_lbl = stretch_local(ratio_db_ml, label_min, label_max)
        diff_str_lbl  = stretch_local(diff_db_ml, label_min, label_max)
        hv_str_lbl    = stretch_local(hv_db_ml, label_min, label_max)
        
        gray_composite_lbl = 0.33 * hv_str_lbl + 0.33 * ratio_str_lbl + 0.33 * diff_str_lbl
        db_img_for_mask = 10 * np.log10(np.where(gray_composite_lbl > 0, gray_composite_lbl, 1e-9))
        labels = get_labels(db_img_for_mask, t_w, t_l, t_c)
        
        # Features: HH_db, HV_db, Ratio
        hh_db = convert_to_db(hh)
        hv_db = convert_to_db(hv)
        ratio = hh_db / (hv_db + 1e-6)
        
        features = np.stack([hh_db, hv_db, ratio], axis=-1)
        
        # Flatten
        features_flat = features.reshape(-1, 3)
        labels_flat = labels.reshape(-1)
        
        # Sample per tile to prevent memory crash
        tile_sample_size = min(5000, len(features_flat))
        if len(features_flat) > 0:
            indices = np.random.choice(len(features_flat), tile_sample_size, replace=False)
            X_all.append(features_flat[indices])
            Y_all.append(labels_flat[indices])
        
        if i > 0 and i % 50 == 0:
            print(f"Processed {i} tiles...")
            
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    
    print(f"\nExtracted {len(X_all)} total sampled pixels.")
    
    # Replace any NaNs
    X_sampled = np.nan_to_num(X_all)
    Y_sampled = Y_all
    
    print(f"Training Random Forest on {len(X_sampled)} sampled pixels...")
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, Y_sampled, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(rf, model_path)
    print("Done!")

if __name__ == "__main__":
    # C-Band
    train_rf_for_band("C", 
                      r'D:\ISRO\Proj\C_Band\E04_HH_tiles', 
                      r'D:\ISRO\Proj\C_Band\E04_HV_tiles', 
                      r'D:\ISRO\Proj\model_rf_C.joblib',
                      -15.0, 5.0, -15.0, 5.0, 8.0, 15.0, 18.0)
    
    # L-Band
    train_rf_for_band("L", 
                      r'D:\ISRO\Proj\L_Band\HH_tiles', 
                      r'D:\ISRO\Proj\L_Band\HV_tiles', 
                      r'D:\ISRO\Proj\model_rf_L.joblib',
                      -25.0, 5.0, -20.0, 5.0, 10.1, 19.6, 20.8)
