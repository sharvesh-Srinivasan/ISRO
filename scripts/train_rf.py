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
C_UPPER = 5.0

def convert_to_db(data):
    data = np.where(data <= 0, 1e-9, data)
    return 10 * np.log10(data)

def stretch_to_8bit(data_db):
    data_db = np.nan_to_num(data_db, nan=C_LOWER)
    data_db = (data_db - C_LOWER) / (C_UPPER - C_LOWER) * 255
    data_db = np.clip(data_db, 0, 255)
    return data_db.astype(np.uint8)

def get_labels(composite_db):
    threshold_water    = [-np.inf, 10.142]
    threshold_land     = [10.142,  19.638]
    threshold_cropland = [19.638,  20.818]
    threshold_mountain = [20.818,  np.inf]

    labels = np.zeros(composite_db.shape, dtype=np.uint8)
    labels[(composite_db >= threshold_land[0]) & (composite_db <= threshold_land[1])] = 1
    labels[(composite_db > threshold_cropland[0]) & (composite_db <= threshold_cropland[1])] = 2
    labels[(composite_db > threshold_mountain[0])] = 3
    # Water remains 0
    return labels

def train_rf():
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
            
        # Synthesize composite
        synth_hv = hv * 1.5
        composite = (hh + hv + synth_hv) / 3.0
        composite_db = convert_to_db(composite)
        
        # Labels
        labels = get_labels(composite_db)
        
        # Features: HH_db, HV_db, Ratio
        hh_db = convert_to_db(hh)
        hv_db = convert_to_db(hv)
        ratio = hh_db / (hv_db + 1e-6)
        
        features = np.stack([hh_db, hv_db, ratio], axis=-1)
        
        # Flatten and append
        X_all.append(features.reshape(-1, 3))
        Y_all.append(labels.reshape(-1))
        
        if i > 0 and i % 5 == 0:
            print(f"Processed {i} tiles...")
            
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    
    print(f"\nExtracted {len(X_all)} total pixels.")
    
    # Decimate dataset to avoid RAM crash (sample 500,000 pixels)
    sample_size = min(500000, len(X_all))
    indices = np.random.choice(len(X_all), sample_size, replace=False)
    X_sampled = X_all[indices]
    Y_sampled = Y_all[indices]
    
    # Replace any NaNs
    X_sampled = np.nan_to_num(X_sampled)
    
    print(f"Training Random Forest on {sample_size} sampled pixels...")
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, Y_sampled, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    iou = jaccard_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean IoU: {iou:.4f}")
    
    model_path = r"D:\ISRO\Proj\model_rf.joblib"
    joblib.dump(rf, model_path)
    print(f"Saved Random Forest model to {model_path}")

if __name__ == "__main__":
    train_rf()
