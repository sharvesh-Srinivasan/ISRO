# ============================================================
# train_stacking_e04.py  —  C-band (EOS-04) version
# ------------------------------------------------------------
# KEY DIFFERENCES from train_stacking.py (L-band):
#
#  1. INPUT FOLDERS  — reads from E04_HH_tiles / E04_HV_tiles,
#                       NOT HH_tiles / HV_tiles.
#                       These are completely separate physical datasets
#                       (EOS-04 C-band vs RISAT L-band).
#
#  2. STRETCH RANGE  — C_LOWER / C_UPPER replace the L-band
#                       hard-coded values of (-7, 3.5).
#                       C-band backscatter has a different dB range
#                       than L-band, so using L-band values clips the
#                       image and destroys contrast in the input features.
#                       Run find_thresholds_e04.py to get correct values.
#
#  3. LABEL THRESHOLDS — map_backscatter_to_label() uses C-band dB
#                       boundaries. L-band used (10.142 / 19.638 / 20.818).
#                       C-band has different absolute backscatter magnitudes
#                       due to its shorter wavelength (5.6 cm vs ~23 cm
#                       for L-band), so those L-band values will assign
#                       wrong class labels to C-band tiles.
#                       Run find_thresholds_e04.py to get correct values.
# ============================================================
import os
import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# CHANGE 1: C-BAND STRETCH RANGE
# ------------------------------------------------------------
# L-band used hard-coded (-7, 3.5) inside every
# convert_to_db_and_stretch() call. C-band (EOS-04) has
# different backscatter magnitudes — using L-band values clips
# most C-band pixels to 0 or 255, making the SVM/MLP features
# meaningless (no variation in the flattened input vectors).
#
# ACTION REQUIRED: Run find_thresholds_e04.py first.
# It prints the correct C_LOWER and C_UPPER for your C-band tiles.
# Paste those values here before training.
C_LOWER = -20.0   # <-- UPDATE: paste C_LOWER from find_thresholds_e04.py
C_UPPER =   5.0   # <-- UPDATE: paste C_UPPER from find_thresholds_e04.py
# (L-band equivalent was: lower=-7, upper=3.5)


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

def map_backscatter_to_label(backscatter):
    """
    Maps a single image's overall mean dB backscatter
    value to its most dominant land cover class.

    CHANGE 2: C-BAND LABEL THRESHOLDS
    -----------------------------------
    L-band used: <=10.142 -> Water, <=19.638 -> Land, <=20.818 -> Cropland
    These threshold values came from L-band backscatter statistics.
    C-band (EOS-04) has different absolute dB levels — the same
    boundaries will label most C-band tiles as the wrong class.

    ACTION REQUIRED: Run find_thresholds_e04.py on the E04 tiles.
    It outputs the correct C-band thresholds under "[3] STACKING THRESHOLDS".
    Paste those values into the if/elif conditions below.

    Example output from find_thresholds_e04.py:
        if backscatter <= X.XXX: return 0  # Water    <-- paste here
        elif backscatter <= X.XXX: return 1  # Land   <-- paste here
        elif backscatter <= X.XXX: return 2  # Crop   <-- paste here
        else: return 3  # Mountain                    <-- no change needed
    """
    # Currently still using L-band values — MUST be replaced after running find_thresholds_e04.py
    if backscatter <= 10.142:                    # <-- UPDATE with C-band value
        return 0  # Water
    elif 10.142 < backscatter <= 19.638:         # <-- UPDATE with C-band value
        return 1  # Land
    elif 19.638 < backscatter <= 20.818:         # <-- UPDATE with C-band value
        return 2  # Crop land
    else:
        return 3  # Mountain

def main():
    # CHANGE 3: C-band (EOS-04) tile folders — different from L-band
    # L-band: HH_tiles / HV_tiles (root of project)
    # C-band: E04_HH_tiles / E04_HV_tiles (generated by aoi_sub_e04.py)
    hh_folder = 'D:/ISRO/Proj/E04_HH_tiles'   # C-band HH polarization tiles
    hv_folder = 'D:/ISRO/Proj/E04_HV_tiles'   # C-band HV polarization tiles

    hh_files = os.listdir(hh_folder)
    hv_files = os.listdir(hv_folder)
    
    X = []
    y = []
    
    print("Step 1: Loading images and calculating overall image classes...")
    
    # Process up to 50 images for speed (increase if you want to use the full dataset)
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
        hh_hv_diff  = hh_data - hv_data_resized
    
        # CHANGE 4: Use C-band stretch range (C_LOWER, C_UPPER) instead of
        # L-band values (-7, 3.5). Using the wrong range clips C-band pixel
        # values and removes the feature variation that SVM/MLP rely on.
        hh_hv_ratio_db = convert_to_db_and_stretch(hh_hv_ratio,    C_LOWER, C_UPPER)  # was: (-7, 3.5)
        hh_hv_diff_db  = convert_to_db_and_stretch(hh_hv_diff,     C_LOWER, C_UPPER)  # was: (-7, 3.5)
        hv             = convert_to_db_and_stretch(hv_data_resized, C_LOWER, C_UPPER)  # was: (-7, 3.5)
    
        gray_composite = 0.33*hv + 0.33 * hh_hv_ratio_db + 0.33 * hh_hv_diff_db
        
        # Resize to 150x150 as defined in the Stacking Notebook
        image_150 = resize(gray_composite, (150, 150), anti_aliasing=True, mode='reflect')
        
        # Calculate overall mean db value and map to label using C-band thresholds
        db_image = convert_to_db(image_150)
        mean_db  = np.nanmean(db_image)
        
        label = map_backscatter_to_label(mean_db)
        
        X.append(image_150)
        y.append(label)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{max_images} images")

    X = np.array(X)
    y = np.array(y)
    
    print("\nStep 2: Data Augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    augmented_images = []
    augmented_labels = []
    num_augmented_samples = 3
    
    for i in range(len(X)):
        img   = X[i]
        label = y[i]
        img   = np.expand_dims(img, axis=-1)
        
        img_augmented_gen = datagen.flow(np.expand_dims(img, axis=0), batch_size=1, shuffle=False)
        
        for j in range(num_augmented_samples):
            img_augmented = next(img_augmented_gen)[0]
            # Resize back to (150, 150) to drop the channels dim
            img_augmented = resize(img_augmented.squeeze(), (150, 150))
            augmented_images.append(img_augmented)
            augmented_labels.append(label)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    print(f"Total augmented samples generated: {len(augmented_images)}")
    
    print("\nStep 3: Flattening data for Machine Learning models...")
    # Reshape for SVM/MLP (Flatten the 150x150 images into 1D arrays of 22,500 features)
    n_samples  = augmented_images.shape[0]
    n_features = 150 * 150
    X_flat     = augmented_images.reshape(n_samples, n_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_flat, augmented_labels, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    print("\nStep 4: Training Base Estimators and Stacking Model...")
    
    # 1. Define base estimators
    svm_clf = SVC(kernel='rbf', gamma='scale', C=1)
    mlp_clf = MLPClassifier(alpha=1, max_iter=1000, random_state=42)
    
    estimator_list = [
        ('svm', svm_clf),
        ('mlp', mlp_clf)
    ]
    
    # 2. Define stacking classifier (with Logistic Regression Meta-Model)
    stack_model = StackingClassifier(
        estimators=estimator_list, 
        final_estimator=LogisticRegression(max_iter=1000)
    )
    
    print("Fitting Stacking Classifier (This usually takes a minute or two)...")
    stack_model.fit(X_train, y_train)
    
    print("\nStep 5: Evaluating Model...")
    y_test_pred = stack_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    mcc      = matthews_corrcoef(y_test, y_test_pred)
    
    # Use weighted to handle potential class imbalances
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print('\n==================================')
    print('Final Stacking Model Performance (Test Set) — C-band (EOS-04)')
    print(f'- Accuracy: {accuracy:.4f}')
    print(f'- MCC:      {mcc:.4f}')
    print(f'- F1 score: {f1:.4f}')
    print('==================================')

if __name__ == '__main__':
    main()
