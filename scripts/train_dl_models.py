import os
import time
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

hh_folder = r'D:\ISRO\Proj\E04_HH_tiles'
hv_folder = r'D:\ISRO\Proj\E04_HV_tiles'

# C-Band stretches
C_LOWER = -20.0
C_UPPER = 5.0

def convert_to_db(data):
    data = np.where(data <= 0, 1e-9, data)
    return 10 * np.log10(data)

def stretch(data_db, c_min, c_max):
    data_db = np.nan_to_num(data_db, nan=c_min)
    data_db = (data_db - c_min) / (c_max - c_min) * 255
    data_db = np.clip(data_db, 0, 255)
    return data_db.astype(np.float32)

def get_labels(composite_db, t_w, t_l, t_c):
    threshold_water    = [-np.inf, t_w]
    threshold_land     = [t_w,  t_l]
    threshold_cropland = [t_l,  t_c]
    threshold_mountain = [t_c,  np.inf]

    labels = np.zeros(composite_db.shape, dtype=np.uint8)
    labels[(composite_db >= threshold_land[0]) & (composite_db <= threshold_land[1])] = 1
    labels[(composite_db > threshold_cropland[0]) & (composite_db <= threshold_cropland[1])] = 2
    labels[(composite_db > threshold_mountain[0])] = 3
    return labels

def load_dataset_for_band(hh_folder, hv_folder, feat_min, feat_max, label_min, label_max, t_w, t_l, t_c):
    print(f"Loading Dataset from {hh_folder}...")
    hh_files = sorted([f for f in os.listdir(hh_folder) if f.endswith('.tif')])
    X_all, Y_all = [], []
    
    for hh_file in hh_files:
        hv_file = hh_file.replace('HH', 'HV')
        hh_path = os.path.join(hh_folder, hh_file)
        hv_path = os.path.join(hv_folder, hv_file)
        
        if not os.path.exists(hv_path): continue
            
        with rasterio.open(hh_path) as src_hh: hh = src_hh.read(1).astype(np.float32)
        with rasterio.open(hv_path) as src_hv: hv = src_hv.read(1).astype(np.float32)
        
        # ML Feature Extraction exactly mapping to api.py logic
        hv_raw = np.where(hv <= 0, 1e-9, hv)
        ratio_raw = hh / hv_raw
        diff_raw = hh - hv
        
        ratio_db_ml = 10 * np.log10(np.where(ratio_raw <= 0, 1e-9, ratio_raw))
        diff_db_ml  = 10 * np.log10(np.where(diff_raw <= 0, 1e-9, diff_raw))
        hv_db_ml    = 10 * np.log10(hv_raw)
        
        # Use feature stretch bounds for the Neural Network Inputs (X)
        def stretch_local(data_db, s_min, s_max):
            data_db = np.nan_to_num(data_db, nan=s_min)
            data_db = (data_db - s_min) / (s_max - s_min) * 255
            data_db = np.clip(data_db, 0, 255)
            return data_db.astype(np.float32)

        ratio_str = stretch_local(ratio_db_ml, feat_min, feat_max)
        diff_str  = stretch_local(diff_db_ml, feat_min, feat_max)
        hv_str_ml = stretch_local(hv_db_ml, feat_min, feat_max)
        
        # Ground Truth Generation: Match api.py's label generation strictly using label_min/max
        ratio_str_lbl = stretch_local(ratio_db_ml, label_min, label_max)
        diff_str_lbl  = stretch_local(diff_db_ml, label_min, label_max)
        hv_str_lbl    = stretch_local(hv_db_ml, label_min, label_max)
        
        gray_composite_lbl = 0.33 * hv_str_lbl + 0.33 * ratio_str_lbl + 0.33 * diff_str_lbl
        db_img_for_mask = 10 * np.log10(np.where(gray_composite_lbl > 0, gray_composite_lbl, 1e-9))
        labels = get_labels(db_img_for_mask, t_w, t_l, t_c)
        
        gray_composite_ml = 0.33 * hv_str_ml + 0.33 * ratio_str + 0.33 * diff_str
        
        # Slice into 256x256 patches
        height, width = gray_composite_ml.shape
        patch_size = 256
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                if y + patch_size > height or x + patch_size > width:
                    continue
                patch_x = gray_composite_ml[y:y+patch_size, x:x+patch_size]
                patch_y = labels[y:y+patch_size, x:x+patch_size]
                
                # Keep training fast and balanced: skip completely empty/water tiles
                if np.sum(patch_y == 0) / (256*256) > 0.95:
                    if np.random.rand() > 0.05: # keep 5%
                        continue
                
                X_all.append(patch_x)
                Y_all.append(patch_y)
                
    X_arr = np.expand_dims(np.array(X_all), axis=-1) / 255.0 # Normalized 0-1
    Y_arr = np.array(Y_all)
    print(f"Dataset Shape: X={X_arr.shape}, Y={Y_arr.shape}")
    return X_arr, Y_arr

def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    # Down
    c1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(16, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    # Bottleneck
    c3 = Conv2D(32, 3, activation='relu', padding='same')(p2)
    # Up
    u4 = UpSampling2D()(c3)
    c4 = Conv2D(16, 3, activation='relu', padding='same')(concatenate([u4, c2]))
    u5 = UpSampling2D()(c4)
    c5 = Conv2D(8, 3, activation='relu', padding='same')(concatenate([u5, c1]))
    # Out
    outputs = Conv2D(4, 1, activation='softmax')(c5)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_shape=(256, 256, 1)):
    # Standard FCN
    inputs = Input(input_shape)
    x = Conv2D(16, 5, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 5, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(4, 1, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_vision_patch_model(input_shape=(256, 256, 1)):
    # Simulates blocky patch-wise processing
    inputs = Input(input_shape)
    x = Conv2D(16, 16, strides=16, activation='relu', padding='valid')(inputs) # Extracts 16x16 block features (16x16 output)
    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(4, 1, activation='linear')(x) # 16x16x4
    # Upsample back to 256x256 by simple nearest neighbor scaling to CREATE block artifacts natively!
    x = UpSampling2D(size=(16, 16), interpolation='nearest')(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_dl_for_band(band_name, hh_folder, hv_folder, feat_min, feat_max, label_min, label_max, t_w, t_l, t_c):
    print(f"\n{'='*40}\nStarting Deep Learning Pipeline for {band_name}-Band\n{'='*40}")
    
    X, Y = load_dataset_for_band(hh_folder, hv_folder, feat_min, feat_max, label_min, label_max, t_w, t_l, t_c)
    epochs = 3 # Fast training
    batch_size = 4
    
    print(f"\n--- Training {band_name}-Band U-Net ---")
    unet = build_unet()
    unet.fit(X, Y, epochs=epochs, batch_size=batch_size)
    unet.save(rf"D:\ISRO\Proj\model_u-net_{band_name}.h5")
    
    print(f"\n--- Training {band_name}-Band CNN ---")
    cnn = build_cnn()
    cnn.fit(X, Y, epochs=epochs, batch_size=batch_size)
    cnn.save(rf"D:\ISRO\Proj\model_cnn_{band_name}.h5")
    
    print(f"\n--- Training {band_name}-Band Vision ---")
    vision = build_vision_patch_model()
    vision.fit(X, Y, epochs=epochs, batch_size=batch_size)
    vision.save(rf"D:\ISRO\Proj\model_vision_{band_name}.h5")
    print(f"All {band_name}-Band models trained and saved successfully!")

if __name__ == "__main__":
    # C-Band uses identical bounds for both features and labels
    train_dl_for_band("C", r'D:\ISRO\Proj\C_Band\E04_HH_tiles', r'D:\ISRO\Proj\C_Band\E04_HV_tiles', -15.0, 5.0, -15.0, 5.0, 8.0, 15.0, 18.0)
    
    # L-Band features use -25.0 to 5.0, BUT the labels MUST be generated using the -20.0 to 5.0 stretch that the thresholds map to perfectly
    train_dl_for_band("L", r'D:\ISRO\Proj\L_Band\HH_tiles', r'D:\ISRO\Proj\L_Band\HV_tiles', -25.0, 5.0, -20.0, 5.0, 10.1, 19.6, 20.8)
