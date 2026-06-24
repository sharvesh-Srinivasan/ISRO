import os
import numpy as np
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
import tensorflow as tf
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

def convert_to_db_and_stretch(data, lower_threshold, upper_threshold):
    # Handle zero/negative/NaN values for log10
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

def generate_mask_lib(db_image):
    # Define the threshold values for different land cover types in dB
    # Adjust these values based on your specific SAR dataset distribution
    threshold_water = [-np.inf, 10.142]  # Water threshold
    threshold_land = [10.142, 19.638]  # Land threshold
    threshold_cropland = [19.638, 20.818]  # Crop land threshold
    threshold_mountain = [20.818, np.inf]  # Mountain threshold

    # Generate the mask library based on the dB image and threshold values
    mask_water = (db_image >= threshold_water[0]) & (db_image <= threshold_water[1])
    mask_land = (db_image >= threshold_land[0]) & (db_image <= threshold_land[1])
    mask_cropland = (db_image >= threshold_cropland[0]) & (db_image <= threshold_cropland[1])
    mask_mountain = (db_image >= threshold_mountain[0]) & (db_image <= threshold_mountain[1])

    mask_lib = {
        'Water': mask_water.astype(int),
        'Land': mask_land.astype(int),
        'Crop land': mask_cropland.astype(int),
        'Mountain': mask_mountain.astype(int),
    }

    return mask_lib

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

def prepare_data():
    hh_folder = 'd:/ISRO/Proj/HH_tiles'
    hv_folder = 'd:/ISRO/Proj/HV_tiles'
    
    if not os.path.exists(hh_folder) or not os.path.exists(hv_folder):
        raise FileNotFoundError(f"Ensure {hh_folder} and {hv_folder} exist.")

    hh_files = os.listdir(hh_folder)
    hv_files = os.listdir(hv_folder)
    
    x = []
    z = []
    
    for hh_file, hv_file in zip(hh_files, hv_files):
        if not hh_file.endswith(('.tif', '.tiff')):
            continue
            
        hh_tif_file = os.path.join(hh_folder, hh_file)
        hv_tif_file = os.path.join(hv_folder, hv_file)
        
        if not os.path.exists(hv_tif_file):
            print(f"Skipping {hh_file} as matching HV file not found.")
            continue
    
        with rasterio.open(hh_tif_file) as hh_ds:
            hh_data = hh_ds.read(1)
            hh_transform = hh_ds.transform
            hh_crs = hh_ds.crs
    
        with rasterio.open(hv_tif_file) as hv_ds:
            hv_data = hv_ds.read(1)
            hv_transform = hv_ds.transform
            hv_crs = hv_ds.crs
    
        # Resize the HV image to match the dimensions of the HH image
        hv_data_resized = resize_hv_data(hv_data, hv_transform, hh_transform, hh_data.shape, hv_crs, hh_crs)
    
        # Set zero and negative values in hv_data_resized to a small positive value to avoid warnings
        hv_data_resized[hv_data_resized <= 0] = 1e-9
    
        # Calculate the HH / HV ratio and HH - HV difference
        hh_hv_ratio = hh_data / hv_data_resized
        hh_hv_diff = hh_data - hv_data_resized
    
        # Convert the data to dB units using the 'convert_to_db_and_stretch' function
        hh_hv_ratio_db = convert_to_db_and_stretch(hh_hv_ratio, -7, 3.5)
        hh_hv_diff_db = convert_to_db_and_stretch(hh_hv_diff, -7, 3.5)
        hv = convert_to_db_and_stretch(hv_data_resized, -7, 3.5)
    
        # Create gray-scale composite by combining the processed HH-HV ratio and HH-HV difference
        gray_composite = 0.33*hv + 0.33 * hh_hv_ratio_db + 0.33 * hh_hv_diff_db
        
        # Dataset Creation: Resize images to 256x256
        image = resize(gray_composite, (256, 256), anti_aliasing=True, mode='reflect')
        db_image = convert_to_db(image)
        
        # Handle NaN and Inf values in db_image
        min_val = np.nanmin(db_image) if len(db_image[~np.isnan(db_image)]) > 0 else 0
        db_image[np.isnan(db_image)] = min_val
        
        finite_pixels = db_image[np.isfinite(db_image)]
        finite_max = np.nanmax(finite_pixels) if len(finite_pixels) > 0 else 0
        db_image[np.isinf(db_image)] = finite_max + 1
        
        # Generate the mask library for the dB image
        mask_lib = generate_mask_lib(db_image)
        
        # Initialize an empty color mask image
        color_mask = np.zeros((db_image.shape[0], db_image.shape[1]), dtype=np.uint8)

        # Assign values for different land cover types in the color mask image
        color_mask[mask_lib['Water'] == 1] = 0
        color_mask[mask_lib['Land'] == 1] = 1
        color_mask[mask_lib['Crop land'] == 1] = 2
        color_mask[mask_lib['Mountain'] == 1] = 3
        
        x.append(image)
        z.append(color_mask)
        
    x = np.array(x)
    z = np.array(z)
    
    return x, z

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

def get_unet():
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def get_fcn():
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    # Convolutional layers (Encoder)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(s)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    
    # Upsampling layers (Decoder)
    u1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u1)
    u2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    
    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(u2)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

def get_vit():
    # A lightweight patch-based Transformer for segmentation
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    # Patch embedding (16x16 patches) -> resulting grid is 16x16 since IMG_HEIGHT=256
    patch_size = 16
    patches = tf.keras.layers.Conv2D(64, (patch_size, patch_size), strides=(patch_size, patch_size))(s)
    
    # Reshape for transformer: (batch, 256, 64)
    x = tf.keras.layers.Reshape((16 * 16, 64))(patches)
    
    # Simple Transformer Block
    for _ in range(2):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(64, activation=tf.nn.gelu)(x3)
        x = tf.keras.layers.Add()([x3, x2])
        
    # Reshape back to 2D feature map
    x = tf.keras.layers.Reshape((16, 16, 64))(x)
    
    # Decoder (upsampling to original resolution)
    x = tf.keras.layers.Conv2DTranspose(64, (patch_size, patch_size), strides=(patch_size, patch_size), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(x)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def main():
    print("Preparing data...")
    x, z = prepare_data()
    
    if len(x) == 0:
        print("No data found for training.")
        return

    # Expand dimensions of x to include channel
    x = np.expand_dims(x, axis=-1)
    
    print("Splitting dataset...")
    x_train, x_test, z_train, z_test = train_test_split(x, z, test_size=0.2, random_state=0)
    
    print("Converting masks to one-hot encoding...")
    z_train_one_hot = to_categorical(z_train, num_classes=4)
    z_test_one_hot = to_categorical(z_test, num_classes=4)
    
    print("Defining models to train...")
    models = {
        "U-Net": get_unet(),
        "CNN (FCN)": get_fcn(),
        "Vision Transformer": get_vit()
    }
    
    results = []
    EPOCHS = 2
    BATCH_SIZE = 4
    
    for model_name, model in models.items():
        print(f"\n{'='*40}\nTraining {model_name}...\n{'='*40}")
        model.compile(optimizer=tf.keras.optimizers.Adam(), 
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                      metrics=['categorical_accuracy'])

        model.fit(x_train, 
                  z_train_one_hot,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(x_test, z_test_one_hot)
                 )

        print(f"Evaluating {model_name}...")
        y_pred_prob = model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=-1).flatten()
        y_true = np.argmax(z_test_one_hot, axis=-1).flatten()
        
        acc = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Model": model_name,
            "Accuracy": f"{acc:.4f}",
            "Mean IoU": f"{iou:.4f}",
            "F1-Score": f"{f1:.4f}"
        })

        save_name = f'd:/ISRO/Proj/model_{model_name.split()[0].lower()}.h5'
        model.save(save_name)
        print(f"Model saved as {save_name}")
        
    print("\n" + "="*60)
    print(" "*20 + "EVALUATION MATRIX")
    print("="*60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
