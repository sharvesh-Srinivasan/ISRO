import os
import sys
import glob
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import the CVNN-PolSAR model
import own_unet

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import cvnn
    CVNN_AVAILABLE = True
except ImportError:
    CVNN_AVAILABLE = False
    print("WARNING: cvnn library not found. Please run 'pip install cvnn' before executing this script.")

def generate_pseudo_mask(tile):
    """
    Generates a 4-class pseudo-mask from a 6-channel Quad-Pol tile.
    Uses Span (Total Power) thresholding to create classes.
    """
    span = np.real(tile[..., 0]) + 2 * np.real(tile[..., 1]) + np.real(tile[..., 2])
    span_db = 10 * np.log10(np.clip(span, 1e-9, None))
    
    # Apply Gaussian filter to smooth out speckle noise
    # Increased sigma to 5.0 for massive label smoothing
    smoothed_db = gaussian_filter(span_db, sigma=5.0)
    
    p25 = np.percentile(smoothed_db, 25)
    p50 = np.percentile(smoothed_db, 50)
    p75 = np.percentile(smoothed_db, 75)
    
    mask = np.zeros_like(smoothed_db, dtype=np.uint8)
    mask[smoothed_db <= p25] = 0
    mask[(smoothed_db > p25) & (smoothed_db <= p50)] = 1
    mask[(smoothed_db > p50) & (smoothed_db <= p75)] = 2
    mask[smoothed_db > p75] = 3
    
    return mask

def augment_data(X, y):
    """
    Artificially expand dataset by rotating and flipping.
    """
    print("Augmenting data...")
    X_aug, y_aug = [], []
    for i in range(len(X)):
        # Original
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Horizontal Flip
        X_aug.append(np.fliplr(X[i]))
        y_aug.append(np.fliplr(y[i]))
        
        # Vertical Flip
        X_aug.append(np.flipud(X[i]))
        y_aug.append(np.flipud(y[i]))
        
        # Rotate 90
        X_aug.append(np.rot90(X[i], k=1, axes=(0,1)))
        y_aug.append(np.rot90(y[i], k=1, axes=(0,1)))
        
    return np.stack(X_aug), np.stack(y_aug)

def normalize_complex_tensor(X):
    """
    Standardize complex tensor. Real and imaginary parts are standardized independently
    across the entire dataset for each channel.
    """
    print("Normalizing complex input tensor...")
    X_norm = np.zeros_like(X, dtype=np.complex64)
    for c in range(X.shape[-1]):
        real_part = np.real(X[..., c])
        imag_part = np.imag(X[..., c])
        
        real_mean, real_std = np.mean(real_part), np.std(real_part)
        imag_mean, imag_std = np.mean(imag_part), np.std(imag_part)
        
        # Avoid division by zero
        real_std = real_std if real_std > 1e-9 else 1.0
        imag_std = imag_std if imag_std > 1e-9 else 1.0
        
        real_norm = (real_part - real_mean) / real_std
        imag_norm = (imag_part - imag_mean) / imag_std
        
        X_norm[..., c] = real_norm + 1j * imag_norm
        
    return X_norm

def load_quadpol_dataset(tiles_dir):
    npy_files = glob.glob(os.path.join(tiles_dir, "*.npy"))
    if not npy_files:
        return np.array([]), np.array([])
        
    X_list = []
    y_list = []
    
    print(f"Found {len(npy_files)} patches. Loading and generating pseudo-masks...")
    
    for f in npy_files:
        tile = np.load(f)
        if tile.shape != (256, 256, 6):
            continue
            
        mask = generate_pseudo_mask(tile)
        X_list.append(tile)
        y_list.append(mask)
        
    X = np.stack(X_list)
    y = np.stack(y_list)
    
    return X, y

def main():
    if not CVNN_AVAILABLE:
        print("Cannot train without CVNN library. Exiting.")
        return
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    tiles_dir = os.path.join(project_root, "QuadPol_tiles")
    
    print(f"Preparing Quad-Pol dataset from: {tiles_dir}")
    X, y = load_quadpol_dataset(tiles_dir)
    
    if len(X) == 0:
        print(f"No data found in {tiles_dir}.")
        return

    print(f"Raw Dataset shape: X={X.shape}, y={y.shape}")
    
    # 1. Normalize Data
    X = normalize_complex_tensor(X)
    
    # 2. Augment Data (1 patch becomes 4)
    X_aug, y_aug = augment_data(X, y)
    print(f"Augmented Dataset shape: X={X_aug.shape}, y={y_aug.shape}")
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
    
    print("Converting masks to one-hot encoding...")
    y_train_hot = to_categorical(y_train, num_classes=4)
    y_test_hot  = to_categorical(y_test,  num_classes=4)
    
    print("Defining CVNN-PolSAR U-Net model (Validation Winner)...")
    cvnn_model = own_unet.get_my_unet_tests(
        16, 
        input_shape=(256, 256, 6), 
        num_classes=4, 
        dtype=np.complex64, 
        tensorflow=False, 
        name="QuadPol_CVNN_UNet_Optimized"
    )
    
    # Override the hardcoded learning rate (1e-5) with a faster 1e-3
    cvnn_model.optimizer.learning_rate.assign(0.001)
    
    # 3. Callbacks for Smarter Training
    # Removed EarlyStopping at user request to force full training
    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss')
    ]
    
    print("Starting Training (forced 50 epochs)...")
    
    history = cvnn_model.fit(
        X_train, y_train_hot,
        validation_data=(X_test, y_test_hot),
        epochs=50,
        batch_size=8,
        verbose=1,
        callbacks=callbacks
    )
    
    model_path = os.path.join(project_root, "model_quadpol_cvnn.h5")
    cvnn_model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()
