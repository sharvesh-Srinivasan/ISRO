import os
import numpy as np
import rasterio
import gc

def load_real_cov(filepath):
    with rasterio.open(filepath) as src:
        data = src.read(1)
    # Return as complex64 with 0 imaginary part
    return data.astype(np.complex64)

def load_complex_cov(filepath):
    with rasterio.open(filepath) as src:
        real_part = src.read(1)
        imag_part = src.read(2)
    return real_part + 1j * imag_part

def patchify_and_save(tensor, patch_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    height, width, channels = tensor.shape
    
    tile_count = 0
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # We want strictly patch_size x patch_size tiles, discard edges if necessary
            if y + patch_size > height or x + patch_size > width:
                continue
            
            patch = tensor[y:y+patch_size, x:x+patch_size, :]
            
            # Skip if all zeros or NaNs
            if np.all(np.abs(patch) <= 1e-9) or np.isnan(patch).all():
                continue
            
            # Save as numpy array to preserve complex64 type easily
            out_path = os.path.join(output_folder, f"quad_patch_{y}_{x}.npy")
            np.save(out_path, patch)
            
            tile_count += 1
            if tile_count % 50 == 0:
                print(f"Generated {tile_count} complex quad-pol tiles...")
                
    print(f"Finished! Generated {tile_count} quad-pol patches in {output_folder}")

if __name__ == "__main__":
    cov_dir = r"extracted_code/265020411/COV"
    output_dir = r"QuadPol_tiles"
    
    print("Loading Diagonals (Real)...")
    hhhh = load_real_cov(os.path.join(cov_dir, "imagery_HHHH.tif"))
    hvhv = load_real_cov(os.path.join(cov_dir, "imagery_HVHV.tif"))
    vvvv = load_real_cov(os.path.join(cov_dir, "imagery_VVVV.tif"))
    
    print("Loading Off-Diagonals (Complex)...")
    hhhv = load_complex_cov(os.path.join(cov_dir, "imagery_HHHV.tif"))
    hhvv = load_complex_cov(os.path.join(cov_dir, "imagery_HHVV.tif"))
    hvvv = load_complex_cov(os.path.join(cov_dir, "imagery_HVVV.tif"))
    
    print("Stacking into (H, W, 6) tensor...")
    # The order will be: HHHH, HVHV, VVVV, HHHV, HHVV, HVVV
    tensor = np.stack([hhhh, hvhv, vvvv, hhhv, hhvv, hvvv], axis=-1)
    
    # Free memory
    del hhhh, hvhv, vvvv, hhhv, hhvv, hvvv
    gc.collect()
    
    print(f"Total tensor shape: {tensor.shape}")
    print("Slicing into 256x256 tiles and saving...")
    patchify_and_save(tensor, 256, output_dir)
