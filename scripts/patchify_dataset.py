import os
import rasterio
from rasterio.windows import Window
import numpy as np

def patchify_large_tif(input_tif, output_folder, patch_size=256):
    """
    Slices a massive .tif file into physical 256x256 patches.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        width = src.width
        height = src.height
        
        print(f"Loaded {input_tif}: {width}x{height}")
        print(f"Slicing into {patch_size}x{patch_size} tiles...")
        
        tile_count = 0
        
        # Slide window across the entire image
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                
                # Create the window slice
                window = Window(x, y, patch_size, patch_size)
                
                # Ensure we don't grab weirdly shaped edge tiles
                if window.width != patch_size or window.height != patch_size:
                    continue
                
                # Read the data in this specific window
                patch_data = src.read(window=window)
                
                # Automatically delete/skip completely black tiles or Nodata
                if np.max(patch_data) <= 0 or np.isnan(patch_data).all() or np.all(patch_data == 0):
                    continue
                
                # Update the geospatial metadata for this specific tile
                kwargs = meta.copy()
                kwargs.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': src.window_transform(window)
                })
                
                # Save the tile
                out_filename = os.path.join(output_folder, f"tile_{y}_{x}.tif")
                with rasterio.open(out_filename, 'w', **kwargs) as dest:
                    dest.write(patch_data)
                    
                tile_count += 1
                if tile_count % 100 == 0:
                    print(f"Generated {tile_count} tiles...")
                    
        print(f"Finished! Generated {tile_count} physical tiles in {output_folder}")

if __name__ == "__main__":
    # Define your massive input files
    massive_hh_path = r"D:\ISRO\Proj\HH.tif"
    massive_hv_path = r"D:\ISRO\Proj\HV.tif"
    
    # Define where the chopped tiles should be saved
    hh_tiles_folder = r"D:\ISRO\Proj\HH_tiles_generated"
    hv_tiles_folder = r"D:\ISRO\Proj\HV_tiles_generated"
    
    # Run the patchifier!
    print("--- Patchifying HH ---")
    patchify_large_tif(massive_hh_path, hh_tiles_folder, patch_size=256)
    
    print("--- Patchifying HV ---")
    patchify_large_tif(massive_hv_path, hv_tiles_folder, patch_size=256)
