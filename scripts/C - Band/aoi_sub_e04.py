import os
import rasterio
from rasterio.windows import Window

# --------------------------------------
# TILE SIZE
# --------------------------------------
TILE_SIZE = 2000

# --------------------------------------
# INPUT FILES
# --------------------------------------
data_dir = r"D:\ISRO\Proj\E04_SAR_MRS_19JAN2026_019004720810_21695_STUC00ZTD_29967_23_DH_D_R_N12327_E075894"

files = [
    (os.path.join(data_dir, r"scene_HH\imagery_HH.tif"), r"D:\ISRO\Proj\E04_HH_tiles"),
    (os.path.join(data_dir, r"scene_HV\imagery_HV.tif"), r"D:\ISRO\Proj\E04_HV_tiles")
]

# --------------------------------------
# PROCESS EACH LAYER
# --------------------------------------
for input_file, output_folder in files:

    os.makedirs(output_folder, exist_ok=True)

    with rasterio.open(input_file) as src:

        width = src.width
        height = src.height

        print(f"\nProcessing {input_file}")
        print(f"Shape: {height} x {width}")

        tile_count = 0

        for row in range(0, height, TILE_SIZE):

            for col in range(0, width, TILE_SIZE):

                window_width = min(TILE_SIZE, width - col)
                window_height = min(TILE_SIZE, height - row)

                window = Window(
                    col,
                    row,
                    window_width,
                    window_height
                )

                transform = src.window_transform(window)

                data = src.read(
                    1,
                    window=window
                )

                output_file = os.path.join(
                    output_folder,
                    f"tile_r{row}_c{col}.tif"
                )

                with rasterio.open(
                    output_file,
                    "w",
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=transform
                ) as dst:

                    dst.write(data, 1)

                tile_count += 1

        print(f"Created {tile_count} tiles")
