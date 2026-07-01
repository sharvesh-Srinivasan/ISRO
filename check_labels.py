import numpy as np

def check_dist(c_min, c_max, t_w, t_l, t_c):
    print(f"\n--- Stretch [{c_min}, {c_max}] with thresholds [{t_w}, {t_l}, {t_c}] ---")
    water_db = c_min + (10**(t_w/10) * (c_max - c_min)) / 255
    land_db = c_min + (10**(t_l/10) * (c_max - c_min)) / 255
    crop_db = c_min + (10**(t_c/10) * (c_max - c_min)) / 255
    print(f"Water boundary:  < {water_db:.1f} dB")
    print(f"Land boundary:   < {land_db:.1f} dB")
    print(f"Crop boundary:   < {crop_db:.1f} dB")
    print(f"Mountain:        > {crop_db:.1f} dB")

# What the user originally had for L-band (clamped to 67% water)
check_dist(-7.0, 3.5, 10.1, 19.6, 20.8)

# What it was when forced to C-band config (mostly mountain)
check_dist(-15.0, 5.0, 8.0, 15.0, 18.0)

# Proposed L-band balanced config
check_dist(-20.0, 5.0, 10.1, 19.6, 20.8)
