import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_smart_colormaps():
    """
    Using your requested hex codes, but re-organizing them 
    by meteorological logic to prevent Hail/Mix confusion.
    """
    # RAIN & HAIL: Greens -> Yellow -> Reds -> User's Magenta Mix codes for Hail
    rain_hail_hex = [
        '#00fb90', '#00bb00', '#008800', # Rain
        '#ffff00', '#ff9100', '#ff0000', # Heavy Rain
        '#ff00ff', '#d100d1', '#ffffff'  # HAIL/EXTREME (User Mix colors used for Hail)
    ]
    cmap_rain = mcolors.LinearSegmentedColormap.from_list("rain_hail", rain_hail_hex)

    # WINTER MIX: Deeper Purples/Indigos (Distinct from Magenta)
    mix_hex = ['#910091', '#4b0082', '#4b0082']
    cmap_mix = mcolors.LinearSegmentedColormap.from_list("winter_mix", mix_hex)

    # SNOW: Your requested Cyan/Blue/White palette
    snow_hex = ['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff']
    cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", snow_hex)

    return cmap_rain, cmap_mix, cmap_snow

def create_composite(dbz_array, flag_array, index, bounds, timestamp_str):
    """
    dbz_array: 2D array of reflectivity
    flag_array: 2D array of MRMS precip types (3=Snow, 6=Mix, 1=Rain)
    """
    cmap_rain, cmap_mix, cmap_snow = get_smart_colormaps()
    norm = plt.Normalize(vmin=5, vmax=75) # Ignore everything below 5dBZ
    
    h, w = dbz_array.shape
    final_rgba = np.zeros((h, w, 4)) 

    # --- THE LOGIC FIXES ---
    
    # 1. HAIL OVERRIDE: If intensity > 45dBZ, it's always Rain/Hail colormap
    is_severe = (dbz_array > 45)

    # 2. FILTERED SNOW: Only trust Snow flag if intensity > 10dBZ (kills warm-air clutter)
    mask_snow = (flag_array == 3) & (dbz_array > 10) & (~is_severe)

    # 3. FILTERED MIX: Only trust Mix flag if intensity > 10dBZ and < 45dBZ
    mask_mix = ((flag_array == 6) | (flag_array == 7)) & (dbz_array > 10) & (~is_severe)

    # 4. RAIN: Default for everything else above 5dBZ
    mask_rain = (dbz_array > 5) & (~mask_snow) & (~mask_mix)

    # --- PAINTING THE IMAGE ---
    
    # Apply Rain/Hail Map
    rain_colors = cmap_rain(norm(dbz_array))
    final_rgba[mask_rain] = rain_colors[mask_rain]

    # Apply Mix Map
    mix_colors = cmap_mix(norm(dbz_array))
    final_rgba[mask_mix] = mix_colors[mask_mix]

    # Apply Snow Map
    snow_colors = cmap_snow(norm(dbz_array))
    final_rgba[mask_snow] = snow_colors[mask_snow]

    # --- SAVE OUTPUT ---
    prefix = "master" if index == 0 else f"master_{index}"
    
    # Save Image
    plt.imsave(f"{OUTPUT_DIR}/{prefix}.png", final_rgba)

    # Save Metadata
    meta = {
        "time": timestamp_str,
        "bounds": bounds
    }
    with open(f"{OUTPUT_DIR}/{prefix}.json", "w") as f:
        json.dump(meta, f)

    print(f"Processed frame {index}")

# Usage Example (Inside your download loop):
# create_composite(data_dbz, data_flags, i, [[lat1, lon1], [lat2, lon2]], "12:45 PM")
