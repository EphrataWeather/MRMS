import os
import gzip
import shutil
import datetime
import requests
import numpy as np
import pygrib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

# --- SETTINGS ---
# Using relative paths for GitHub Actions
OUTPUT_DIR = "public/data"
TEMP_DIR = "temp"

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def get_smart_colormaps():
    # Rain/Hail: Magenta used for >45dBZ
    rain_hex = ['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#ff00ff', '#ffffff']
    cmap_rain = mcolors.LinearSegmentedColormap.from_list("rain", rain_hex)
    # Mix: Indigo/Deep Purples
    mix_hex = ['#4b0082', '#910091', '#800080']
    cmap_mix = mcolors.LinearSegmentedColormap.from_list("mix", mix_hex)
    # Snow: Cyan/Blues
    snow_hex = ['#00ffff', '#80ffff', '#adc5ff', '#5a82ff']
    cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", snow_hex)
    return cmap_rain, cmap_mix, cmap_snow

def process_frame(dbz_array, flag_array, bounds, time_str):
    cmap_rain, cmap_mix, cmap_snow = get_smart_colormaps()
    norm = plt.Normalize(vmin=5, vmax=75)
    
    # Logic Overrides
    is_severe = (dbz_array > 45)
    mask_snow = (flag_array == 3) & (dbz_array > 10) & (~is_severe)
    mask_mix = ((flag_array == 6) | (flag_array == 7)) & (dbz_array > 10) & (~is_severe)
    mask_rain = (dbz_array > 5) & (~mask_snow) & (~mask_mix)

    h, w = dbz_array.shape
    rgba = np.zeros((h, w, 4))
    
    # Efficient painting
    if np.any(mask_rain): rgba[mask_rain] = cmap_rain(norm(dbz_array[mask_rain]))
    if np.any(mask_mix):  rgba[mask_mix] = cmap_mix(norm(dbz_array[mask_mix]))
    if np.any(mask_snow): rgba[mask_snow] = cmap_snow(norm(dbz_array[mask_snow]))

    # Shift existing frames
    for i in range(13, -1, -1):
        old_name = "master" if i == 0 else f"master_{i}"
        new_name = f"master_{i+1}"
        if os.path.exists(f"{OUTPUT_DIR}/{old_name}.png"):
            shutil.copyfile(f"{OUTPUT_DIR}/{old_name}.png", f"{OUTPUT_DIR}/{new_name}.png")
            shutil.copyfile(f"{OUTPUT_DIR}/{old_name}.json", f"{OUTPUT_DIR}/{new_name}.json")

    # Save new master
    plt.imsave(f"{OUTPUT_DIR}/master.png", rgba)
    with open(f"{OUTPUT_DIR}/master.json", "w") as f:
        json.dump({"time": time_str, "bounds": bounds}, f)

def run():
    now = datetime.datetime.utcnow()
    # Check 6-8 minutes back to ensure file availability
    check_time = now - datetime.timedelta(minutes=8)
    time_str = check_time.strftime("%Y%m%d-%H%M00")
    
    base_url = "https://mrms.ncep.noaa.gov/data/2D"
    urls = {
        "dbz": f"{base_url}/MergedReflectivityQC/MRMS_MergedReflectivityQC_00.50_{time_str}.grib2.gz",
        "flag": f"{base_url}/PrecipFlag/MRMS_PrecipFlag_00.50_{time_str}.grib2.gz"
    }

    files = {}
    for key, url in urls.items():
        local_gz = f"{TEMP_DIR}/{key}.gz"
        local_grib = f"{TEMP_DIR}/{key}.grib2"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                print(f"File not found on NOAA server: {key} ({time_str})")
                return
            with open(local_gz, 'wb') as f: f.write(r.content)
            with gzip.open(local_gz, 'rb') as f_in, open(local_grib, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            files[key] = local_grib
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return

    # Read Data using pygrib
    try:
        with pygrib.open(files['dbz']) as gb:
            msg = gb[1]
            dbz = msg.values
            lats, lons = msg.latlons()
        with pygrib.open(files['flag']) as gb:
            flag = gb[1].values

        bounds = [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
        display_time = check_time.strftime("%I:%M %p UTC")
        
        process_frame(dbz, flag, bounds, display_time)
        print(f"Successfully updated radar for {display_time}")
    except Exception as e:
        print(f"Error processing GRIB data: {e}")

if __name__ == "__main__":
    run()
