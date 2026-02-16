import time
import datetime
import requests
import gzip
import shutil
import os
import numpy as np
import pygrib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

# --- CONFIGURATION ---
OUTPUT_DIR = "public/data"
TEMP_DIR = "temp_mrms"
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- PART 1: THE SMART COLORMAPS (From previous fix) ---
def get_smart_colormaps():
    # Rain/Hail (Green -> Red -> Magenta -> White)
    rain_hex = ['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#ff00ff', '#ffffff']
    cmap_rain = mcolors.LinearSegmentedColormap.from_list("rain", rain_hex)
    
    # Mix (Deep Purple/Indigo - No Magenta)
    mix_hex = ['#4b0082', '#910091', '#800080']
    cmap_mix = mcolors.LinearSegmentedColormap.from_list("mix", mix_hex)
    
    # Snow (Cyan -> Blue)
    snow_hex = ['#00ffff', '#80ffff', '#adc5ff', '#5a82ff']
    cmap_snow = mcolors.LinearSegmentedColormap.from_list("snow", snow_hex)
    
    return cmap_rain, cmap_mix, cmap_snow

# --- PART 2: THE IMAGE GENERATOR (The Fix) ---
def generate_frame(dbz_array, flag_array, index, bounds, timestamp_str):
    print(f"   > Generating Frame {index}...")
    cmap_rain, cmap_mix, cmap_snow = get_smart_colormaps()
    norm = plt.Normalize(vmin=5, vmax=75)
    
    h, w = dbz_array.shape
    final_rgba = np.zeros((h, w, 4))

    # --- LOGIC RULES ---
    # 1. HAIL OVERRIDE: >45dBZ is always Rain/Hail
    is_severe = (dbz_array > 45)

    # 2. FILTERED SNOW: Flag 3, >10dBZ, not severe
    mask_snow = (flag_array == 3) & (dbz_array > 10) & (~is_severe)

    # 3. FILTERED MIX: Flag 6 or 7, >10dBZ, not severe
    mask_mix = ((flag_array == 6) | (flag_array == 7)) & (dbz_array > 10) & (~is_severe)

    # 4. RAIN: Everything else >5dBZ
    mask_rain = (dbz_array > 5) & (~mask_snow) & (~mask_mix)

    # --- PAINTING ---
    final_rgba[mask_rain] = cmap_rain(norm(dbz_array[mask_rain]))
    final_rgba[mask_mix] = cmap_mix(norm(dbz_array[mask_mix]))
    final_rgba[mask_snow] = cmap_snow(norm(dbz_array[mask_snow]))

    # Save
    prefix = "master" if index == 0 else f"master_{index}"
    plt.imsave(f"{OUTPUT_DIR}/{prefix}.png", final_rgba)
    
    # Metadata
    meta = { "time": timestamp_str, "bounds": bounds }
    with open(f"{OUTPUT_DIR}/{prefix}.json", "w") as f:
        json.dump(meta, f)

# --- PART 3: THE DOWNLOADER (The "Missing Link") ---
def download_file(url, local_filename):
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        return False

def get_latest_data():
    # MRMS updates every 2 mins. We try to fetch the file from ~2-4 mins ago to ensure it exists.
    # Note: MRMS file names use UTC time.
    now = datetime.datetime.utcnow()
    # Round down to even minute
    delta_min = now.minute % 2
    check_time = now - datetime.timedelta(minutes=4 + delta_min) 
    
    time_str = check_time.strftime("%Y%m%d-%H%M00")
    print(f"Checking for data at: {time_str} UTC...")

    # Define URLs (Reflectivity and Flags)
    # MRMS usually gzips their files (.grib2.gz)
    url_dbz = f"{BASE_URL}/MergedReflectivityQC/MRMS_MergedReflectivityQC_00.50_{time_str}.grib2.gz"
    url_flag = f"{BASE_URL}/PrecipFlag/MRMS_PrecipFlag_00.50_{time_str}.grib2.gz"

    file_dbz_gz = f"{TEMP_DIR}/dbz.grib2.gz"
    file_flag_gz = f"{TEMP_DIR}/flag.grib2.gz"
    file_dbz = f"{TEMP_DIR}/dbz.grib2"
    file_flag = f"{TEMP_DIR}/flag.grib2"

    # Download
    if not download_file(url_dbz, file_dbz_gz):
        print("   ! DBZ Download failed (File might not be ready yet)")
        return False
    if not download_file(url_flag, file_flag_gz):
        print("   ! Flag Download failed")
        return False

    # Unzip
    with gzip.open(file_dbz_gz, 'rb') as f_in:
        with open(file_dbz, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    with gzip.open(file_flag_gz, 'rb') as f_in:
        with open(file_flag, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Read GRIB2 Data
    grbs_dbz = pygrib.open(file_dbz)
    grbs_flag = pygrib.open(file_flag)

    msg_dbz = grbs_dbz[1] # First message
    msg_flag = grbs_flag[1]

    # Extract Data & Bounds
    dbz_data = msg_dbz.values
    flag_data = msg_flag.values
    lat, lon = msg_dbz.latlons()
    
    # Calculate Mapbox Bounds [Lat, Lon] -> [South, West, North, East]
    # MRMS is Top-Left origin usually.
    bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]

    # Format Time for Display (Convert UTC to EST roughly or keep UTC)
    display_time = check_time.strftime("%I:%M %p UTC")

    return dbz_data, flag_data, bounds, display_time

# --- PART 4: THE MAIN LOOP ---
def main():
    history_dbz = []
    history_flags = []
    history_meta = []

    print("--- MRMS Ingest System Started ---")
    
    while True:
        try:
            result = get_latest_data()
            
            if result:
                dbz, flag, bounds, time_str = result
                
                # Check if this is a new timestamp (compare with last known)
                if not history_meta or history_meta[-1]['time'] != time_str:
                    print(f"   * New Data Found! Processing {time_str}")
                    
                    history_dbz.append(dbz)
                    history_flags.append(flag)
                    history_meta.append({'time': time_str, 'bounds': bounds})

                    # Keep only last 15 frames
                    if len(history_dbz) > 15:
                        history_dbz.pop(0)
                        history_flags.pop(0)
                        history_meta.pop(0)

                    # RE-RENDER ALL FRAMES (To keep animation smooth)
                    # We render in reverse order so index 0 is always the "latest"
                    # logic for the frontend: Master.png = Latest
                    for i in range(len(history_dbz)):
                        # Inverse index: 0 = Latest, 14 = Oldest
                        real_idx = len(history_dbz) - 1 - i
                        generate_frame(
                            history_dbz[real_idx], 
                            history_flags[real_idx], 
                            i, # Output index (0 is master.png)
                            history_meta[real_idx]['bounds'],
                            history_meta[real_idx]['time']
                        )
                    
                    print("   * Update Complete.")
                else:
                    print("   . No new data yet.")
            
            # Wait 2 minutes before next check
            time.sleep(120)

        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            time.sleep(60) # Wait 1 min on crash before retry

if __name__ == "__main__":
    main()
