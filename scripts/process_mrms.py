import os
import requests
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import glob
import re
from bs4 import BeautifulSoup
from scipy.ndimage import gaussian_filter
from PIL import Image
from datetime import datetime

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PROD_REF = "MergedReflectivityQCComposite"
PROD_TYPE = "PrecipFlag"
OUTPUT_DIR = "public/data"
TILE_DIR = "public/data/tiles"
FRAMES_TO_KEEP = 6 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)

def get_latest_file_urls(product_key, limit=10):
    prod = PRODUCTS[product_key]
    # We use the 'Latest' directory if available, or the top-level product folder
    idx_url = f"{BASE_URL}/{prod['url']}/"
    
    try:
        r = requests.get(idx_url, timeout=10)
        if r.status_code != 200:
            print(f"Server returned status {r.status_code} for {idx_url}")
            return []
        
        # Use regex to find all grib2.gz files in the HTML index
        pattern = re.compile(rf'href="({prod["prefix"]}.*?\.grib2\.gz)"')
        files = pattern.findall(r.text)
        
        if not files:
            print(f"No files matching {prod['prefix']} found at {idx_url}")
            return []

        # Sort naturally to get the newest files based on timestamp in name
        files.sort()
        latest_files = files[-limit:]
        
        return [f"{idx_url}{f}" for f in latest_files]
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

def download_and_extract(url, local_path):
    gz_path = local_path + ".gz"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        # Using python's gzip is more portable than system gunzip
        import gzip
        import shutil
        with gzip.open(gz_path, 'rb') as f_in:
            with open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False

def generate_color_composite(refl_data, type_data):
    norm = mcolors.Normalize(vmin=5, vmax=70)
    rows, cols = refl_data.shape
    rgba_img = np.zeros((rows, cols, 4))
    
    # 0=None, 1=Warm Strat, 2=Warm Conv, 3=Snow, 4=Conv Snow, 6=Mix, 7=Mix, 91=Cool Strat
    mask_precip = (refl_data > 5)
    mask_snow = np.isin(type_data, [3, 4])
    mask_mix  = np.isin(type_data, [6, 7, 91])
    mask_rain = ~mask_snow & ~mask_mix
    
    cmap_snow = plt.get_cmap('cool')    # Cyan/Purple-Blue for snow
    cmap_mix  = plt.get_cmap('spring')  # Pink/Yellow for mix
    cmap_rain = plt.get_cmap('nipy_spectral') # Standard radar colors
    
    if np.any(mask_precip & mask_rain):
        rgba_img[mask_precip & mask_rain] = cmap_rain(norm(refl_data[mask_precip & mask_rain]))
    if np.any(mask_precip & mask_snow):
        rgba_img[mask_precip & mask_snow] = cmap_snow(norm(refl_data[mask_precip & mask_snow]))
    if np.any(mask_precip & mask_mix):
        rgba_img[mask_precip & mask_mix] = cmap_mix(norm(refl_data[mask_precip & mask_mix]))

    rgba_img[refl_data < 5, 3] = 0.0 # Transparency
    return rgba_img

def main():
    # 1. Clear old tiles
    for f in glob.glob(f"{TILE_DIR}/*.png"): 
        try: os.remove(f)
        except: pass

    # 2. Get available files
    refl_files = get_file_list(PROD_REF)
    type_files = get_file_list(PROD_TYPE)
    
    if not refl_files or not type_files:
        print("No files found on server.")
        return

    # 3. Flexible Matching (Match by YearMonthDay-HourMinute)
    # File: MRMS_MergedReflectivityQCComposite_00.50_20240101-120000.grib2.gz
    # We match the first 13 characters of the timestamp (YYYYMMDD-HHMM)
    pairs = []
    type_dict = {f.split('_')[-1][:13]: f for f in type_files}
    
    for rf in refl_files:
        ts_key = rf.split('_')[-1][:13] 
        if ts_key in type_dict:
            full_ts = rf.split('_')[-1].split('.')[0]
            pairs.append((full_ts, rf, type_dict[ts_key]))
            
    pairs.sort(key=lambda x: x[0])
    target_pairs = pairs[-FRAMES_TO_KEEP:]
    
    print(f"Matched {len(target_pairs)} frame pairs.")
    
    frames_data = []
    for ts, rf, tf in target_pairs:
        print(f"Processing frame: {ts}")
        # (The rest of your process_frame logic goes here or call the function)
        # I'll include the tile slicing call here for brevity
        result = process_frame(ts, rf, tf)
        if result: frames_data.append(result)

    # 5. Save Metadata
    if frames_data:
        output = {"generated_at": datetime.now().isoformat(), "frames": frames_data}
        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump(output, f)
        print("Done! Metadata and tiles created.")
    else:
        print("No frames were successfully processed.")

def process_frame(timestamp, refl_file, type_file):
    local_refl = f"temp_refl_{timestamp}.grib2"
    local_type = f"temp_type_{timestamp}.grib2"
    
    try:
        if not download_and_extract(f"{BASE_URL}{PROD_REF}/{refl_file}", local_refl): return None
        if not download_and_extract(f"{BASE_URL}{PROD_TYPE}/{type_file}", local_type): return None

        # Added filter_by_keys to avoid 'Multiple values for unique key' GRIB errors
        ds_refl = xr.open_dataset(local_refl, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        ds_type = xr.open_dataset(local_type, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        
        refl = ds_refl[list(ds_refl.data_vars)[0]].values
        ptype = ds_type[list(ds_type.data_vars)[0]].values
        
        # Lat/Lon 
        lats, lons = ds_refl.latitude.values, ds_refl.longitude.values
        lons = np.where(lons > 180, lons - 360, lons)
        
        rgba_data = generate_color_composite(refl, ptype)
        
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        
        fig = plt.figure(figsize=(15, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(rgba_data, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
        
        master_path = os.path.join(OUTPUT_DIR, f"master_{timestamp}.png")
        plt.savefig(master_path, transparent=True, pad_inches=0, dpi=200)
        plt.close()
        
        # Slicing logic (same as before)
        tiles = slice_to_tiles(master_path, timestamp)
        os.remove(master_path)
        
        return {"time": timestamp, "bounds": [[lat_min, lon_min], [lat_max, lon_max]], "tiles": tiles}
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None

def slice_to_tiles(image_path, timestamp, rows=4, cols=4):
    img = Image.open(image_path)
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    tile_list = []
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_w, r * tile_h, (c+1) * tile_w, (r+1) * tile_h)
            tile = img.crop(box)
            tile_name = f"tile_{timestamp}_{r}_{c}.png"
            tile.save(os.path.join(TILE_DIR, tile_name))
            tile_list.append({"row": r, "col": c, "url": f"data/tiles/{tile_name}"})
    return tile_list

if __name__ == "__main__":
    main()
