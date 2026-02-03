import os
import shutil
import requests
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from datetime import datetime
from bs4 import BeautifulSoup

# --- CONFIG ---
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")
os.makedirs(TILE_DIR, exist_ok=True)

# --- COLORS (Strict MRMS Flags) ---
rain_colors = ['#00FB90', '#00BB00', '#FFFF00', '#FF8C00', '#FF0000', '#B90000']
cmap_rain = mcolors.ListedColormap(rain_colors)
snow_colors = ['#00008B', '#0000FF', '#4169E1', '#ADD8E6', '#FFFFFF']
cmap_snow = mcolors.ListedColormap(snow_colors)
mix_colors = ['#FF69B4', '#FF00FF', '#9A00F6', '#4B0082']
cmap_mix = mcolors.ListedColormap(mix_colors)

def get_latest_urls(prod):
    url = f"https://mrms.ncep.noaa.gov/data/2D/{prod}/"
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        return [url + l for l in sorted(links, reverse=True)]
    except: return []

def download_and_extract(urls, name):
    for url in urls:
        fn = f"{name}.grib2.gz"
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(fn, 'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                os.system(f"gunzip -f {fn}")
                return f"{name}.grib2"
        except: continue
    return None

def slice_to_tiles(image_path, out_dir):
    img = Image.open(image_path)
    w, h = img.size
    # Create 4x4 tiles (adjust rows/cols if you want more detail)
    rows, cols = 4, 4 
    tile_w, tile_h = w // cols, h // rows
    
    os.makedirs(out_dir, exist_ok=True)
    
    for r in range(rows):
        for c in range(cols):
            left, upper = c * tile_w, r * tile_h
            right, lower = left + tile_w, upper + tile_h
            tile = img.crop((left, upper, right, lower))
            tile.save(os.path.join(out_dir, f"tile_{r}_{c}.png"))

def backfill_history(current_meta, current_master_path):
    """
    If history frames (1-9) are missing, fill them with the current frame
    so the website doesn't crash with 404 errors.
    """
    print("Checking history integrity...")
    for i in range(1, 10):
        meta_path = os.path.join(OUTPUT_DIR, f"metadata_{i}.json")
        img_path = os.path.join(OUTPUT_DIR, f"master_{i}.png")
        tile_path = os.path.join(OUTPUT_DIR, f"tiles_{i}")

        # If metadata is missing, we assume the whole frame is missing
        if not os.path.exists(meta_path):
            print(f"Backfilling missing history frame {i}...")
            
            # 1. Copy Metadata
            with open(meta_path, "w") as f:
                json.dump(current_meta, f)
            
            # 2. Copy Master Image
            shutil.copy(current_master_path, img_path)
            
            # 3. Copy Tiles (Recursive copy)
            if os.path.exists(tile_path):
                shutil.rmtree(tile_path)
            shutil.copytree(TILE_DIR, tile_path)

def process():
    print("Downloading Data...")
    ref_file = download_and_extract(get_latest_urls("MergedReflectivityQCComposite"), "ref")
    flag_file = download_and_extract(get_latest_urls("PrecipFlag"), "flag")
    
    if not ref_file or not flag_file:
        print("Failed to download data.")
        return

    print("Processing GRIB2 Data...")
    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    
    # Define Bounds
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Setup Plot
    fig = plt.figure(figsize=(24, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v = ref.values
    flag_v = flag.values
    
    # Filter Noise (< 10dBZ)
    ref_v[ref_v < 10] = np.nan 

    # --- Apply Flags ---
    rain_mask = np.isin(flag_v, [1, 6, 10, 91, 96])
    snow_mask = (flag_v == 3)
    mix_mask = (flag_v == 7)

    # Render Layers
    ax.imshow(np.where(rain_mask, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_rain, norm=mcolors.Normalize(10, 75), interpolation='nearest')
    ax.imshow(np.where(snow_mask, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_snow, norm=mcolors.Normalize(10, 50), interpolation='nearest')
    ax.imshow(np.where(mix_mask, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_mix, norm=mcolors.Normalize(10, 50), interpolation='nearest')

    # Save Master Frame
    master_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master_path, transparent=True, pad_inches=0)
    plt.close()

    # Generate Tiles from Master
    slice_to_tiles(master_path, TILE_DIR)

    # Save Metadata
    # IMPORTANT: cast numpy floats to python floats for JSON
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%I:%M %p"),
        "timestamp": datetime.now().timestamp()
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "metadata_0.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # --- FIX: Backfill History if Empty ---
    backfill_history(meta, master_path)

if __name__ == "__main__":
    process()
