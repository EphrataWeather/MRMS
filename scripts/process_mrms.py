import os
import requests
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
import json
import glob
from bs4 import BeautifulSoup
from scipy.ndimage import gaussian_filter
from PIL import Image

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PROD_REF = "MergedReflectivityQCComposite"
OUTPUT_DIR = "public/data"
TILE_DIR = "public/data/tiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)

def get_latest_files(product, count=1):
    url = f"{BASE_URL}{product}/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        links.sort()
        return [url + l for l in links[-count:]]
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []

def slice_to_tiles(image_path, rows=4, cols=4):
    img = Image.open(image_path)
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    
    tile_data = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w if c < cols - 1 else w
            lower = upper + tile_h if r < rows - 1 else h
            
            tile = img.crop((left, upper, right, lower))
            tile_name = f"tile_{r}_{c}.png"
            path = os.path.join(TILE_DIR, tile_name)
            tile.save(path)
            tile_data.append({"row": r, "col": c, "url": f"data/tiles/{tile_name}"})
    return tile_data

def process_grib(url):
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    try:
        # Download
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        os.system(f"gunzip -f {local_gz}")

        # Open Data
        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # Coordinate Fix (0-360 to -180-180)
        lats = da.latitude.values
        lons = da.longitude.values
        lons = np.where(lons > 180, lons - 360, lons)
        
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        # DATA ALIGNMENT FIX: 
        # 1. Flip the array vertically to fix the "too south/mirrored" issue
        # 2. Apply smoothing
        data_fixed = np.flipud(da.values)
        smoothed = gaussian_filter(data_fixed, sigma=0.8)
        data_to_plot = np.where(smoothed > 0.5, smoothed, np.nan)

        # Generate High-Res Image
        fig = plt.figure(figsize=(24, 12), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # origin='upper' ensures top of array is top of image
        ax.imshow(data_to_plot, origin='upper', cmap='nipy_spectral', 
                  norm=mcolors.Normalize(0, 75), aspect='auto')
        
        master_path = os.path.join(OUTPUT_DIR, "master_radar.png")
        plt.savefig(master_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        tiles = slice_to_tiles(master_path)
        timestamp = filename.split('_')[-1].split('.')[0]
        
        return {"tiles": tiles, "bounds": bounds, "time": timestamp}

    finally:
        if os.path.exists(local_grib): os.remove(local_grib)
        if os.path.exists(local_gz): os.remove(local_gz)

def main():
    for f in glob.glob(f"{TILE_DIR}/*.png"): os.remove(f)
    ref_files = get_latest_files(PROD_REF, 1)
    if ref_files:
        result = process_grib(ref_files[0])
        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump({"latest": result}, f)

if __name__ == "__main__":
    main()
