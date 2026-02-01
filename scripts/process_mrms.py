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
    except: return []

def slice_to_tiles(image_path, rows=4, cols=4):
    img = Image.open(image_path)
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    tile_data = []
    for r in range(rows):
        for c in range(cols):
            left, upper = c * tile_w, r * tile_h
            right = left + tile_w if c < cols-1 else w
            lower = upper + tile_h if r < rows-1 else h
            tile = img.crop((left, upper, right, lower))
            tile_name = f"tile_{r}_{c}.png"
            tile.save(os.path.join(TILE_DIR, tile_name))
            tile_data.append({"row": r, "col": c, "url": f"data/tiles/{tile_name}"})
    return tile_data

def process_grib(url):
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    try:
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        os.system(f"gunzip -f {local_gz}")

        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # Coordinate handling
        lats = da.latitude.values
        lons = np.where(da.longitude.values > 180, da.longitude.values - 360, da.longitude.values)
        
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())

        # FIX: We use 'origin=upper' in imshow later, so we use raw data here
        data_to_plot = da.values
        smoothed = gaussian_filter(data_to_plot, sigma=0.6)
        final_data = np.where(smoothed > 0.5, smoothed, np.nan)

        # High-Res Rendering with Fixed Aspect Ratio
        h_w_ratio = (lat_max - lat_min) / (lon_max - lon_min)
        fig = plt.figure(figsize=(25, 25 * h_w_ratio), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # CRITICAL FIX: origin='upper' + explicit extent prevents the "South Shift"
        ax.imshow(final_data, 
                  extent=[lon_min, lon_max, lat_min, lat_max], 
                  origin='upper', 
                  cmap='nipy_spectral', 
                  norm=mcolors.Normalize(0, 75), 
                  interpolation='bilinear')
        
        master_path = os.path.join(OUTPUT_DIR, "master_radar.png")
        # pad_inches=0 and NO bbox_inches='tight' preserves the geographic frame
        plt.savefig(master_path, transparent=True, pad_inches=0, dpi=450) 
        plt.close()

        tiles = slice_to_tiles(master_path)
        timestamp = filename.split('_')[-1].split('.')[0]
        return {"tiles": tiles, "bounds": [[lat_min, lon_min], [lat_max, lon_max]], "time": timestamp}
    
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
