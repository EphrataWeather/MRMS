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
os.makedirs(TILE_DIR, exist_ok=True)

def slice_to_tiles(image_path, rows=4, cols=4):
    img = Image.open(image_path)
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    
    tile_paths = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = left + tile_w
            lower = upper + tile_h
            
            tile = img.crop((left, upper, right, lower))
            tile_name = f"tile_{r}_{c}.png"
            path = os.path.join(TILE_DIR, tile_name)
            tile.save(path)
            tile_paths.append({"row": r, "col": c, "url": f"data/tiles/{tile_name}"})
    return tile_paths

def process_grib(url):
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    # Download
    with requests.get(url, stream=True) as r:
        with open(local_gz, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    os.system(f"gunzip -f {local_gz}")

    try:
        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # Longitude Fix & Flip for "Too North" issue
        lats, lons = da.latitude.values, da.longitude.values
        lons = np.where(lons > 180, lons - 360, lons)
        bounds = [[float(np.min(lats)), float(np.min(lons))], [float(np.max(lats)), float(np.max(lons))]]
        
        data = np.flipud(da.values)
        data = gaussian_filter(data, sigma=1.0)
        data = np.where(data > 0.5, data, np.nan)

        # Save Master High-Res Image
        fig = plt.figure(figsize=(24, 12), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75), aspect='auto', interpolation='bilinear')
        
        master_path = os.path.join(OUTPUT_DIR, "master_radar.png")
        plt.savefig(master_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        # Create Tiles
        tiles = slice_to_tiles(master_path)
        
        timestamp = filename.split('_')[-1].split('.')[0]
        return {"tiles": tiles, "bounds": bounds, "time": timestamp}

    finally:
        if os.path.exists(local_grib): os.remove(local_grib)

def main():
    ref_files = get_latest_files(PROD_REF, 1) # Just get the newest one
    if ref_files:
        result = process_grib(ref_files[0])
        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump(result, f)

# (Add your get_latest_files function here from previous steps)
