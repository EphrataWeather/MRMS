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

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PROD_REF = "MergedReflectivityQCComposite"
OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_grib(url, product_name):
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    # Download & Decompress
    try:
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.system(f"gunzip -f {local_gz}")
    except: return None

    try:
        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # 1. FIX POSITION: Extract bounds and flip data if needed
        lats = da.latitude.values
        lons = da.longitude.values
        
        # Correct Longitude (0-360 to -180-180)
        lons = np.where(lons > 180, lons - 360, lons)
        
        bounds = [[float(np.min(lats)), float(np.min(lons))], 
                  [float(np.max(lats)), float(np.max(lons))]]

        # 2. SMOOTHING
        raw_data = da.values
        # Flip the array vertically to fix the "too north" / upside down issue
        raw_data = np.flipud(raw_data) 
        
        smoothed = gaussian_filter(raw_data, sigma=1.0)
        data_to_plot = np.where(smoothed > 0.5, smoothed, np.nan)

    except Exception as e:
        print(f"Error: {e}")
        return None

    # 3. GENERATE HIGH-RES IMAGE
    # We use a larger figure size to simulate "tiled" detail
    fig = plt.figure(figsize=(20, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    cmap = 'nipy_spectral'
    norm = mcolors.Normalize(vmin=0, vmax=75)

    ax.imshow(data_to_plot, origin='upper', cmap=cmap, norm=norm, aspect='auto', interpolation='bilinear')
    
    timestamp = filename.split('_')[-1].split('.')[0]
    out_name = f"{product_name}_{timestamp}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Save at high DPI for "Zoom-in" detail
    plt.savefig(out_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    if os.path.exists(local_grib): os.remove(local_grib)
    
    return {"url": f"data/{out_name}", "bounds": bounds, "time": timestamp}

# ... (main function logic remains similar to previous versions)
