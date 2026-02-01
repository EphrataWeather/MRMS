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

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PROD_REF = "MergedReflectivityQCComposite"
PROD_TYPE = "PrecipFlag"
OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_files(product, count=10):
    url = f"{BASE_URL}{product}/"
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        links.sort()
        return [url + l for l in links[-count:]]
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []

def process_grib(url, product_name):
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    # 1. Download
    print(f"--- Processing {filename} ---")
    try:
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Download failed: {e}")
        return None
    
    # 2. Decompress
    os.system(f"gunzip -f {local_gz}")

    # 3. Open with Xarray
    try:
        # backend_kwargs helps ignore the "Truncating time" errors safely
        ds = xr.open_dataset(local_grib, engine='cfgrib', 
                             backend_kwargs={'errors': 'ignore'})
    except Exception as e:
        print(f"FAILED to open GRIB: {e}")
        if os.path.exists(local_grib): os.remove(local_grib)
        return None

    # Get Data
    try:
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # --- DEBUG LOGGING ---
        val_min = float(da.min().values)
        val_max = float(da.max().values)
        print(f"   Variable: {var_name}")
        print(f"   Data Range: {val_min} to {val_max}")
        # ---------------------

        min_lat = float(da.latitude.min())
        max_lat = float(da.latitude.max())
        min_lon = float(da.longitude.min())
        max_lon = float(da.longitude.max())
        
        # bounds = [[South, West], [North, East]]
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        print(f"   Bounds: {bounds}")

    except Exception as e:
        print(f"Error reading data vars: {e}")
        return None

    # 4. Visualization
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if product_name == "Reflectivity":
        # MRMS: -999 is missing, -99 is no precip.
        # We mask anything <= 0 to be transparent
        data = da.where(da > 0)
        
        # Standard Radar Colors (Reflectivity)
        # Using a simple nipy_spectral if pyart isn't installed
        cmap = 'nipy_spectral'
        norm = mcolors.Normalize(vmin=0, vmax=75)
    
    else: # PrecipType
        # MRMS Flags: -99 (No Data), 0 (No Precip), 1 (Warm Strat), 3 (Snow), 6 (Hail)
        # We mask < 0. We want to see 0 (No precip) as transparent? Usually yes.
        data = da.where(da > 0) 
        
        # Custom Colors for Precip Types
        # 1:Green(Rain), 3:Blue(Snow), 6:Red(Hail) - Simplified
        colors = ['#00ff00', '#00ff00', '#0000ff', '#0000ff', '#ff0000', '#ff0000', '#ff00ff']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=10)

    ax.imshow(data, origin='upper', cmap=cmap, norm=norm, aspect='auto')
    
    timestamp = filename.split('_')[-1].split('.')[0]
    out_name = f"{product_name}_{timestamp}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Cleanup
    if os.path.exists(local_grib): os.remove(local_grib)
    
    return {
        "url": f"data/{out_name}",
        "bounds": bounds,
        "time": timestamp,
        "type": product_name
    }

def main():
    metadata = {"generated_at": str(datetime.now()), "files": []}
    
    # Clean old
    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
        os.remove(f)

    # Process Reflectivity
    print("Fetching Reflectivity...")
    ref_files = get_latest_files(PROD_REF, 5)
    for url in ref_files:
        meta = process_grib(url, "Reflectivity")
        if meta: metadata["files"].append(meta)

    # Process Precip Type
    print("Fetching Precip Type...")
    type_files = get_latest_files(PROD_TYPE, 5)
    for url in type_files:
        meta = process_grib(url, "PrecipType")
        if meta: metadata["files"].append(meta)

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print("Done. Metadata saved.")

if __name__ == "__main__":
    main()
