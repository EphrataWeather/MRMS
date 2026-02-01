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
from scipy.ndimage import gaussian_filter # Added for smoothing

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
    
    # 1. Download & Decompress
    try:
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.system(f"gunzip -f {local_gz}")
    except Exception as e:
        print(f"Download/Gzip error: {e}")
        return None

    # 2. Open Data
    try:
        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # Longitude Correction
        min_lat, max_lat = float(da.latitude.min()), float(da.latitude.max())
        min_lon, max_lon = float(da.longitude.min()), float(da.longitude.max())
        if min_lon > 180: min_lon -= 360
        if max_lon > 180: max_lon -= 360
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # Data Smoothing (The "Professional" Look)
        # We replace NaNs with 0, smooth, then mask 0s back to transparent
        raw_values = da.values
        smoothed_values = gaussian_filter(raw_values, sigma=0.8) # Adjust sigma for more/less blur
        data_to_plot = np.where(smoothed_values > 0.5, smoothed_values, np.nan)

    except Exception as e:
        print(f"Data processing error: {e}")
        return None

    # 3. High-Res Visualization
    # Increase figsize for more base pixels
    fig = plt.figure(figsize=(12, 12), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if product_name == "Reflectivity":
        cmap = 'nipy_spectral'
        norm = mcolors.Normalize(vmin=0, vmax=75)
    else:
        colors = ['#00ff00', '#00ff00', '#0000ff', '#0000ff', '#ff0000', '#ff0000', '#ff00ff']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=10)

    # interpolation='bilinear' makes the pixels blend together smoothly
    ax.imshow(data_to_plot, origin='upper', cmap=cmap, norm=norm, 
              aspect='auto', interpolation='bilinear')
    
    timestamp = filename.split('_')[-1].split('.')[0]
    out_name = f"{product_name}_{timestamp}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    # Save with higher DPI (300 is high-quality, 600 is print-quality)
    plt.savefig(out_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    if os.path.exists(local_grib): os.remove(local_grib)
    
    return {"url": f"data/{out_name}", "bounds": bounds, "time": timestamp, "type": product_name}

# ... (rest of main remains the same) ...
def main():
    metadata = {"generated_at": str(datetime.now()), "files": []}
    for f in glob.glob(f"{OUTPUT_DIR}/*.png"): os.remove(f)
    
    print("Fetching Reflectivity...")
    ref_files = get_latest_files(PROD_REF, 5)
    for url in ref_files:
        meta = process_grib(url, "Reflectivity")
        if meta: metadata["files"].append(meta)

    print("Fetching Precip Type...")
    type_files = get_latest_files(PROD_TYPE, 5)
    for url in type_files:
        meta = process_grib(url, "PrecipType")
        if meta: metadata["files"].append(meta)

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)
    print("Done.")

if __name__ == "__main__":
    main()
