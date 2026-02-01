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
# Product sub-paths
PROD_REF = "MergedReflectivityQCComposite"
PROD_TYPE = "PrecipFlag"

OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_files(product, count=10):
    """Scrapes the NCEP directory for the latest GRIB2 files."""
    url = f"{BASE_URL}{product}/"
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        # Find all .gz links, sort by name (timestamp)
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        links.sort()
        return [url + l for l in links[-count:]]
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []

def process_grib(url, product_name):
    """Downloads, decodes, and renders a GRIB2 file to PNG."""
    filename = url.split('/')[-1]
    local_gz = f"temp_{filename}"
    local_grib = local_gz.replace(".gz", "")
    
    # 1. Download
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        with open(local_gz, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # 2. Decompress (using gzip tool or python)
    os.system(f"gunzip -f {local_gz}")

    # 3. Open with Xarray + cfgrib
    try:
        ds = xr.open_dataset(local_grib, engine='cfgrib')
    except Exception as e:
        print(f"Failed to open GRIB: {e}")
        return None

    # Get raw data array
    # Variable names in MRMS GRIB2 vary slightly, usually unknown or paramId
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    
    # Get Bounds (Note: MRMS coords are usually Lat/Lon)
    min_lat = float(da.latitude.min())
    max_lat = float(da.latitude.max())
    min_lon = float(da.longitude.min())
    max_lon = float(da.longitude.max())
    bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # 4. Visualization Setup
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Colormap Logic
    if product_name == "Reflectivity":
        # Standard NWS Radar Colors
        cmap = plt.get_cmap('pyart_HomeyerRainbow') if 'pyart' in plt.colormaps() else 'nipy_spectral'
        # Filter low values (clutter)
        data = da.where(da > 0)
        norm = mcolors.Normalize(vmin=0, vmax=75)
    else: # PrecipType
        # Custom discrete map for MRMS PrecipFlag
        # 0=None, 1=Warm Strat, 2=Warm Conv, 3=Snow, 4=Conv Snow, 6=Hail, 7=Rain/Hail, etc.
        data = da.where(da >= 0)
        colors = ['#00000000', '#00ff00', '#ff0000', '#00ffff', '#0000ff', '#ff00ff', '#aaaaaa']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=7)

    ax.imshow(data, origin='upper', cmap=cmap, norm=norm, aspect='auto')
    
    # Save transparent PNG
    timestamp = filename.split('_')[-1].split('.')[0]
    out_name = f"{product_name}_{timestamp}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, transparent=True, dpi=150)
    plt.close()
    
    # Clean up temp files
    if os.path.exists(local_grib): os.remove(local_grib)
    
    return {
        "url": f"data/{out_name}",
        "bounds": bounds,
        "time": timestamp,
        "type": product_name
    }

def main():
    metadata = {"generated_at": str(datetime.now()), "files": []}
    
    # Clean old files
    for f in glob.glob(f"{OUTPUT_DIR}/*.png"):
        os.remove(f)

    # Process Reflectivity
    ref_files = get_latest_files(PROD_REF, 5)
    for url in ref_files:
        meta = process_grib(url, "Reflectivity")
        if meta: metadata["files"].append(meta)

    # Process Precip Type
    type_files = get_latest_files(PROD_TYPE, 5)
    for url in type_files:
        meta = process_grib(url, "PrecipType")
        if meta: metadata["files"].append(meta)

    # Save Metadata for Frontend
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    main()
