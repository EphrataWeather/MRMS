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

# Ensure directories exist
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
    """Slices a large master image into a grid of smaller tiles."""
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
    
    # 1. Download & Decompress
    print(f"Processing: {filename}")
    try:
        with requests.get(url, stream=True) as r:
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.system(f"gunzip -f {local_gz}")
    except Exception as e:
        print(f"Download error: {e}")
        return None

    # 2. Open with Xarray
    try:
        ds = xr.open_dataset(local_grib, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        
        # Longitude/Latitude Fix
        lats = da.latitude.values
        lons = da.longitude.values
        lons = np.where(lons > 180, lons - 360, lons) # 0-360 to -180-180
        
        bounds = [[float(np.min(lats)), float(np.min(lons))], 
                  [float(np.max(lats)), float(np.max(lons))]]

        # Prepare Data: Flip vertically to fix alignment, apply smooth blur
        raw_values = np.flipud(da.values) 
        smoothed = gaussian_filter(raw_values, sigma=1.0)
        # Mask out values below 1dBZ to keep background transparent
        data_to_plot = np.where(smoothed > 1.0, smoothed, np.nan)

    except Exception as e:
        print(f"Xarray error: {e}")
        return None

    # 3. Create High-Res Master Image
    fig = plt.figure(figsize=(24, 12), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # NOAA standard-ish reflectivity colormap
    cmap = 'nipy_spectral'
    norm = mcolors.Normalize(vmin=0, vmax=75)

    ax.imshow(data_to_plot, origin='upper', cmap=cmap, norm=norm, 
              aspect='auto', interpolation='bilinear')
    
    master_path = os.path.join(OUTPUT_DIR, "master_radar.png")
    plt.savefig(master_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    # 4. Create Tiles
    print("Slicing master image into tiles...")
    tiles = slice_to_tiles(master_path, rows=4, cols=4)
    
    # Cleanup
    if os.path.exists(local_grib): os.remove(local_grib)
    
    timestamp = filename.split('_')[-1].split('.')[0]
    return {
        "tiles": tiles,
        "bounds": bounds,
        "time": timestamp,
        "type": "Reflectivity"
    }

def main():
    # Clean up old tiles
    for f in glob.glob(f"{TILE_DIR}/*.png"):
        os.remove(f)

    print("Fetching latest MRMS data...")
    ref_files = get_latest_files(PROD_REF, 1)
    
    if not ref_files:
        print("No files found.")
        return

    result = process_grib(ref_files[0])
    
    if result:
        # Final JSON metadata for the frontend
        metadata = {
            "generated_at": str(datetime.now()),
            "latest": result
        }
        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump(metadata, f)
        print("Workflow Complete. Metadata and tiles saved.")

if __name__ == "__main__":
    main()
