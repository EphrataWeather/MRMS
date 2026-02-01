import os
import requests
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from datetime import datetime
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
REF_PROD = "MergedReflectivityQCComposite"
FLAG_PROD = "PrecipFlag"
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")

# Ensure directories exist
os.makedirs(TILE_DIR, exist_ok=True)

def get_latest_url(prod):
    """Scrapes the MRMS directory for the most recent GRIB2 file."""
    url = f"{BASE_URL}{prod}/"
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        if not links:
            raise Exception(f"No files found for {prod}")
        return url + sorted(links)[-1]
    except Exception as e:
        print(f"Error scraping {prod}: {e}")
        return None

def download_and_extract(url, name):
    """Downloads and unzips the GRIB2 file."""
    fn = f"{name}.grib2.gz"
    print(f"Downloading {name}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fn, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    os.system(f"gunzip -f {fn}")
    return f"{name}.grib2"

def slice_to_tiles(image_path, frame_dir):
    """Slices the master image into a 4x4 grid for faster web loading."""
    img = Image.open(image_path)
    w, h = img.size
    rows, cols = 4, 4
    tile_w, tile_h = w // cols, h // rows
    tile_paths = []
    
    for r in range(rows):
        for c in range(cols):
            left, upper = c * tile_w, r * tile_h
            right, lower = left + tile_w, upper + tile_h
            tile = img.crop((left, upper, right, lower))
            name = f"tile_{r}_{c}.png"
            tile.save(os.path.join(frame_dir, name))
            # The URL stored in metadata should be relative to the web root
            tile_paths.append({
                "row": r, 
                "col": c, 
                "url": f"data/tiles_0/{name}"
            })
    return tile_paths

def process():
    # 1. Fetch Latest Data
    ref_url = get_latest_url(REF_PROD)
    flag_url = get_latest_url(FLAG_PROD)
    
    if not ref_url or not flag_url:
        print("Could not find latest data. Exiting.")
        return

    ref_file = download_and_extract(ref_url, "ref")
    flag_file = download_and_extract(flag_url, "flag")
    
    # 2. Open Datasets
    # Note: cfgrib is required as the engine
    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    # Interpolate Flag grid to match Reflectivity grid resolution (1km)
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    # Convert 0-360 lon to -180 to 180
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # 3. Create the Visualization
    # Figure size matches a typical US aspect ratio
    fig = plt.figure(figsize=(20, 10))
    # add_axes([left, bottom, width, height]) at 0,0,1,1 removes all padding
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    
    # Force axes to exactly match data bounds (Fixes South Shift)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_data = ref.values
    flag_data = flag.values

    # Masks based on MRMS PrecipFlag: 1=Rain, 2=Snow, 3=Mix, 4=Ice
    rain_mask = (flag_data == 1) & (ref_data > 0)
    snow_mask = (flag_data == 2) & (ref_data > 0)
    mix_mask = (flag_data >= 3) & (ref_data > 0)

    # Plot Rain (Green/Yellow/Red)
    ax.imshow(np.where(rain_mask, ref_data, np.nan), 
              extent=ext, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75))
    
    # Plot Snow (Blues)
    ax.imshow(np.where(snow_mask, ref_data, np.nan), 
              extent=ext, origin='upper', cmap='Blues', norm=mcolors.Normalize(0, 75))
    
    # Plot Mix/Ice (Purples/Pinks)
    ax.imshow(np.where(mix_mask, ref_data, np.nan), 
              extent=ext, origin='upper', cmap='RdPu', norm=mcolors.Normalize(0, 75))

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    # pad_inches=0 is vital for alignment
    plt.savefig(master_path, transparent=True, pad_inches=0, dpi=400)
    plt.close()

    # 4. Generate Tiles and Metadata
    tiles = slice_to_tiles(master_path, TILE_DIR)
    
    metadata = {
        "tiles": tiles,
        "bounds": [
            [float(lats.min()), float(lons.min())], 
            [float(lats.max()), float(lons.max())]
        ],
        "time": datetime.now().strftime("%H:%M UTC"),
        "product": "MRMS P-Type Composite"
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(metadata, f)
    
    print(f"Successfully processed frame at {metadata['time']}")

if __name__ == "__main__":
    process()
