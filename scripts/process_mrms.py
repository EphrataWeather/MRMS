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
import time

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
REF_PROD = "MergedReflectivityQCComposite"
FLAG_PROD = "PrecipFlag"
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")

os.makedirs(TILE_DIR, exist_ok=True)

def get_latest_urls(prod):
    """Scrapes the MRMS directory and returns a list of recent files."""
    url = f"{BASE_URL}{prod}/"
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        # Return full URLs sorted newest to oldest
        return [url + l for l in sorted(links, reverse=True)]
    except Exception as e:
        print(f"Error scraping {prod}: {e}")
        return []

def download_with_retry(urls, name):
    """Tries downloading the newest file; if 404, tries the next one."""
    for url in urls:
        fn = f"{name}.grib2.gz"
        print(f"Attempting to download {name}: {url}")
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(fn, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                os.system(f"gunzip -f {fn}")
                return f"{name}.grib2"
            else:
                print(f"Skipping {url} (Status: {r.status_code})")
        except Exception as e:
            print(f"Failed {url}: {e}")
    return None

def slice_to_tiles(image_path, frame_dir):
    img = Image.open(image_path)
    w, h = img.size
    rows, cols = 4, 4
    tile_w, tile_h = w // cols, h // rows
    tile_paths = []
    for r in range(rows):
        for c in range(cols):
            tile = img.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h))
            name = f"tile_{r}_{c}.png"
            tile.save(os.path.join(frame_dir, name))
            tile_paths.append({"row": r, "col": c, "url": f"data/tiles_0/{name}"})
    return tile_paths

def process():
    # Get lists of files
    ref_urls = get_latest_urls(REF_PROD)
    flag_urls = get_latest_urls(FLAG_PROD)
    
    if not ref_urls or not flag_urls:
        print("Could not find file lists.")
        return

    # Download with fallback logic
    ref_file = download_with_retry(ref_urls, "ref")
    flag_file = download_with_retry(flag_urls, "flag")
    
    if not ref_file or not flag_file:
        print("Failed to download one or both products.")
        return

    # Open Datasets
    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Visualization
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v, flag_v = ref.values, flag.values
    
    # Layering
    ax.imshow(np.where((flag_v == 1) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75))
    ax.imshow(np.where((flag_v == 2) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='Blues', norm=mcolors.Normalize(0, 75))
    ax.imshow(np.where((flag_v >= 3) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='RdPu', norm=mcolors.Normalize(0, 75))

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master_path, transparent=True, pad_inches=0, dpi=400)
    plt.close()

    # Tiling & Metadata
    tiles = slice_to_tiles(master_path, TILE_DIR)
    meta = {
        "tiles": tiles, 
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%H:%M UTC")
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)
    
    print(f"Success! Data saved at {meta['time']}")

if __name__ == "__main__":
    process()
