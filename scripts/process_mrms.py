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

# --- CONFIG ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
REF_PROD = "MergedReflectivityQCComposite"
FLAG_PROD = "PrecipFlag"
OUTPUT_DIR = "public/data"
# We always write to tiles_0; the YAML "Rotate" step moves them to 1, 2, 3 later
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")
os.makedirs(TILE_DIR, exist_ok=True)

# --- PROFESSIONAL COLOR BINS (AccuWeather Style) ---
# Rain: Green -> Yellow -> Red
rain_list = ['#007d00', '#00fb90', '#ffff00', '#ff8c00', '#ff0000', '#b90000']
cmap_rain = mcolors.LinearSegmentedColormap.from_list('rain', rain_list, N=15)

# Snow: Royal Blues -> White
# (Using deep blues to avoid confusion with Mix)
snow_list = ['#00008b', '#0000ff', '#4169e1', '#add8e6', '#ffffff']
cmap_snow = mcolors.LinearSegmentedColormap.from_list('snow', snow_list, N=10)

# Mix: Hot Pinks -> Purples
mix_list = ['#ff69b4', '#ff00ff', '#9a00f6', '#4b0082']
cmap_mix = mcolors.LinearSegmentedColormap.from_list('mix', mix_list, N=10)

def get_latest_urls(prod):
    url = f"{BASE_URL}{prod}/"
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        return [url + l for l in sorted(links, reverse=True)]
    except: return []

def download_and_extract(urls, name):
    for url in urls:
        fn = f"{name}.grib2.gz"
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(fn, 'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                os.system(f"gunzip -f {fn}")
                return f"{name}.grib2"
        except: continue
    return None

def slice_to_tiles(image_path, frame_dir):
    """Slices the high-res master into a 4x4 grid for fast web loading."""
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
            # Relative path for the web frontend
            tile_paths.append({"row": r, "col": c, "url": f"data/tiles_0/{name}"})
    return tile_paths

def process():
    ref_file = download_and_extract(get_latest_urls(REF_PROD), "ref")
    flag_file = download_and_extract(get_latest_urls(FLAG_PROD), "flag")
    
    if not ref_file or not flag_file:
        print("Data not available.")
        return

    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # High-Res Master Image (Sharp/Non-blurry)
    fig = plt.figure(figsize=(24, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v = ref.values
    flag_v = flag.values
    ref_v[ref_v < 10] = np.nan # Clean clutter

    # Draw layers
    ax.imshow(np.where(flag_v == 1, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_rain, norm=mcolors.Normalize(10, 75), interpolation='none')
    ax.imshow(np.where(flag_v == 2, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_snow, norm=mcolors.Normalize(10, 50), interpolation='none')
    ax.imshow(np.where(flag_v >= 3, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_mix, norm=mcolors.Normalize(10, 50), interpolation='none')

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master_path, transparent=True, pad_inches=0)
    plt.close()

    # Create the Tiles
    tiles = slice_to_tiles(master_path, TILE_DIR)

    # Metadata for this specific frame
    meta = {
        "tiles": tiles,
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%I:%M %p")
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)
    
    print(f"Frame 0 processed and tiled at {meta['time']}")

if __name__ == "__main__":
    process()
