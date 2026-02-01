import os, requests, json, glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from PIL import Image
from datetime import datetime
from bs4 import BeautifulSoup

# --- CONFIG ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
REF_PROD = "MergedReflectivityQCComposite"
FLAG_PROD = "PrecipFlag"
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")

os.makedirs(TILE_DIR, exist_ok=True)

def get_latest(prod):
    url = f"{BASE_URL}{prod}/"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
    return url + sorted(links)[-1]

def download_and_extract(url, name):
    fn = f"{name}.grib2.gz"
    with requests.get(url, stream=True) as r:
        with open(fn, 'wb') as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    os.system(f"gunzip -f {fn}")
    return f"{name}.grib2"

def slice_to_tiles(image_path, frame_dir):
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
            tile_paths.append({"row": r, "col": c, "url": f"data/tiles_0/{name}"})
    return tile_paths

def process():
    # 1. Get Data
    ref_file = download_and_extract(get_latest(REF_PROD), "ref")
    flag_file = download_and_extract(get_latest(FLAG_PROD), "flag")
    
    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    # Interpolate Flag grid to match Reflectivity grid resolution
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # 2. Plotting
    fig = plt.figure(figsize=(20, 10), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ref_data = ref.values
    flag_data = flag.values

    # Precip Flags: 1=Rain, 2=Snow, 3=Mix/Sleet, 4=Ice
    rain_mask = (flag_data == 1) & (ref_data > 0)
    snow_mask = (flag_data == 2) & (ref_data > 0)
    mix_mask = (flag_data >= 3) & (ref_data > 0)

    # Plot Rain (Nipy Greens/Reds)
    ax.imshow(np.where(rain_mask, ref_data, np.nan), extent=ext, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75))
    # Plot Snow (Blues)
    ax.imshow(np.where(snow_mask, ref_data, np.nan), extent=ext, origin='upper', cmap='Blues', norm=mcolors.Normalize(0, 75))
    # Plot Mix (Purples/Pinks)
    ax.imshow(np.where(mix_mask, ref_data, np.nan), extent=ext, origin='upper', cmap='RdPu', norm=mcolors.Normalize(0, 75))

    master = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master, transparent=True, pad_inches=0, dpi=400)
    plt.close()

    # 3. Finalize
    tiles = slice_to_tiles(master, TILE_DIR)
    meta = {
        "tiles": tiles, 
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]], 
        "time": datetime.now().strftime("%H:%M UTC")
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    process()
