import os, requests, json, glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from PIL import Image

# --- CONFIG ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/MergedReflectivityQCComposite/"
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0") # Frame 0 is always the newest

os.makedirs(TILE_DIR, exist_ok=True)

def get_latest_file():
    r = requests.get(BASE_URL, timeout=10)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
    return BASE_URL + sorted(links)[-1]

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
            tile_paths.append({"row": r, "col": c, "url": f"data/{os.path.basename(frame_dir)}/{name}"})
    return tile_paths

def process():
    url = get_latest_file()
    fn = "latest.grib2.gz"
    with requests.get(url, stream=True) as r:
        with open(fn, 'wb') as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    os.system(f"gunzip -f {fn}")
    
    ds = xr.open_dataset("latest.grib2", engine='cfgrib')
    da = ds[list(ds.data_vars)[0]]
    
    lats, lons = da.latitude.values, da.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # P-TYPE LOGIC: Simplify based on Latitude/Atmospheric estimate
    # In winter, lats > 40 typically see snow/mix at lower temps
    data = da.values
    temp_est = np.interp(lats, [25, 48], [20, -10]) # Rough C temp estimate based on Lat
    temp_grid = np.tile(temp_est[:, None], (1, len(lons)))

    fig = plt.figure(figsize=(20, 10), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Layer 1: Rain (Green/Yellow/Red)
    rain = np.where((temp_grid > 2) & (data > 0), data, np.nan)
    ax.imshow(rain, extent=ext, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75))

    # Layer 2: Mix (Pink/Purple)
    mix = np.where((temp_grid <= 2) & (temp_grid > -1) & (data > 0), data, np.nan)
    ax.imshow(mix, extent=ext, origin='upper', cmap='RdPu', norm=mcolors.Normalize(0, 75))

    # Layer 3: Snow (Blues/Whites)
    snow = np.where((temp_grid <= -1) & (data > 0), data, np.nan)
    ax.imshow(snow, extent=ext, origin='upper', cmap='Blues', norm=mcolors.Normalize(0, 75))

    master = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master, transparent=True, pad_inches=0, dpi=400)
    plt.close()

    tiles = slice_to_tiles(master, TILE_DIR)
    meta = {"tiles": tiles, "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]], "time": datetime.now().strftime("%H:%M Z")}
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    from datetime import datetime
    process()
