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
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")
os.makedirs(TILE_DIR, exist_ok=True)

# --- PROFESSIONAL COLOR TABLES ---
# Rain: Green -> Yellow -> Red
cmap_rain = mcolors.LinearSegmentedColormap.from_list('rain', ["#00fb90", "#00bb00", "#ffff00", "#ff0000", "#b90000"], N=256)
# Snow: Light Blue -> Deep Blue -> White
cmap_snow = mcolors.LinearSegmentedColormap.from_list('snow', ["#00dcf5", "#005df5", "#002df5", "#ffffff"], N=256)
# Ice/Mix: Pink -> Bright Purple
cmap_mix = mcolors.LinearSegmentedColormap.from_list('mix', ["#ff00f6", "#9a00f6", "#5500a1"], N=256)

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

def process():
    ref_file = download_and_extract(get_latest_urls(REF_PROD), "ref")
    flag_file = download_and_extract(get_latest_urls(FLAG_PROD), "flag")
    
    if not ref_file or not flag_file: return

    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # HIGH RES FIGURE: Increase DPI and use a larger figsize for sharpness
    fig = plt.figure(figsize=(24, 12), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v = ref.values
    flag_v = flag.values
    
    # Filter out weak echoes (below 5dBZ) to keep the map clean
    ref_v[ref_v < 5] = np.nan

    # PLOT LAYERS with 'nearest' interpolation for sharpness
    # Rain (Flag 1)
    ax.imshow(np.where(flag_v == 1, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_rain, norm=mcolors.Normalize(5, 75), interpolation='nearest', zorder=1)
    # Snow (Flag 2)
    ax.imshow(np.where(flag_v == 2, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_snow, norm=mcolors.Normalize(5, 75), interpolation='nearest', zorder=2)
    # Mix/Ice (Flag 3, 4, etc)
    ax.imshow(np.where(flag_v >= 3, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_mix, norm=mcolors.Normalize(5, 75), interpolation='nearest', zorder=3)

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    # Save with high DPI to prevent the 'zoomed in' look
    plt.savefig(master_path, transparent=True, pad_inches=0, dpi=600)
    plt.close()

    # Generate metadata
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%H:%M UTC")
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    process()
