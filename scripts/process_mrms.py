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
OUTPUT_DIR = "public/data"
TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_0")
os.makedirs(TILE_DIR, exist_ok=True)

# --- ACCUWEATHER STYLE COLOR BINS ---
# Rain (Flag 1): Greens to Reds
rain_colors = ['#00FB90', '#00BB00', '#FFFF00', '#FF8C00', '#FF0000', '#B90000']
cmap_rain = mcolors.ListedColormap(rain_colors)

# Snow (Flag 2): Deep Blue to White (NO PURPLES HERE)
snow_colors = ['#00008B', '#0000FF', '#4169E1', '#ADD8E6', '#FFFFFF']
cmap_snow = mcolors.ListedColormap(snow_colors)

# Mix/Ice (Flag 3+): Hot Pinks and Purples
mix_colors = ['#FF69B4', '#FF00FF', '#9A00F6', '#4B0082']
cmap_mix = mcolors.ListedColormap(mix_colors)

def get_latest_urls(prod):
    url = f"https://mrms.ncep.noaa.gov/data/2D/{prod}/"
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
    ref_file = download_and_extract(get_latest_urls("MergedReflectivityQCComposite"), "ref")
    flag_file = download_and_extract(get_latest_urls("PrecipFlag"), "flag")
    
    if not ref_file or not flag_file: return

    ds_ref = xr.open_dataset(ref_file, engine='cfgrib')
    ds_flag = xr.open_dataset(flag_file, engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # High-Res 4K-style output
    fig = plt.figure(figsize=(24, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v = ref.values
    flag_v = flag.values
    ref_v[ref_v < 10] = np.nan 

    # Plotting using the professional binned color maps
    # Nearest interpolation prevents "blur" between rain and snow
    ax.imshow(np.where(flag_v == 1, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_rain, norm=mcolors.Normalize(10, 75), interpolation='nearest')
    ax.imshow(np.where(flag_v == 3, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_snow, norm=mcolors.Normalize(10, 50), interpolation='nearest')
    ax.imshow(np.where(flag_v >= 2, ref_v, np.nan), extent=ext, origin='upper', cmap=cmap_mix, norm=mcolors.Normalize(10, 50), interpolation='nearest')

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master_path, transparent=True, pad_inches=0)
    plt.close()

    # Create metadata
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%I:%M %p")
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    process()
