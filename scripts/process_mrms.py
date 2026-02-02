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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- PROFESSIONAL WEATHER BRANDING COLORS ---
# Rain: Light Green -> Dark Green -> Yellow -> Orange -> Red -> Crimson
rain_colors = ['#00A500', '#007D00', '#005000', '#FFFF00', '#FF8C00', '#FF0000', '#B40000']
cmap_rain = mcolors.LinearSegmentedColormap.from_list('accu_rain', rain_colors, N=256)

# Snow: Light Blue -> Medium Blue -> Deep Royal Blue -> White
# (Using AccuWeather-style deep blues for snow)
snow_colors = ['#ADD8E6', '#87CEEB', '#0000FF', '#00008B', '#FFFFFF']
cmap_snow = mcolors.LinearSegmentedColormap.from_list('accu_snow', snow_colors, N=256)

# Mix/Ice: Light Pink -> Hot Pink -> Dark Purple
# (This clearly distinguishes sleet/freezing rain from pure snow)
mix_colors = ['#FFC0CB', '#FF69B4', '#FF00FF', '#800080']
cmap_mix = mcolors.LinearSegmentedColormap.from_list('accu_mix', mix_colors, N=256)

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
    # Interpolate flag to match ref grid exactly
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    # ULTRA HIGH RES FIGURE (Fixes "Zoomed in" look)
    # 30x15 at 300 DPI creates a massive 9000px wide image
    fig = plt.figure(figsize=(30, 15), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v = ref.values
    flag_v = flag.values
    
    # Hide noise (clutter) below 10 dBZ
    ref_v[ref_v < 10] = np.nan

    # PLOT LAYERS
    # interpolation='none' ensures no "blurry" blending between colors
    # Rain (Flag 1)
    ax.imshow(np.where(flag_v == 1, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_rain, norm=mcolors.Normalize(10, 75), interpolation='none', zorder=1)
    
    # Snow (Flag 2)
    ax.imshow(np.where(flag_v == 2, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_snow, norm=mcolors.Normalize(10, 50), interpolation='none', zorder=2)
    
    # Mix/Ice (Flag 3+)
    ax.imshow(np.where(flag_v >= 3, ref_v, np.nan), extent=ext, origin='upper', 
              cmap=cmap_mix, norm=mcolors.Normalize(10, 50), interpolation='none', zorder=3)

    master_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master_path, transparent=True, pad_inches=0)
    plt.close()

    # Create metadata for Leaflet
    meta = {
        "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]],
        "time": datetime.now().strftime("%I:%M %p UTC")
    }
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    process()
