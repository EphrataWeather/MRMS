import os, requests, json, xr, numpy as np
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

def get_latest(prod):
    url = f"{BASE_URL}{prod}/"
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
    return url + sorted(links)[-1]

def process():
    # Download
    for p, n in [(REF_PROD, "ref"), (FLAG_PROD, "flag")]:
        with open(f"{n}.grib2.gz", 'wb') as f:
            f.write(requests.get(get_latest(p)).content)
        os.system(f"gunzip -f {n}.grib2.gz")

    ds_ref = xr.open_dataset("ref.grib2", engine='cfgrib')
    ds_flag = xr.open_dataset("flag.grib2", engine='cfgrib')
    
    ref = ds_ref[list(ds_ref.data_vars)[0]]
    flag = ds_flag[list(ds_flag.data_vars)[0]].interp_like(ref, method='nearest')

    lats, lons = ref.latitude.values, ref.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    
    # EXACT BOUNDS (Prevents Shifting)
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig = plt.figure(figsize=(20, 10), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])

    ref_v, flag_v = ref.values, flag.values
    
    # Layering
    ax.imshow(np.where((flag_v == 1) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='nipy_spectral', norm=mcolors.Normalize(0, 75))
    ax.imshow(np.where((flag_v == 2) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='Blues', norm=mcolors.Normalize(0, 75))
    ax.imshow(np.where((flag_v >= 3) & (ref_v > 0), ref_v, np.nan), extent=ext, origin='upper', cmap='RdPu', norm=mcolors.Normalize(0, 75))

    master = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(master, transparent=True, pad_inches=0, dpi=400)
    plt.close()

    # Tiling
    img = Image.open(master)
    w, h = img.size
    tile_paths = []
    for r in range(4):
        for c in range(4):
            tile = img.crop((c*(w//4), r*(h//4), (c+1)*(w//4), (r+1)*(h//4)))
            name = f"tile_{r}_{c}.png"
            tile.save(os.path.join(TILE_DIR, name))
            tile_paths.append({"row": r, "col": c, "url": f"data/tiles_0/{name}"})

    meta = {"tiles": tile_paths, "bounds": [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]], "time": datetime.now().strftime("%H:%M UTC")}
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    process()
