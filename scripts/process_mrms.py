import os
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from datetime import datetime
from bs4 import BeautifulSoup
from PIL import Image
import json
import shutil

BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PRODUCT = "PrecipitationType"

OUTPUT_ROOT = "public/data"
SCANS_DIR = f"{OUTPUT_ROOT}/scans"
ROWS, COLS = 4, 4
MAX_SCANS = 5

os.makedirs(SCANS_DIR, exist_ok=True)

PRECIP_COLORS = mcolors.ListedColormap([
    (0, 0, 0, 0),        # 0 none
    (0, 1, 0, 0.8),      # 1 rain
    (0.4, 0.7, 1, 0.9),  # 2 snow
    (1, 0, 1, 0.9),      # 3 freezing rain
    (1, 0.6, 0, 0.9),    # 4 ice pellets
    (0.6, 0.6, 1, 0.9)  # 5 mix
])

def get_latest_files(count=MAX_SCANS):
    r = requests.get(f"{BASE_URL}{PRODUCT}/", timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    files = sorted(
        [a["href"] for a in soup.find_all("a") if a["href"].endswith(".grib2.gz")]
    )
    return [f"{BASE_URL}{PRODUCT}/{f}" for f in files[-count:]]

def render_and_tile(ds, out_dir):
    var = list(ds.data_vars)[0]
    da = ds[var]

    lats = da.latitude.values
    lons = np.where(da.longitude.values > 180,
                    da.longitude.values - 360,
                    da.longitude.values)

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    data = gaussian_filter(da.values, 0.3)

    fig = plt.figure(figsize=(25, 14), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.imshow(
        data,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        cmap=PRECIP_COLORS,
        norm=mcolors.BoundaryNorm(range(7), PRECIP_COLORS.N)
    )

    master = f"{out_dir}/master.png"
    plt.savefig(master, dpi=400, transparent=True, pad_inches=0)
    plt.close()

    img = Image.open(master)
    w, h = img.size
    tw, th = w // COLS, h // ROWS

    tiles = []
    for r in range(ROWS):
        for c in range(COLS):
            tile = img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            name = f"tile_{r}_{c}.png"
            tile.save(f"{out_dir}/{name}")
            tiles.append({"row": r, "col": c, "url": name})

    return {
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "tiles": tiles
    }

def process():
    scans = get_latest_files()

    if not scans:
        return

    shutil.rmtree(SCANS_DIR)
    os.makedirs(SCANS_DIR)

    metadata = []

    for i, url in enumerate(reversed(scans)):
        gz = f"tmp_{i}.gz"
        grib = gz.replace(".gz", "")

        with requests.get(url, stream=True) as r:
            with open(gz, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        os.system(f"gunzip -f {gz}")

        ds = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"errors": "ignore"})

        scan_dir = f"{SCANS_DIR}/scan_{i}"
        os.makedirs(scan_dir)

        info = render_and_tile(ds, scan_dir)
        info["time"] = url.split("_")[-1].split(".")[0]
        info["path"] = f"scans/scan_{i}"

        metadata.append(info)

        os.remove(grib)

    with open(f"{OUTPUT_ROOT}/metadata.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    process()
