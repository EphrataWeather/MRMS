import os
import glob
import gzip
import shutil
import requests
import re
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration ---
IMG_DIR = "static/images"
DATA_DIR = "mrms_data"
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

PRODUCTS = {
    "reflectivity": {
        "url": "MergedReflectivityQCComposite",
        "prefix": "MRMS_MergedReflectivityQCComposite",
        "cmap": "jet",
        "vmin": 0, "vmax": 75
    },
    "precip": {
        "url": "PrecipRate",
        "prefix": "MRMS_PrecipRate",
        "cmap": "coolwarm",
        "vmin": 0, "vmax": 100
    }
}

def get_latest_file_urls(product_key, limit=10):
    prod = PRODUCTS[product_key]
    idx_url = f"{BASE_URL}/{prod['url']}/"
    try:
        r = requests.get(idx_url, timeout=15)
        pattern = re.compile(rf'href="({prod["prefix"]}.*?\.grib2\.gz)"')
        all_files = sorted(list(set(pattern.findall(r.text))))
        return [f"{idx_url}{f}" for f in all_files[-limit:]]
    except:
        return []

def process_grib_to_png(grib_path, out_path, product_key):
    try:
        unzipped = grib_path.replace(".gz", "")
        with gzip.open(grib_path, 'rb') as f_in, open(unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        ds = xr.open_dataset(unzipped, engine='cfgrib')
        data = ds[list(ds.data_vars)[0]].values
        prod = PRODUCTS[product_key]
        data = np.where(data < prod['vmin'], np.nan, data)

        plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        
        cmap = plt.get_cmap(prod['cmap']).copy()
        cmap.set_bad(color='none') 
        ax.imshow(data, origin='upper', cmap=cmap, 
                  norm=mcolors.Normalize(vmin=prod['vmin'], vmax=prod['vmax']), aspect='auto')
        
        plt.savefig(out_path, format='png', transparent=True, dpi=100)
        plt.close()
        os.remove(unzipped)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    manifest = {"reflectivity": [], "precip": []}
    
    for p_key in PRODUCTS:
        urls = get_latest_file_urls(p_key)
        for url in urls:
            fname = url.split('/')[-1]
            img_name = fname.replace('.grib2.gz', '.png')
            local_gz = os.path.join(DATA_DIR, fname)
            local_img = os.path.join(IMG_DIR, img_name)

            if not os.path.exists(local_img):
                print(f"Processing {fname}...")
                r = requests.get(url)
                with open(local_gz, 'wb') as f:
                    f.write(r.content)
                process_grib_to_png(local_gz, local_img, p_key)
                if os.path.exists(local_gz): os.remove(local_gz)

            if os.path.exists(local_img):
                manifest[p_key].append({
                    "url": f"static/images/{img_name}",
                    "time": fname.split('_')[-1].split('.')[0],
                    "bounds": [[20.0, -130.0], [55.0, -60.0]]
                })
    
    # Save the manifest so the HTML can read it without a Flask server
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f)

if __name__ == "__main__":
    main()
