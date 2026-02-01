import os
import glob
import gzip
import shutil
import requests
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = "mrms_data"
IMG_DIR = "static/images"
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D"

for d in [DATA_DIR, IMG_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

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
    headers = {'User-Agent': 'Mozilla/5.0 MRMS-Viewer-App/1.0'}
    
    try:
        r = requests.get(idx_url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        
        pattern = re.compile(rf'href="({prod["prefix"]}.*?\.grib2\.gz)"')
        all_files = sorted(list(set(pattern.findall(r.text))))
        return [f"{idx_url}{f}" for f in all_files[-limit:]]
    except Exception as e:
        print(f"Scraper Error: {e}")
        return []

def process_grib_to_png(grib_path, out_path, product_key):
    try:
        unzipped = grib_path.replace(".gz", "")
        with gzip.open(grib_path, 'rb') as f_in, open(unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Requires cfgrib and eccodes installed
        ds = xr.open_dataset(unzipped, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        data = ds[var_name].values
        
        prod = PRODUCTS[product_key]
        data = np.where(data < prod['vmin'], np.nan, data)

        plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)

        cmap = plt.get_cmap(prod['cmap']).copy()
        cmap.set_bad(color='none') 
        
        ax.imshow(data, origin='upper', cmap=cmap, 
                  norm=mcolors.Normalize(vmin=prod['vmin'], vmax=prod['vmax']), 
                  aspect='auto')
        
        plt.savefig(out_path, format='png', transparent=True, dpi=100)
        plt.close()
        if os.path.exists(unzipped): os.remove(unzipped)
        return True
    except Exception as e:
        print(f"GRIB processing failed: {e}")
        return False

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_frames')
def get_frames():
    p_type = request.args.get('type', 'reflectivity')
    if p_type not in PRODUCTS:
        return jsonify({"error": "Invalid product type"}), 400

    urls = get_latest_file_urls(p_type, limit=10)
    frames = []

    for url in urls:
        fname = url.split('/')[-1]
        local_gz = os.path.join(DATA_DIR, fname)
        img_name = fname.replace('.grib2.gz', '.png')
        local_img = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(local_img):
            r = requests.get(url)
            with open(local_gz, 'wb') as f:
                f.write(r.content)
            process_grib_to_png(local_gz, local_img, p_type)
            if os.path.exists(local_gz): os.remove(local_gz)

        if os.path.exists(local_img):
            ts = fname.split('_')[-1].split('.')[0]
            frames.append({
                "url": f"/static/images/{img_name}",
                "bounds": [[20.0, -130.0], [55.0, -60.0]], # Standard CONUS MRMS bounds
                "time": ts
            })

    return jsonify(frames)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
