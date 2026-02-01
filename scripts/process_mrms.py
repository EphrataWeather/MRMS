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

# --- Configuration ---
DATA_DIR = "mrms_data"
IMG_DIR = "static/images"
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

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
    """Robustly scrapes the NOAA directory for the latest files."""
    prod = PRODUCTS[product_key]
    idx_url = f"{BASE_URL}/{prod['url']}/"
    
    headers = {'User-Agent': 'Mozilla/5.0 MRMS-Viewer-App/1.0'}
    
    try:
        r = requests.get(idx_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        
        # Matches the filename pattern within the HTML index
        pattern = re.compile(rf'href="({prod["prefix"]}.*?\.grib2\.gz)"')
        all_files = list(set(pattern.findall(r.text)))
        
        if not all_files:
            return []

        # Sort by timestamp (contained in filename) and take the newest
        all_files.sort()
        return [f"{idx_url}{f}" for f in all_files[-limit:]]
    except Exception as e:
        print(f"Scraping error: {e}")
        return []

def process_grib_to_png(grib_path, out_path, product_key):
    """Decodes GRIB2 data and exports a transparent map layer."""
    try:
        unzipped = grib_path.replace(".gz", "")
        with gzip.open(grib_path, 'rb') as f_in, open(unzipped, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Open dataset (requires cfgrib and eccodes)
        ds = xr.open_dataset(unzipped, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        data = ds[var_name].values
        
        # Clean up binary data: convert 'no coverage' values to NaN
        prod = PRODUCTS[product_key]
        data = np.where(data < prod['vmin'], np.nan, data)

        plt.figure(figsize=(12, 12), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)

        cmap = plt.get_cmap(prod['cmap']).copy()
        cmap.set_bad(color='none') 
        
        ax.imshow(data, origin='upper', cmap=cmap, 
                  norm=mcolors.Normalize(vmin=prod['vmin'], vmax=prod['vmax']), 
                  aspect='auto')
        
        plt.savefig(out_path, format='png', transparent=True, dpi=150)
        plt.close()
        os.remove(unzipped) # Clean up unzipped GRIB
        return True
    except Exception as e:
        print(f"Error processing {grib_path}: {e}")
        return False

def main():
    """Main execution loop for GitHub Actions or Local Cron."""
    for p_key in PRODUCTS:
        print(f"Fetching latest files for: {p_key}")
        urls = get_latest_file_urls(p_key, limit=10)
        
        if not urls:
            print(f"Skipping {p_key}: No files found on server.")
            continue

        for url in urls:
            filename = url.split('/')[-1]
            local_gz = os.path.join(DATA_DIR, filename)
            img_name = filename.replace('.grib2.gz', '.png')
            local_img = os.path.join(IMG_DIR, img_name)

            if not os.path.exists(local_img):
                print(f"Downloading & Processing {filename}...")
                r = requests.get(url)
                with open(local_gz, 'wb') as f:
                    f.write(r.content)
                
                success = process_grib_to_png(local_gz, local_img, p_key)
                if success:
                    # Optional: remove raw gz after processing to save space
                    os.remove(local_gz)

if __name__ == "__main__":
    main()
