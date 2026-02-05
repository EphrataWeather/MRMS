import os
import json
import datetime
import requests
import gzip
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 6  # How many animation frames to create
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
REFLECT_PREFIX = "CONUS/MergedReflectivityQCComposite_00.50"
PTYPE_PREFIX = "CONUS/PrecipitationType"

def get_s3_keys(date_str, prefix):
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(request_url, timeout=15)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        return sorted([c.find('s3:Key', ns).text for c in root.findall('s3:Contents', ns) if c.find('s3:Key', ns).text.endswith('.grib2.gz')])
    except: return []

def download_and_extract(key, filename):
    r = requests.get(f"{BUCKET_URL}/{key}", stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def create_custom_cmap(p_type):
    """
    Creates a specific color scheme based on MRMS PrecipType:
    1: Warm Rain, 3: Snow, 4: Ice Pellets, 6: Freezing Rain, 7: Wet Snow, 10: Mixed
    """
    if p_type in [3, 7]: # Snow (Blues)
        colors = ['#afc6ff', '#89a9ff', '#5a82ff', '#2d58ff', '#0026ff']
    elif p_type in [4, 6, 10]: # Ice/Mixed (Pinks/Purples)
        colors = ['#ffdaff', '#ffb3ff', '#ff80ff', '#e600e6', '#b300b3']
    else: # Default/Rain (Greens/Yellows/Reds)
        colors = ['#00fb90', '#00bb00', '#ffff00', '#ff9100', '#ff0000', '#d20000']
    return ListedColormap(colors)

def process_frame(index, reflect_key):
    # 1. Derive the matching PrecipType key by replacing the prefix
    ptype_key = reflect_key.replace(REFLECT_PREFIX, PTYPE_PREFIX).replace("MergedReflectivityQCComposite_00.50", "PrecipitationType")
    
    print(f"Processing Frame {index}...")
    try:
        download_and_extract(reflect_key, "ref.grib2")
        download_and_extract(ptype_key, "ptype.grib2")
        
        # 2. Load Data
        ds_ref = xr.open_dataset("ref.grib2", engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepType': 'instant'}})
        ds_pt = xr.open_dataset("ptype.grib2", engine="cfgrib", backend_kwargs={'filter_by_keys': {'stepType': 'instant'}})
        
        # 3. Clean Coordinates
        for ds in [ds_ref, ds_pt]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # 4. Subset
        ref = ds_ref[list(ds_ref.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        pt = ds_pt[list(ds_pt.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # 5. Plotting with PrecipType Logic
        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        
        # We plot different types separately to apply unique colormaps
        # Rain
        rain_mask = ref.where((pt == 1) & (ref > 5))
        ax.imshow(rain_mask, extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP], cmap=create_custom_cmap(1), vmin=5, vmax=75, aspect='equal', interpolation='nearest')
        
        # Snow
        snow_mask = ref.where(((pt == 3) | (pt == 7)) & (ref > 5))
        ax.imshow(snow_mask, extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP], cmap=create_custom_cmap(3), vmin=5, vmax=75, aspect='equal', interpolation='nearest')
        
        # Ice/Mix
        ice_mask = ref.where(((pt == 4) | (pt == 6) | (pt == 10)) & (ref > 5))
        ax.imshow(ice_mask, extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP], cmap=create_custom_cmap(4), vmin=5, vmax=75, aspect='equal', interpolation='nearest')

        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), transparent=True, dpi=100, pad_inches=0)
        plt.close()

        # Save Metadata for the first frame
        if index == 0:
            ts = datetime.datetime.utcfromtimestamp(ds_ref.time.values.astype(int) * 1e-9)
            meta = {"bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": ts.strftime("%b %d, %H:%M UTC")}
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f: json.dump(meta, f)

    except Exception as e:
        print(f"Error frame {index}: {e}")

if __name__ == "__main__":
    now = datetime.datetime.utcnow()
    keys = get_s3_keys(now.strftime("%Y%m%d"), REFLECT_PREFIX)
    
    if not keys: # Try yesterday
        keys = get_s3_keys((now - datetime.timedelta(days=1)).strftime("%Y%m%d"), REFLECT_PREFIX)

    if keys:
        # Get the latest N keys, reversed so index 0 is the newest
        latest_keys = keys[-NUM_FRAMES:][::-1]
        for i, key in enumerate(latest_keys):
            process_frame(i, key)
