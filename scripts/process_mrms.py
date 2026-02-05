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
from matplotlib.colors import ListedColormap

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 23.0
LON_LEFT, LON_RIGHT = -125.0, -66.5
OUTPUT_DIR = "public/data"
NUM_FRAMES = 6 
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
RATE_PREFIX = "CONUS/SurfacePrecipRate"
FLAG_PREFIX = "CONUS/PrecipFlag"

def get_s3_keys(date_str, prefix):
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(request_url, timeout=15)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        return sorted([c.find('s3:Key', ns).text for c in root.findall('s3:Contents', ns) 
                       if c.find('s3:Key', ns).text.endswith('.grib2.gz')])
    except: return []

def download_and_extract(key, filename):
    r = requests.get(f"{BUCKET_URL}/{key}", stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def get_colormap(p_type):
    """
    Custom colormaps for different precip categories.
    PrecipFlag Values: 1,2,5,7,8 (Rain) | 3,7 (Snow) | 4,6,10 (Ice/Mix)
    """
    if p_type == 'snow':
        return ListedColormap(['#afc6ff', '#89a9ff', '#5a82ff', '#2d58ff', '#0026ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ffdaff', '#ffb3ff', '#ff80ff', '#e600e6', '#b300b3'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#ffff00', '#ff9100', '#ff0000', '#d20000'])

def process_frame(index, rate_key):
    # Match the PrecipFlag file to the SurfacePrecipRate timestamp
    flag_key = rate_key.replace(RATE_PREFIX, FLAG_PREFIX).replace("SurfacePrecipRate", "PrecipFlag")
    
    print(f"Processing Frame {index}: {rate_key.split('/')[-1]}")
    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        # Open datasets
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # Normalize Longitude
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        
        # Define precip masks based on NOAA Flag values
        # Rain: 1 (Warm), 2 (Strat), 5 (Conv), 7 (Trop-Conv), 8 (Trop-Strat)
        rain_mask = rate.where((flag == 1) | (flag == 2) | (flag == 5) | (flag == 7) | (flag == 8))
        # Snow: 3 (Snow)
        snow_mask = rate.where(flag == 3)
        # Ice/Mixed: 4 (Ice Pellets), 6 (Freezing Rain), 10 (Mixed)
        ice_mask = rate.where((flag == 4) | (flag == 6) | (flag == 10))

        # Plot each layer (Vmax=50mm/hr for rain, lower for snow/ice usually looks better)
        common_params = {'extent': [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP], 'aspect': 'equal', 'interpolation': 'nearest'}
        
        ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=50, **common_params)
        ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=10, **common_params)
        ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=10, **common_params)

        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), transparent=True, dpi=100, pad_inches=0)
        plt.close()

        if index == 0:
            ts = datetime.datetime.utcfromtimestamp(ds_rate.time.values.astype(int) * 1e-9)
            meta = {"bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": ts.strftime("%b %d, %H:%M UTC")}
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f: json.dump(meta, f)

    except Exception as e:
        print(f"Error frame {index}: {e}")

if __name__ == "__main__":
    now = datetime.datetime.utcnow()
    # Check today, then yesterday if today's folder is empty
    for d in [0, 1]:
        date_str = (now - datetime.timedelta(days=d)).strftime("%Y%m%d")
        keys = get_s3_keys(date_str, RATE_PREFIX)
        if keys: break

    if keys:
        # Get the latest N files for animation
        latest_keys = keys[-NUM_FRAMES:][::-1]
        for i, key in enumerate(latest_keys):
            process_frame(i, key)
    else:
        print("No data found in the last 48 hours.")
