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
# Note: In 2026, some products moved to include version subfolders
RATE_PREFIX = "CONUS/SurfacePrecipRate"
FLAG_PREFIX = "CONUS/PrecipFlag"

def get_s3_keys(date_str, prefix):
    """Fetches and parses S3 XML response using a robust namespace-free method."""
    # We try both the base prefix and the common '00.00' subfolder
    search_prefixes = [f"{prefix}/{date_str}/", f"{prefix}/00.00/{date_str}/"]
    
    all_keys = []
    for p in search_prefixes:
        request_url = f"{BUCKET_URL}/?list-type=2&prefix={p}"
        try:
            r = requests.get(request_url, timeout=15)
            if r.status_code != 200:
                continue
            
            # Robust XML parsing (ignores namespaces)
            root = ET.fromstring(r.content)
            for content in root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
                key = content.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key').text
                if key.endswith('.grib2.gz'):
                    all_keys.append(key)
        except Exception as e:
            print(f"Error searching {p}: {e}")
            
    return sorted(all_keys)

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Failed to download {url}")
        
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#afc6ff', '#89a9ff', '#5a82ff', '#2d58ff', '#0026ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ffdaff', '#ffb3ff', '#ff80ff', '#e600e6', '#b300b3'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#ffff00', '#ff9100', '#ff0000', '#d20000'])

def process_frame(index, rate_key):
    # Derive the matching PrecipFlag key
    # This logic matches the timestamp string at the end of the filename
    timestamp_part = rate_key.split('_')[-1] # e.g., 20260205-010000.grib2.gz
    
    # We search the flag directory for a matching timestamp
    date_folder = rate_key.split('/')[-2]
    flag_keys = get_s3_keys(date_folder, FLAG_PREFIX)
    flag_key = next((k for k in flag_keys if timestamp_part in k), None)
    
    if not flag_key:
        print(f"Skipping frame {index}: No matching PrecipFlag found for {rate_key}")
        return

    print(f"Processing Frame {index}: Rate[{rate_key.split('/')[-1]}] Flag[{flag_key.split('/')[-1]}]")
    
    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # Normalize Longitude (0-360 to -180-180)
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        
        # PrecipFlag Values: 1-2, 5, 7-8 (Rain) | 3 (Snow) | 4,6,10 (Ice/Mix)
        rain_mask = rate.where((flag == 1) | (flag == 2) | (flag == 5) | (flag == 7) | (flag == 8))
        snow_mask = rate.where(flag == 3)
        ice_mask = rate.where((flag == 4) | (flag == 6) | (flag == 10))

        params = {'extent': [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP], 'aspect': 'equal', 'interpolation': 'nearest'}
        
        # We plot rain first, then snow/ice on top
        ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=50, **params)
        ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=10, **params)
        ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=10, **params)

        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), transparent=True, dpi=100, pad_inches=0)
        plt.close()

        if index == 0:
            ts = datetime.datetime.utcfromtimestamp(ds_rate.time.values.astype(int) * 1e-9)
            meta = {"bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": ts.strftime("%b %d, %H:%M UTC")}
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f: json.dump(meta, f)

    except Exception as e:
        print(f"Processing failed for frame {index}: {e}")

if __name__ == "__main__":
    now = datetime.datetime.utcnow()
    keys = []
    
    # Check last 3 days
    for i in range(3):
        d_str = (now - datetime.timedelta(days=i)).strftime("%Y%m%d")
        print(f"Searching AWS for {d_str}...")
        keys = get_s3_keys(d_str, RATE_PREFIX)
        if keys:
            print(f"Found {len(keys)} files for {d_str}")
            break

    if keys:
        latest_keys = keys[-NUM_FRAMES:][::-1]
        for i, key in enumerate(latest_keys):
            process_frame(i, key)
    else:
        print("CRITICAL: No MRMS Rate data found on S3. Check if BUCKET_URL or prefixes have changed.")
