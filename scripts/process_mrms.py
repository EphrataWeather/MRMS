import os
import json
import requests
import gzip
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime, timezone, timedelta

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 23.0
LON_LEFT, LON_RIGHT = -125.0, -66.5
OUTPUT_DIR = "public/data"
NUM_FRAMES = 6 
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
# Corrected Prefixes for Surface-level products
RATE_PREFIX = "CONUS/SurfacePrecipRate_00.00"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

def get_s3_keys(date_str, prefix):
    """Fetches list of available GRIB files from S3 for a specific product and date."""
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    keys = []
    try:
        r = requests.get(request_url, timeout=15)
        if r.status_code != 200:
            return []
        
        # Parse XML with S3 namespace
        root = ET.fromstring(r.content)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        for contents in root.findall('s3:Contents', ns):
            key = contents.find('s3:Key', ns).text
            if key.endswith('.grib2.gz'):
                keys.append(key)
        return sorted(keys)
    except Exception as e:
        print(f"Error listing S3 keys: {e}")
        return []

def download_and_extract(key, filename):
    """Downloads a .gz from S3 and extracts it to a local GRIB2 file."""
    r = requests.get(f"{BUCKET_URL}/{key}", stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def get_colormap(p_type):
    """Returns color schemes for rain, snow, and ice/mix."""
    if p_type == 'snow':
        return ListedColormap(['#afc6ff', '#89a9ff', '#5a82ff', '#2d58ff', '#0026ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ffdaff', '#ffb3ff', '#ff80ff', '#e600e6', '#b300b3'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#ffff00', '#ff9100', '#ff0000', '#d20000'])

def process_frame(index, rate_key, flag_keys):
    # Match PrecipFlag to Rate using the timestamp in the filename (e.g., 20260205-200000)
    timestamp = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    
    if not flag_key:
        print(f"No matching flag file for timestamp {timestamp}. Skipping.")
        return

    print(f"Processing Frame {index}: {timestamp}")
    
    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # Adjust longitude to -180 to 180
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        
        # Categorize by PrecipFlag:
        # 1,2,5,7,8 = Rain | 3 = Snow | 4,6,10 = Ice/Mix
        rain_mask = rate.where((flag == 1) | (flag == 2) | (flag == 5) | (flag == 7) | (flag == 8))
        snow_mask = rate.where(flag == 3)
        ice_mask = rate.where((flag == 4) | (flag == 6) | (flag == 10))

        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]
        
        # Plot layers (only show data where rate > 0.1 mm/hr to reduce noise)
        ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=50, extent=extent, aspect='equal', interpolation='nearest')
        ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=10, extent=extent, aspect='equal', interpolation='nearest')
        ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=10, extent=extent, aspect='equal', interpolation='nearest')

        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), transparent=True, dpi=100, pad_inches=0)
        plt.close()

        if index == 0:
            ts = datetime.fromtimestamp(ds_rate.time.values.astype(int) * 1e-9, tz=timezone.utc)
            meta = {"bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": ts.strftime("%b %d, %H:%M UTC")}
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f: json.dump(meta, f)

        # Cleanup
        os.remove("rate.grib2")
        os.remove("flag.grib2")

    except Exception as e:
        print(f"Error processing frame {index}: {e}")

if __name__ == "__main__":
    now_utc = datetime.now(timezone.utc)
    found_keys = False

    # Check the last 3 days to ensure we find data regardless of UTC roll-over
    for d in range(3):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        print(f"Checking S3 for {date_str}...")
        
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if rate_keys and flag_keys:
            print(f"Success: Found {len(rate_keys)} rate files and {len(flag_keys)} flag files.")
            # Take the most recent frames
            latest_rates = rate_keys[-NUM_FRAMES:][::-1]
            for i, r_key in enumerate(latest_rates):
                process_frame(i, r_key, flag_keys)
            found_keys = True
            break
            
    if not found_keys:
        print("CRITICAL: No MRMS data found for the last 72 hours. Check bucket accessibility.")
