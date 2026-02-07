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
import pytz

# --- CONFIGURATION ---
# Using precise 0.01 degree intervals ensures the MRMS grid aligns perfectly.
LAT_TOP, LAT_BOT = 50.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

def discover_rate_prefix():
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                if "PrecipRate" in p or "SurfacePrecip" in p:
                    return p.rstrip("/")
    except: pass
    return "CONUS/SurfacePrecipRate_00.00"

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        return sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])
    except: return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    r = requests.get(url, stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: # Rain (Standard NWS-ish Scale)
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

def process_frame(index, rate_key, flag_keys):
    timestamp_str = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp_str in k), None)
    
    if not flag_key: return

    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        # Open with explicit settings
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # 1. Normalize Longitude and SORT coordinates
        # This ensures the array indexing [0,0] is definitely the top-left (North-West)
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # 2. Slice to exact bounds
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # 3. Create Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # --- PIXEL-PERFECT PLOTTING ---
        # Calculate aspect ratio to avoid internal Matplotlib stretching
        height_px, width_px = rain.shape
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # The extent must match the sliced coordinates exactly
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]

        # Use origin='upper' because we sorted latitude as Descending (Top = Max Lat)
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, extent=extent, origin='upper', interpolation='nearest')
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, origin='upper', interpolation='nearest')
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, origin='upper', interpolation='nearest')

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        
        # CRITICAL: Do NOT use bbox_inches='tight'. It re-crops the image and breaks the alignment.
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # 4. Save Metadata
        utc_dt = datetime.fromtimestamp(ds_rate.time.values.astype(int) * 1e-9, tz=timezone.utc)
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        meta = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time": et_dt.strftime("%I:%M %p ET"),
            "vmax_applied": 15
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Processed {img_name} - Time: {meta['time']}")

    except Exception as e:
        print(f"Error on frame {index}: {e}")
    finally:
        for f in ["rate.grib2", "flag.grib2"]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    # Try today's folder, then yesterday's if it's early morning
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if len(rate_keys) >= NUM_FRAMES:
            latest = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, r_key in enumerate(latest):
                process_frame(idx, r_key, flag_keys)
            break
