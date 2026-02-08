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
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

def process_frame(index, rate_key, flag_keys):
    timestamp_str = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp_str in k), None)
    
    if not flag_key: return

    try:
        tmp_r, tmp_f = f"rate_{index}.grib2", f"flag_{index}.grib2"
        download_and_extract(rate_key, tmp_r)
        download_and_extract(flag_key, tmp_f)
        
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib")
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib")
        
        # 1. Coordinate Normalization
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # 2. Precise Slicing
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # --- FIX: CALCUALTE PIXEL EDGES TO PREVENT NORTH/WEST DRIFT ---
        # MRMS resolution is 0.01. The 'actual' boundary is half a pixel beyond the center coordinate.
        res = 0.01
        actual_top = float(rate.latitude.max()) + (res / 2)
        actual_bot = float(rate.latitude.min()) - (res / 2)
        actual_left = float(rate.longitude.min()) - (res / 2)
        actual_right = float(rate.longitude.max()) - (res / 2)

        # 3. Create Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # --- PIXEL-PERFECT PLOTTING ---
        height_px, width_px = rain.shape
        # Force exact pixel dimensions
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # Extent must match the 'Actual' calculated edges
        extent = [actual_left, actual_right, actual_bot, actual_top]

        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, extent=extent, origin='upper', interpolation='nearest')
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, origin='upper', interpolation='nearest')
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, origin='upper', interpolation='nearest')

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        
        # Save without any clipping or tight-box adjustments
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # --- FIX: ROBUST TIMESTAMP EXTRACTION ---
        # Try 'valid_time' first (standard for cfgrib), fallback to 'time'
        raw_time = ds_rate.get('valid_time', ds_rate.get('time')).values
        if isinstance(raw_time, np.ndarray): raw_time = raw_time[0]
        
        # Convert numpy datetime64 to python datetime
        utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        # --- FIX: SYNC BOUNDS WITH IMAGE EDGES ---
        meta = {
            "bounds": [[actual_bot, actual_left], [actual_top, actual_right]],
            "time": et_dt.strftime("%I:%M %p ET"),
            "vmax_applied": 15
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Processed {img_name} - Time: {meta['time']} - Bounds Fixed.")

    except Exception as e:
        print(f"Error on frame {index}: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if len(rate_keys) >= NUM_FRAMES:
            # Get latest frames in chronological order for the loop, 
            # but index 0 remains the absolute latest.
            latest = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, r_key in enumerate(latest):
                process_frame(idx, r_key, flag_keys)
            break
