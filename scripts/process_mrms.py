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
        
        # Open with filtering to avoid multi-message coordinate confusion
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # 1. Coordinate Normalization
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        
        # Force strict sort: Latitude North -> South
        ds_rate = ds_rate.sortby("latitude", ascending=False)
        ds_flag = ds_flag.sortby("latitude", ascending=False)

        # 2. Slice with boundary logic
        # We use slice(LAT_TOP, LAT_BOT) because latitude is DESCENDING
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(
            latitude=slice(LAT_TOP, LAT_BOT), 
            longitude=slice(LON_LEFT, LON_RIGHT)
        )
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(
            latitude=slice(LAT_TOP, LAT_BOT), 
            longitude=slice(LON_LEFT, LON_RIGHT)
        )

        # --- THE ULTIMATE ALIGNMENT FIX ---
        # Get the actual coordinates from the sliced data
        lats = rate.latitude.values
        lons = rate.longitude.values
        res = 0.01
        
        # Calculate the EXTENT edges (centers +/- half resolution)
        # This is what Matplotlib and Leaflet use to anchor the image.
        actual_top = lats[0] + (res / 2)
        actual_bot = lats[-1] - (res / 2)
        actual_left = lons[0] - (res / 2)
        actual_right = lons[-1] + (res / 2)

        # 3. Create Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # --- PIXEL-PERFECT PLOTTING ---
        height_px, width_px = rain.shape
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # Define the box exactly
        extent = [actual_left, actual_right, actual_bot, actual_top]

        # Use origin='upper' because our array starts at the North (index 0)
        common_params = dict(extent=extent, origin='upper', interpolation='nearest')

        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **common_params)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **common_params)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **common_params)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # --- FIX: ROBUST TIMESTAMP ---
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
            # Fallback for older cfgrib versions
            utc_dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        meta = {
            "bounds": [[actual_bot, actual_left], [actual_top, actual_right]],
            "time": et_dt.strftime("%I:%M %p ET"),
            "vmax_applied": 15
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Processed {img_name} | {meta['time']} | Bounds: {actual_top:.3f}N to {actual_bot:.3f}N")

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
            latest = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, r_key in enumerate(latest):
                process_frame(idx, r_key, flag_keys)
            break
