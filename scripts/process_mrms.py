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

# --- COLORMAPS ---
def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

def discover_rate_prefix():
    print("Discovering Rate Prefix...")
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                if "PrecipRate" in p or "SurfacePrecip" in p:
                    print(f"Found Prefix: {p.rstrip('/')}")
                    return p.rstrip("/")
    except Exception as e:
        print(f"Prefix discovery failed: {e}")
    return "CONUS/SurfacePrecipRate_00.00"

def get_s3_keys(date_str, prefix):
    print(f"Checking S3 for {date_str} in {prefix}...")
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: 
            print(f"HTTP Error {r.status_code} for {url}")
            return []
        root = ET.fromstring(r.content)
        keys = sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])
        print(f"Found {len(keys)} keys.")
        return keys
    except Exception as e: 
        print(f"S3 Listing Error: {e}")
        return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    print(f"Downloading {key}...")
    r = requests.get(url, stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def process_frame(index, rate_key, flag_keys):
    # FUZZY MATCHING: Match YYYYMMDD-HHMM (first 13 chars of timestamp)
    # Rate Key Ex: ..._20240520-120005.grib2.gz
    timestamp_part = rate_key.split('_')[-1] # 20240520-120005.grib2.gz
    time_prefix = timestamp_part[:13]        # 20240520-1200
    
    flag_key = next((k for k in flag_keys if time_prefix in k), None)
    
    if not flag_key: 
        print(f"Skipping frame {index}: No matching flag file for {time_prefix}")
        return

    tmp_r = f"rate_{index}.grib2"
    tmp_f = f"flag_{index}.grib2"

    try:
        download_and_extract(rate_key, tmp_r)
        download_and_extract(flag_key, tmp_f)
        
        # Open datasets
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib")
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib")
        
        # 1. Normalize Longitude (-180 to 180)
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # 2. Sort Lat (Descending) & Lon (Ascending)
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        
        # 3. Slice Data
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # --- EXTENT CALCULATION (Fixes North Shift) ---
        res = 0.01
        lats = rate.latitude.values
        lons = rate.longitude.values
        
        # Edge = Center +/- Half Resolution
        north = lats[0] + (res / 2)
        south = lats[-1] - (res / 2)
        west  = lons[0] - (res / 2)
        east  = lons[-1] + (res / 2)
        
        extent = [west, east, south, north]

        # 4. Create Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # --- PLOTTING (Fixes Resolution) ---
        height_px, width_px = rain.shape
        # High DPI (300) + No Interpolation = Crisp Pixels
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=800)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        plot_args = dict(extent=extent, origin='upper', interpolation='none')
        
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **plot_args)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **plot_args)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **plot_args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # 5. Metadata
        # Robust time extraction
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
             # Fallback to parsing filename
             utc_dt = datetime.strptime(timestamp_part.split('.')[0], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        meta = {
            "bounds": [[south, west], [north, east]], 
            "time": et_dt.strftime("%I:%M %p ET")
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"SUCCESS: {img_name} | {meta['time']}")

    except Exception as e:
        print(f"CRITICAL ERROR on frame {index}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    processed_count = 0
    
    # Loop over today and yesterday to find files
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        print(f"--- Processing Date: {date_str} ---")
        
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if not rate_keys:
            print("No Rate keys found.")
            continue
            
        # Sort and take the latest available
        # Reverse so index 0 is the NEWEST
        latest_rate = sorted(rate_keys)[::-1]
        
        # Limit to however many frames we want, but don't crash if fewer exist
        frames_to_process = latest_rate[:NUM_FRAMES]
        
        print(f"Processing {len(frames_to_process)} frames...")
        
        for idx, r_key in enumerate(frames_to_process):
            process_frame(idx, r_key, flag_keys)
            processed_count += 1
            
        if processed_count > 0:
            print("Batch complete.")
            break
            
    if processed_count == 0:
        print("NO FRAMES PROCESSED. Check S3 connectivity or Prefix.")
