import os
import json
import requests
import gzip
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
from PIL import Image
from datetime import datetime, timezone, timedelta
import pytz

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 23.0
LON_LEFT, LON_RIGHT = -125.0, -66.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S3 Prefixes
RATE_PREFIX = "CONUS/SurfacePrecipRate_00.00"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# Color Palettes (RGBA)
RAIN_COLORS = [
    (0, 251, 144, 255), (0, 187, 0, 255), (0, 136, 0, 255),
    (255, 255, 0, 255), (255, 145, 0, 255), (255, 0, 0, 255),
    (210, 0, 0, 255), (145, 0, 0, 255)
]
SNOW_COLORS = [(0, 255, 255, 255), (150, 200, 255, 255), (255, 255, 255, 255)]
ICE_COLORS = [(255, 0, 255, 255), (200, 0, 200, 255)]

def get_s3_keys(date_str, prefix):
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        return sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])
    except Exception as e:
        print(f"S3 Connection Error: {e}")
        return []

def download_and_extract(key, filename):
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/{key}"
    r = requests.get(url, stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def process_frame(index, rate_key, flag_keys):
    # Loose match: find a flag file within the same minute as the rate file
    # Example: CONUS_SurfacePrecipRate_00.00_20240520-120005.grib2.gz -> 20240520-1200
    time_part = rate_key.split("_")[-1][:13] 
    flag_key = next((k for k in flag_keys if time_part in k), None)
    
    if not flag_key:
        print(f"Skipping frame {index}: No matching flag for {time_part}")
        return

    rate_file, flag_file = f"r_{index}.grib", f"f_{index}.grib"
    try:
        download_and_extract(rate_key, rate_file)
        download_and_extract(flag_key, flag_file)
        
        # Load with CFGRIB
        ds_r = xr.open_dataset(rate_file, engine="cfgrib")
        ds_f = xr.open_dataset(flag_file, engine="cfgrib")

        # 1. Coordinate Alignment & Sorting
        # Longitude to -180 to 180
        ds_r.coords['longitude'] = (ds_r.longitude + 180) % 360 - 180
        ds_f.coords['longitude'] = (ds_f.longitude + 180) % 360 - 180
        
        # MANDATORY: Sort Latitude Descending (North to South)
        # This fixes the 'North' drift by ensuring row 0 is the top edge.
        ds_r = ds_r.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_f = ds_f.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # 2. Precise Slicing
        sub_r = ds_r.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        sub_f = ds_f.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # Detect variable names (often 'unknown' or 'prate')
        v_r = list(sub_r.data_vars)[0]
        v_f = list(sub_f.data_vars)[0]
        
        rate_data = sub_r[v_r].values
        flag_data = sub_f[v_f].values

        # 3. Create RGBA Array
        h, w = rate_data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Masks based on MRMS PrecipFlag specs
        # 1,2,5,7,8,10 = Rain | 3 = Snow | 4,6 = Ice
        rain_mask = np.isin(flag_data, [1, 2, 5, 7, 8, 10]) & (rate_data >= 0.1)
        snow_mask = (flag_data == 3) & (rate_data >= 0.1)
        ice_mask = np.isin(flag_data, [4, 6]) & (rate_data >= 0.1)

        def color_pixels(mask, data, colors, vmax_val):
            if not np.any(mask): return
            normalized = np.clip(data[mask] / vmax_val, 0, 1)
            indices = (normalized * (len(colors) - 1)).astype(int)
            palette = np.array(colors, dtype=np.uint8)
            rgba[mask] = palette[indices]

        color_pixels(rain_mask, rate_data, RAIN_COLORS, 15.0) # Vmax=15
        color_pixels(snow_mask, rate_data, SNOW_COLORS, 5.0)
        color_pixels(ice_mask, rate_data, ICE_COLORS, 5.0)

        # 4. Save as raw PNG (PIL ensures zero padding)
        img = Image.fromarray(rgba, 'RGBA')
        out_name = "master.png" if index == 0 else f"master_{index}.png"
        img.save(os.path.join(OUTPUT_DIR, out_name))

        # 5. Export Bounds for Leaflet
        meta = {
            "bounds": [
                [float(sub_r.latitude.min()), float(sub_r.longitude.min())],
                [float(sub_r.latitude.max()), float(sub_r.longitude.max())]
            ],
            "time": datetime.now(pytz.timezone('US/Eastern')).strftime("%I:%M %p ET")
        }
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)

        print(f"Successfully created: {out_name}")

    except Exception as e:
        print(f"Error processing frame {index}: {e}")
    finally:
        for f in [rate_file, flag_file]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    found = False
    for d in range(2):
        d_str = (now - timedelta(days=d)).strftime("%Y%m%d")
        print(f"Checking S3 for date: {d_str}...")
        r_keys = get_s3_keys(d_str, RATE_PREFIX)
        f_keys = get_s3_keys(d_str, FLAG_PREFIX)
        
        if r_keys and f_keys:
            print(f"Found {len(r_keys)} rate files. Processing...")
            latest_r = sorted(r_keys)[-NUM_FRAMES:][::-1]
            for idx, k in enumerate(latest_r):
                process_frame(idx, k, f_keys)
            found = True
            break
    if not found:
        print("Could not find any MRMS data for today or yesterday.")
