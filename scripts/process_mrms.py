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
LAT_TOP, LAT_BOT = 50.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 15
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- S3 PREFIXES ---
RATE_PREFIX = "CONUS/SurfacePrecipRate_00.00"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# --- COLOR PALETTES (RGBA) ---
# Rain: Green -> Yellow -> Red (vmax=15)
RAIN_COLORS = [
    (0, 251, 144, 255), (0, 187, 0, 255), (0, 136, 0, 255),
    (255, 255, 0, 255), (255, 145, 0, 255), (255, 0, 0, 255),
    (210, 0, 0, 255), (145, 0, 0, 255)
]
# Snow: Cyan -> Light Blue -> White
SNOW_COLORS = [(0, 255, 255, 255), (128, 255, 255, 255), (255, 255, 255, 255)]
# Ice: Pink -> Purple
ICE_COLORS = [(255, 0, 255, 255), (128, 0, 128, 255)]

def get_s3_keys(date_str, prefix):
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        return sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])
    except: return []

def download_and_extract(key, filename):
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/{key}"
    r = requests.get(url, stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def process_frame(index, rate_key, flag_keys):
    # Match the flag file to the rate file by timestamp
    timestamp = rate_key.split("_")[-1].split(".")[0]
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    if not flag_key: return

    try:
        rate_file, flag_file = f"rate_{index}.grib", f"flag_{index}.grib"
        download_and_extract(rate_key, rate_file)
        download_and_extract(flag_key, flag_file)
        
        ds_r = xr.open_dataset(rate_file, engine="cfgrib")
        ds_f = xr.open_dataset(flag_file, engine="cfgrib")

        # 1. NORMALIZE & SORT (Crucial for alignment)
        for ds in [ds_r, ds_f]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        
        # Force North-to-South (Latitude Descending) and West-to-East (Longitude Ascending)
        ds_r = ds_r.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_f = ds_f.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # 2. SLICE (Top, Bot / Left, Right)
        sub_r = ds_r.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        sub_f = ds_f.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        rate_data = sub_r[list(sub_r.data_vars)[0]].values
        flag_data = sub_f[list(sub_f.data_vars)[0]].values

        # 3. MAPPING PIXELS
        height, width = rate_data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)

        vmax = 15.0
        # Flags: 1,2,5,7,8,10=Rain | 3=Snow | 4,6=Ice
        rain_mask = np.isin(flag_data, [1, 2, 5, 7, 8, 10]) & (rate_data >= 0.1)
        snow_mask = (flag_data == 3) & (rate_data >= 0.1)
        ice_mask = np.isin(flag_data, [4, 6]) & (rate_data >= 0.1)

        def apply_colors(mask, data, palette, max_val):
            if not np.any(mask): return
            indices = ((data[mask] / max_val) * (len(palette) - 1)).astype(int)
            indices = np.clip(indices, 0, len(palette) - 1)
            pal_np = np.array(palette, dtype=np.uint8)
            rgba[mask] = pal_np[indices]

        apply_colors(rain_mask, rate_data, RAIN_COLORS, vmax)
        apply_colors(snow_mask, rate_data, SNOW_COLORS, 5.0) # Snow uses lower scale for visibility
        apply_colors(ice_mask, rate_data, ICE_COLORS, 5.0)

        # 4. SAVE IMAGE (No padding, no margins)
        img = Image.fromarray(rgba, 'RGBA')
        img_name = "master.png" if index == 0 else f"master_{index}.png"
        img.save(os.path.join(OUTPUT_DIR, img_name))

        # 5. METADATA (Precise coordinates for Leaflet)
        meta = {
            "bounds": [
                [float(sub_r.latitude.min()), float(sub_r.longitude.min())],
                [float(sub_r.latitude.max()), float(sub_r.longitude.max())]
            ],
            "time": datetime.now(pytz.timezone('US/Eastern')).strftime("%I:%M %p ET")
        }
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Frame {index} generated: Rain/Snow/Ice separated.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        for f in [f"rate_{index}.grib", f"flag_{index}.grib"]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    for d in range(2):
        date_str = (now - timedelta(days=d)).strftime("%Y%m%d")
        r_keys = get_s3_keys(date_str, RATE_PREFIX)
        f_keys = get_s3_keys(date_str, FLAG_PREFIX)
        if r_keys and f_keys:
            latest_r = sorted(r_keys)[-NUM_FRAMES:][::-1]
            for idx, k in enumerate(latest_r):
                process_frame(idx, k, f_keys)
            break
