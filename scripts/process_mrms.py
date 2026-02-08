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
# We use integers/exact decimals to prevent floating point drift
LAT_TOP, LAT_BOT = 50.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLOR DEFINITIONS (RGBA) ---
# Format: (R, G, B, A)
RAIN_COLORS = [
    (0, 251, 144, 255), (0, 187, 0, 255), (0, 136, 0, 255),
    (255, 255, 0, 255), (255, 145, 0, 255), (255, 0, 0, 255),
    (210, 0, 0, 255), (145, 0, 0, 255)
]

def get_color_for_value(val, vmax=15.0):
    if val < 0.1 or np.isnan(val): return (0, 0, 0, 0)
    idx = int((val / vmax) * (len(RAIN_COLORS) - 1))
    idx = min(max(idx, 0), len(RAIN_COLORS) - 1)
    return RAIN_COLORS[idx]

# --- S3 HELPERS ---
BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    r = requests.get(url, timeout=10)
    if r.status_code != 200: return []
    root = ET.fromstring(r.content)
    return sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    r = requests.get(url, stream=True)
    with open(filename + ".gz", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename + ".gz")

def process_frame(index, rate_key):
    try:
        download_and_extract(rate_key, f"rate_{index}.grib2")
        ds = xr.open_dataset(f"rate_{index}.grib2", engine="cfgrib")
        
        # 1. Normalize Longitude and Sort
        ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        # CRITICAL: Force North-to-South and West-to-East
        ds = ds.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        
        # 2. Slice strictly
        data_var = list(ds.data_vars)[0]
        subset = ds[data_var].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        
        # 3. Create Image Array (Height, Width, RGBA)
        raw_values = subset.values
        height, width = raw_values.shape
        rgba_data = np.zeros((height, width, 4), dtype=np.uint8)

        # Apply the Rain scale (vmax=15)
        # Vectorized coloring for speed
        vmax = 15.0
        mask = (raw_values >= 0.1) & (~np.isnan(raw_values))
        indices = ((raw_values[mask] / vmax) * (len(RAIN_COLORS) - 1)).astype(int)
        indices = np.clip(indices, 0, len(RAIN_COLORS) - 1)
        
        # Fill RGBA
        colors_np = np.array(RAIN_COLORS, dtype=np.uint8)
        rgba_data[mask] = colors_np[indices]

        # 4. Save via PIL (Ensures No Margins)
        img = Image.fromarray(rgba_data, 'RGBA')
        img_name = "master.png" if index == 0 else f"master_{index}.png"
        img.save(os.path.join(OUTPUT_DIR, img_name))

        # 5. Metadata (Use the EXACT coordinates from the data subset)
        # This eliminates the "Floating" issue by telling Leaflet exactly where the pixels end
        actual_bounds = [
            [float(subset.latitude.min()), float(subset.longitude.min())], # South West
            [float(subset.latitude.max()), float(subset.longitude.max())]  # North East
        ]
        
        utc_dt = datetime.fromtimestamp(ds.time.values.astype(int) * 1e-9, tz=timezone.utc)
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        meta = {
            "bounds": actual_bounds,
            "time": et_dt.strftime("%I:%M %p ET")
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Frame {index} generated with vmax=15 and strict bounds.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        f_path = f"rate_{index}.grib2"
        if os.path.exists(f_path): os.remove(f_path)

if __name__ == "__main__":
    # Note: SurfacePrecipRate_00.00 is the most common key
    PREFIX = "CONUS/SurfacePrecipRate_00.00"
    now_utc = datetime.now(timezone.utc)
    
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        keys = get_s3_keys(date_str, PREFIX)
        if len(keys) >= NUM_FRAMES:
            latest = sorted(keys)[-NUM_FRAMES:][::-1]
            for idx, k in enumerate(latest):
                process_frame(idx, k)
            break
