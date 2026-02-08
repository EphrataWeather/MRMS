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
LAT_TOP, LAT_BOT = 50.0, 22.0
LON_LEFT, LON_RIGHT = -128.0, -65.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 15
os.makedirs(OUTPUT_DIR, exist_ok=True)

# RGBA scale for Vmax = 15
RAIN_COLORS = [
    (0, 251, 144, 255), (0, 187, 0, 255), (0, 136, 0, 255),
    (255, 255, 0, 255), (255, 145, 0, 255), (255, 0, 0, 255),
    (210, 0, 0, 255), (145, 0, 0, 255)
]

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"

def discover_prefix():
    """Finds the correct SurfacePrecipRate folder on S3."""
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for e in root.iter():
            if e.tag.endswith('Prefix') and ("SurfacePrecipRate" in e.text or "PrecipRate" in e.text):
                return e.text.rstrip("/")
    except: pass
    return "CONUS/SurfacePrecipRate_00.00"

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

def process_frame(index, key):
    try:
        temp_grib = f"temp_{index}.grib2"
        download_and_extract(key, temp_grib)
        ds = xr.open_dataset(temp_grib, engine="cfgrib")
        
        # 1. Coordinate Normalization
        ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        
        # Force Latitude to be Descending (North at index 0)
        # Force Longitude to be Ascending (West at index 0)
        ds = ds.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        
        # 2. Slice (Always [Max, Min] for descending coordinates)
        subset = ds.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        
        # Determine the data variable (usually 'unknown' or 'paramId_0')
        var_name = [v for v in subset.data_vars if 'latitude' in subset[v].coords][0]
        data = subset[var_name].values
        
        if data.size == 0:
            print(f"Frame {index}: Slice resulted in empty data.")
            return

        # 3. Build RGBA Image with PIL
        height, width = data.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Apply Vmax = 15 scaling
        vmax = 15.0
        mask = (data >= 0.1) & (~np.isnan(data))
        if np.any(mask):
            indices = ((data[mask] / vmax) * (len(RAIN_COLORS) - 1)).astype(int)
            indices = np.clip(indices, 0, len(RAIN_COLORS) - 1)
            colors_np = np.array(RAIN_COLORS, dtype=np.uint8)
            rgba[mask] = colors_np[indices]

        # 4. Save Image
        img = Image.fromarray(rgba, 'RGBA')
        name = "master.png" if index == 0 else f"master_{index}.png"
        img.save(os.path.join(OUTPUT_DIR, name))

        # 5. Metadata (Pull actual edge coordinates to prevent drift)
        meta = {
            "bounds": [
                [float(subset.latitude.min()), float(subset.longitude.min())],
                [float(subset.latitude.max()), float(subset.longitude.max())]
            ],
            "time": datetime.now(pytz.timezone('US/Eastern')).strftime("%I:%M %p ET")
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)

        print(f"Successfully created {name}")

    except Exception as e:
        print(f"Failed frame {index}: {e}")
    finally:
        if os.path.exists(temp_grib): os.remove(temp_grib)

if __name__ == "__main__":
    PREFIX = discover_prefix()
    print(f"Using Prefix: {PREFIX}")
    
    now = datetime.now(timezone.utc)
    for i in range(2):
        date_str = (now - timedelta(days=i)).strftime("%Y%m%d")
        keys = get_s3_keys(date_str, PREFIX)
        if keys:
            latest_keys = sorted(keys)[-NUM_FRAMES:][::-1]
            for idx, k in enumerate(latest_keys):
                process_frame(idx, k)
            break
