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

# --- CONFIGURATION ---
LAT_TOP = 50.0
LAT_BOT = 20.0
LON_LEFT = -130.0
LON_RIGHT = -60.0

OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AWS S3 Bucket Information
BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
PREFIX_BASE = "CONUS/MergedReflectivityQCComposite_00.50"

def get_s3_file_list(date_str):
    """
    Fetches the list of files from the S3 bucket for a specific date.
    Returns a sorted list of file Keys (paths).
    """
    # AWS S3 List Objects URL
    # We use prefix to look inside the specific folder for that date
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={PREFIX_BASE}/{date_str}/"
    
    try:
        r = requests.get(request_url, timeout=15)
        if r.status_code != 200:
            return []
        
        # Parse XML response from S3
        root = ET.fromstring(r.content)
        # XML namespace for S3
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        keys = []
        for content in root.findall('s3:Contents', ns):
            key = content.find('s3:Key', ns).text
            if key.endswith('.grib2.gz'):
                keys.append(key)
        
        return sorted(keys)
    except Exception as e:
        print(f"S3 Listing Error: {e}")
        return []

def get_latest_mrms_url():
    """Finds the latest file on AWS, checking Today then Yesterday."""
    now_utc = datetime.datetime.utcnow()
    
    # 1. Try TODAY
    date_str = now_utc.strftime("%Y%m%d")
    print(f"Checking AWS S3 for date: {date_str}...")
    keys = get_s3_file_list(date_str)
    
    if keys:
        return f"{BUCKET_URL}/{keys[-1]}"

    # 2. If today is empty, try YESTERDAY
    print("Today is empty. Checking yesterday...")
    yesterday = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")
    keys = get_s3_file_list(yesterday)
    
    if keys:
        return f"{BUCKET_URL}/{keys[-1]}"

    return None

def process_data():
    # 1. Get URL from AWS
    url = get_latest_mrms_url()
    if not url:
        print("CRITICAL: No data found on AWS.")
        return

    print(f"Downloading: {url}")
    
    # 2. Download & Decompress (.gz -> .grib2)
    try:
        r = requests.get(url, stream=True)
        # Save compressed file
        with open("temp.grib2.gz", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Decompress it
        print("Decompressing GRIB2 file...")
        with gzip.open("temp.grib2.gz", "rb") as f_in:
            with open("temp.grib2", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    except Exception as e:
        print(f"Download/Decompression failed: {e}")
        return

    # 3. Open Data with xarray
    try:
        # filter_by_keys is crucial for these complex MRMS files
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", 
                             backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'heightAboveSea'}})
    except Exception:
        # Fallback
        try:
            ds = xr.open_dataset("temp.grib2", engine="cfgrib")
        except Exception as e:
            print(f"GRIB Read Error: {e}")
            return

    # 4. Fix Coordinates (0-360 -> -180-180)
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')
    
    # Get the data variable (Reflectivity)
    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    # 5. Crop to Region
    subset = data.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
    
    # Filter noise (< 5 dBZ)
    radar_clean = subset.where(subset > 5)

    # 6. Plot High-Res Image
    # Using 'equal' aspect ratio to fix the North/South stretching
    fig = plt.figure(figsize=(40, 20), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    
    ax.imshow(radar_clean, 
              extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP],
              aspect='equal', 
              cmap='nipy_spectral', 
              vmin=0, vmax=75,
              interpolation='nearest') 
    
    plt.axis('off')
    
    # Save Image
    out_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(out_path, transparent=True, dpi=100, pad_inches=0)
    plt.close()

    # 7. Metadata
    valid_time = ds.time.values
    ts = datetime.datetime.utcfromtimestamp(valid_time.astype(int) * 1e-9)
    time_str = ts.strftime("%b %d, %H:%M UTC")

    meta = {
        "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
        "time": time_str
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

    print(f"Success. AWS Data processed for {time_str}")

if __name__ == "__main__":
    process_data()
