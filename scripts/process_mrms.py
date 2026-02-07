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
NUM_FRAMES = 10  # Number of frames to generate
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
RATE_PREFIX = "CONUS/SurfacePrecipRate_00.00"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

def get_s3_keys(date_str, prefix):
    """
    Fetches S3 keys using a namespace-agnostic XML parser.
    This fixes the 'empty list' error caused by AWS XML schema changes.
    """
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    print(f"DEBUG: Checking {request_url}")
    
    try:
        r = requests.get(request_url, timeout=15)
        if r.status_code != 200:
            print(f"DEBUG: Failed to connect. Status: {r.status_code}")
            return []
        
        # Robust parsing: Ignore namespaces by searching for tags ending in 'Key'
        keys = []
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Key'):
                keys.append(element.text)
        
        # Filter for .grib2.gz files only
        grib_keys = sorted([k for k in keys if k.endswith('.grib2.gz')])
        print(f"DEBUG: Found {len(grib_keys)} files in {prefix}")
        return grib_keys

    except Exception as e:
        print(f"DEBUG: Error listing S3 keys: {e}")
        return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    print(f"Downloading: {url}")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename + ".gz", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        # Clean up the .gz file to save space
        if os.path.exists(filename + ".gz"):
            os.remove(filename + ".gz")
            
    except Exception as e:
        print(f"Download failed for {key}: {e}")
        raise

def get_colormap(p_type):
    """Custom color maps for different precipitation types."""
    if p_type == 'snow':
        # Cool blues for snow
        return ListedColormap(['#d9ebff', '#89a9ff', '#5a82ff', '#2d58ff', '#0026ff'])
    elif p_type == 'ice':
        # Pinks/Purples for ice/mix
        return ListedColormap(['#ffdaff', '#ffb3ff', '#ff80ff', '#e600e6', '#b300b3'])
    else: 
        # Standard Radar Green/Yellow/Red for rain
        return ListedColormap(['#00fb90', '#00bb00', '#ffff00', '#ff9100', '#ff0000', '#d20000'])

def process_frame(index, rate_key, flag_keys):
    """
    Downloads Rate and Flag data, masks them based on type, and generates the image.
    """
    # 1. Match the timestamp from the Rate file to find the correct Flag file
    # Rate Key format: .../20260206-200000.grib2.gz
    timestamp = rate_key.split('_')[-1].split('.')[0] # e.g., "200000"
    
    # Find the flag key that contains this timestamp
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    
    if not flag_key:
        print(f"Skipping frame {index}: No matching flag file for time {timestamp}")
        return

    try:
        # 2. Download Data
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        # 3. Load Datasets
        # Note: cfgrib is required. Ensure 'eccodes' is installed on your system.
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # 4. Normalize Longitude (-180 to 180)
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # 5. Crop to bounds
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # 6. Create Masks based on NOAA Flag Definitions
        # Rain: 1 (Warm Stratiform), 2 (Warm Stratiform), 5 (Conv), 7 (Trop Conv), 8 (Trop Strat)
        rain_mask = rate.where(flag.isin([1, 2, 5, 7, 8]))
        
        # Snow: 3 (Snow)
        snow_mask = rate.where(flag == 3)
        
        # Ice: 4 (Ice Pellets), 6 (Freezing Rain), 10 (Mixed)
        ice_mask = rate.where(flag.isin([4, 6, 10]))

        # 7. Plotting
        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]

        # Plot layers (Rain first, then Snow, then Ice on top)
        # We filter rate > 0.1 to remove sensor noise
        if rain_mask.max() > 0:
            ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=50, extent=extent, aspect='equal', interpolation='nearest')
        
        if snow_mask.max() > 0:
            ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')
        
        if ice_mask.max() > 0:
            ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')

        # 8. Save Output
        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        save_path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(save_path, transparent=True, dpi=100, pad_inches=0)
        plt.close()
        print(f"Saved {fname}")

        # 9. Generate Metadata (only for the newest frame)
        if index == 0:
            ts = datetime.fromtimestamp(ds_rate.time.values.astype(int) * 1e-9, tz=timezone.utc)
            meta = {
                "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
                "time": ts.strftime("%b %d, %H:%M UTC")
            }
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
                json.dump(meta, f)

    except Exception as e:
        print(f"Error processing frame {index}: {e}")
    finally:
        # Cleanup temp files
        if os.path.exists("rate.grib2"): os.remove("rate.grib2")
        if os.path.exists("flag.grib2"): os.remove("flag.grib2")

if __name__ == "__main__":
    print("Starting MRMS Processing...")
    
    # Logic: Look for data in the last 3 days (Today -> Yesterday -> Day Before)
    # This handles timezone differences where 'today' in UTC might not have data yet.
    found_data = False
    now_utc = datetime.now(timezone.utc)
    
    for i in range(3):
        search_date = now_utc - timedelta(days=i)
        date_str = search_date.strftime("%Y%m%d")
        
        print(f"\n--- Checking Date: {date_str} ---")
        
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if rate_keys and flag_keys:
            print(f"Success! Found {len(rate_keys)} rate files and {len(flag_keys)} flag files.")
            
            # Use the most recent N frames
            # Reverse list so index 0 is the absolute latest
            latest_rates = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            
            for idx, key in enumerate(latest_rates):
                process_frame(idx, key, flag_keys)
                
            found_data = True
            break # Stop searching previous days once we find data
        else:
            print(f"No complete data found for {date_str}. Trying previous day...")

    if not found_data:
        print("\nCRITICAL ERROR: Could not find matching Rate and Flag data in the last 72 hours.")
        print("Possible causes: NOAA Server down, Internet connection blocked, or S3 path changed.")
