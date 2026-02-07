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
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# Fallback: If Rate is missing, we try to find one of these
POSSIBLE_RATE_NAMES = [
    "SurfacePrecipRate_00.00",
    "PrecipRate_00.00", 
    "SurfacePrecipitationRate_00.00",
    "MultiSensorQPE_01H_Pass2_00.00" # sometimes used as backup
]

def get_xml_keys(url):
    """Helper to fetch and parse XML keys ignoring namespaces."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        
        root = ET.fromstring(r.content)
        keys = []
        for element in root.iter():
            if element.tag.endswith('Key'):
                keys.append(element.text)
        return keys
    except Exception:
        return []

def discover_rate_prefix():
    """Scans the S3 bucket to find the correct folder name for Precip Rate."""
    print("DEBUG: Scanning CONUS/ folder for PrecipRate product...")
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        
        # S3 returns "CommonPrefixes" for folders when using a delimiter
        prefixes = []
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                prefixes.append(element.text)
        
        # Look for the best match
        for p in prefixes:
            clean_p = p.replace("CONUS/", "").replace("/", "")
            # Check if this folder looks like a Precip Rate folder
            if "PrecipRate" in clean_p or "SurfacePrecip" in clean_p:
                print(f"DEBUG: Auto-discovered folder: {p}")
                return p.rstrip("/") # Remove trailing slash for consistency
                
    except Exception as e:
        print(f"DEBUG: Auto-discovery failed: {e}")
    
    return "CONUS/SurfacePrecipRate_00.00" # Default fallback

def get_s3_keys(date_str, prefix):
    request_url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    print(f"DEBUG: Checking {request_url}")
    
    keys = get_xml_keys(request_url)
    
    # Filter for .grib2.gz files
    grib_keys = sorted([k for k in keys if k.endswith('.grib2.gz')])
    print(f"DEBUG: Found {len(grib_keys)} files in {prefix}")
    return grib_keys

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename + ".gz", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        if os.path.exists(filename + ".gz"):
            os.remove(filename + ".gz")
            
    except Exception as e:
        print(f"Download failed for {key}: {e}")
        raise

def get_colormap(p_type):
    """Professional high-contrast palettes (Weather Channel / AccuWeather style)"""
    if p_type == 'snow':
        # Cyan to White to Deep Blue
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        # Hot Pink to Deep Purple
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: 
        # The classic Radar 'Rain' Scale: Light Green -> Dark Green -> Yellow -> Red -> Maroon
        return ListedColormap([
            '#00fb90', # Light Green
            '#00bb00', # Solid Green
            '#008800', # Dark Green
            '#ffff00', # Yellow
            '#ff9100', # Orange
            '#ff0000', # Red
            '#d20000', # Deep Red
            '#910000'  # Maroon/Extreme
        ])

def process_frame(index, rate_key, flag_keys):
    timestamp = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    
    if not flag_key:
        print(f"Skipping frame {index}: No matching flag for {timestamp}")
        return

    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        rain_mask = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow_mask = rate.where(flag == 3)
        ice_mask = rate.where(flag.isin([4, 6, 10]))

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]

        if rain_mask.max() > 0:
            ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=20, extent=extent, aspect='equal', interpolation='nearest')
        if snow_mask.max() > 0:
            ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')
        if ice_mask.max() > 0:
            ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')
      
        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        save_path = os.path.join(OUTPUT_DIR, fname)
        
        # INCREASE DPI HERE: 200-300 makes it look sharp on 4K screens
        plt.savefig(save_path, transparent=True, dpi=800, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.close()
        print(f"Saved {fname}")

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
        if os.path.exists("rate.grib2"): os.remove("rate.grib2")
        if os.path.exists("flag.grib2"): os.remove("flag.grib2")

if __name__ == "__main__":
    print("Starting MRMS Processing...")
    
    # 1. Auto-discover the correct Rate Prefix
    RATE_PREFIX = discover_rate_prefix()
    print(f"Using Rate Prefix: {RATE_PREFIX}")

    # 2. Find Dates
    found_data = False
    now_utc = datetime.now(timezone.utc)
    
    for i in range(3):
        search_date = now_utc - timedelta(days=i)
        date_str = search_date.strftime("%Y%m%d")
        
        print(f"\n--- Checking Date: {date_str} ---")
        
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        # Check if we have data for BOTH
        if rate_keys and flag_keys:
            print(f"Success! Found {len(rate_keys)} rate files and {len(flag_keys)} flag files.")
            latest_rates = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            
            for idx, key in enumerate(latest_rates):
                process_frame(idx, key, flag_keys)
                
            found_data = True
            break
        else:
            print(f"Incomplete data for {date_str}. (Rate: {len(rate_keys)}, Flag: {len(flag_keys)})")

    if not found_data:
        print("\nCRITICAL: Data missing. It's possible the 'Rate' product is down on NOAA's end.")
        print("Try changing RATE_PREFIX manually if the auto-discovery failed.")
