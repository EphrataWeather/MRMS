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
LAT_TOP, LAT_BOT = 50.0, 20.0  # Full CONUS coverage including Gulf
LON_LEFT, LON_RIGHT = -130.0, -65.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# --- HELPERS ---

def discover_rate_prefix():
    """Scans S3 to find the correct PrecipRate folder (names occasionally shift)."""
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                if "PrecipRate" in p or "SurfacePrecip" in p:
                    return p.rstrip("/")
    except Exception: pass
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

# --- CORE PROCESSING ---

def process_frame(index, rate_key, flag_keys):
    # Match the flag file to the rate file based on timestamp in filename
    timestamp_str = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp_str in k), None)
    
    if not flag_key:
        print(f"Skipping frame {index}: No matching flag data.")
        return

    try:
        download_and_extract(rate_key, "rate.grib2")
        download_and_extract(flag_key, "flag.grib2")
        
        ds_rate = xr.open_dataset("rate.grib2", engine="cfgrib")
        ds_flag = xr.open_dataset("flag.grib2", engine="cfgrib")
        
        # Normalize longitude to -180 to 180
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # Select our geographic area
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # Split data by precipitation type
        # Rain: 1, 2, 5, 7, 8 | Snow: 3 | Ice/Mix: 4, 6, 10
        rain_mask = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow_mask = rate.where(flag == 3)
        ice_mask = rate.where(flag.isin([4, 6, 10]))

        # --- THE ALIGNMENT FIX ---
        # We flip the numpy arrays vertically to match standard Map Projections
        def prepare_data(mask):
            data = mask.where(mask > 0.1).values
            return np.flipud(data) # This prevents the "Too Far North" error

        rain_data = prepare_data(rain_mask)
        snow_data = prepare_data(snow_mask)
        ice_data  = prepare_data(ice_mask)

        fig = plt.figure(figsize=(20, 10), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]

        # Plot using origin='upper' to align with the flipped data
        if np.nanmax(rain_data) > 0.1:
            ax.imshow(rain_data, cmap=get_colormap('rain'), vmin=0.1, vmax=50, extent=extent, aspect='equal', interpolation='nearest', origin='upper')
        if np.nanmax(snow_data) > 0.1:
            ax.imshow(snow_data, cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest', origin='upper')
        if np.nanmax(ice_data) > 0.1:
            ax.imshow(ice_data, cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest', origin='upper')

        plt.axis('off')
        img_name = "master.png" if index == 0 else f"master_{index}.png"
        
        # Save high-res transparent image
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- METADATA & TIMEZONE CONVERSION ---
        # Convert GRIB UTC time to US/Eastern
        utc_dt = datetime.fromtimestamp(ds_rate.time.values.astype(int) * 1e-9, tz=timezone.utc)
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        meta = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time": et_dt.strftime("%I:%M %p ET"),
            "raw_ts": et_dt.isoformat()
        }
        
        # Save metadata for this specific frame
        meta_name = f"metadata_{index}.json"
        with open(os.path.join(OUTPUT_DIR, meta_name), "w") as f:
            json.dump(meta, f)

        print(f"Successfully processed {img_name} ({meta['time']})")

    except Exception as e:
        print(f"Error processing frame {index}: {e}")
    finally:
        for f in ["rate.grib2", "flag.grib2"]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    # Try today, then yesterday if data isn't populated yet
    for i in range(2):
        date_str = (now_utc - timedelta(days=i)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if len(rate_keys) >= NUM_FRAMES:
            # Grab the 10 most recent files
            latest_rates = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, r_key in enumerate(latest_rates):
                process_frame(idx, r_key, flag_keys)
            break
        else:
            print(f"Not enough data for {date_str}, checking previous day...")
