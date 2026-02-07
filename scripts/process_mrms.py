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
import pytz # Requires: pip install pytz

# --- CONFIGURATION ---
# Adjusted bounds to ensure full CONUS coverage (further South/East focus)
LAT_TOP, LAT_BOT = 50.0, 20.0  # Lowered bottom to 20.0 for Gulf/Florida coverage
LON_LEFT, LON_RIGHT = -130.0, -65.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

def discover_rate_prefix():
    """Scans the S3 bucket to find the correct folder name for Precip Rate."""
    print("DEBUG: Scanning CONUS/ folder for PrecipRate product...")
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                clean_p = p.replace("CONUS/", "").replace("/", "")
                if "PrecipRate" in clean_p or "SurfacePrecip" in clean_p:
                    return p.rstrip("/")
    except Exception:
        pass
    return "CONUS/SurfacePrecipRate_00.00"

def get_xml_keys(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        return [e.text for e in root.iter() if e.tag.endswith('Key')]
    except: return []

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    keys = get_xml_keys(url)
    return sorted([k for k in keys if k.endswith('.grib2.gz')])

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    try:
        r = requests.get(url, stream=True)
        with open(filename + ".gz", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename + ".gz")
    except Exception as e:
        print(f"Download failed: {e}")

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: 
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

def process_frame(index, rate_key, flag_keys):
    timestamp = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    
    if not flag_key: return

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
            ax.imshow(rain_mask.where(rain_mask > 0.1), cmap=get_colormap('rain'), vmin=0.1, vmax=15, extent=extent, aspect='equal', interpolation='nearest')
        if snow_mask.max() > 0:
            ax.imshow(snow_mask.where(snow_mask > 0.1), cmap=get_colormap('snow'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')
        if ice_mask.max() > 0:
            ax.imshow(ice_mask.where(ice_mask > 0.1), cmap=get_colormap('ice'), vmin=0.1, vmax=5, extent=extent, aspect='equal', interpolation='nearest')

        plt.axis('off')
        fname = "master.png" if index == 0 else f"master_{index}.png"
        
        # High DPI for crispness
        plt.savefig(os.path.join(OUTPUT_DIR, fname), transparent=True, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved {fname}")

        if index == 0:
            # Timezone Conversion: UTC -> US/Eastern
            utc_time = datetime.fromtimestamp(ds_rate.time.values.astype(int) * 1e-9, tz=timezone.utc)
            et_time = utc_time.astimezone(pytz.timezone('US/Eastern'))
            
            meta = {
                "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
                "time": et_time.strftime("%I:%M %p ET"), # e.g., 04:30 PM ET
                "raw_ts": et_time.isoformat()
            }
            with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
                json.dump(meta, f)

    except Exception as e:
        print(f"Error frame {index}: {e}")
    finally:
        if os.path.exists("rate.grib2"): os.remove("rate.grib2")
        if os.path.exists("flag.grib2"): os.remove("flag.grib2")

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    for i in range(3):
        date_str = (now_utc - timedelta(days=i)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        
        if rate_keys and flag_keys:
            latest = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, key in enumerate(latest):
                process_frame(idx, key, flag_keys)
            break
