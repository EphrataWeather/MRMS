import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime, timezone, timedelta
import pytz
import gc

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 24.0 # Mercator distorts heavily < 20N, so 24N is safer for CONUS
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# --- SESSION SETUP ---
session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

# --- MERCATOR MATH ---
def lat_to_mercator_y(lat):
    """Convert latitude to Mercator Y (normalized units)"""
    lat_rad = np.radians(lat)
    return np.log(np.tan(np.pi / 4 + lat_rad / 2))

def mercator_y_to_lat(y):
    """Convert Mercator Y back to latitude"""
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

def discover_rate_prefix():
    print("Finding current Rate prefix...")
    url = f"{BUCKET_URL}/?list-type=2&prefix=CONUS/&delimiter=/"
    try:
        r = session.get(url, timeout=10)
        root = ET.fromstring(r.content)
        for element in root.iter():
            if element.tag.endswith('Prefix'):
                p = element.text
                if "PrecipRate" in p or "SurfacePrecip" in p:
                    return p.rstrip("/")
    except: pass
    return "CONUS/SurfacePrecipRate_00.00"

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        return sorted([e.text for e in root.iter() if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')])
    except: return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    print(f"  Downloading {key}...", end="", flush=True)
    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename + ".gz", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename + ".gz")
        print(" Done.")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def process_frame(index, rate_key, flag_keys):
    timestamp_part = rate_key.split('_')[-1]
    time_prefix = timestamp_part[:13]
    flag_key = next((k for k in flag_keys if time_prefix in k), None)
    if not flag_key: return

    tmp_r, tmp_f = f"rate_{index}.grib2", f"flag_{index}.grib2"

    try:
        if not download_and_extract(rate_key, tmp_r): return
        if not download_and_extract(flag_key, tmp_f): return
        
        # Load Data
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # Normalize Longitude
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # Initial Rough Slice (to save memory before warping)
        # We slice a bit wider than needed to avoid edge artifacts during interpolation
        ds_rate = ds_rate.sel(latitude=slice(LAT_TOP+1, LAT_BOT-1), longitude=slice(LON_LEFT, LON_RIGHT))
        ds_flag = ds_flag.sel(latitude=slice(LAT_TOP+1, LAT_BOT-1), longitude=slice(LON_LEFT, LON_RIGHT))

        # --- THE FIX: REPROJECT TO WEB MERCATOR ---
        # 1. Calculate the Mercator Y bounds for our desired box
        merc_top = lat_to_mercator_y(LAT_TOP)
        merc_bot = lat_to_mercator_y(LAT_BOT)
        
        # 2. Create a new linearly spaced Y-axis in MERCATOR space
        # This simulates the "stretching" Leaflet expects
        height_pixels = 800 # Arbitrary high resolution height
        new_merc_y = np.linspace(merc_top, merc_bot, height_pixels)
        
        # 3. Convert these Mercator Y steps back to True Latitude
        # These are the unevenly spaced latitudes we need to sample from the original data
        target_lats = mercator_y_to_lat(new_merc_y)
        
        # 4. Interpolate the data onto this new grid
        # We use nearest neighbor to preserve the crisp radar look (no blurring)
        rate_merc = ds_rate.interp(latitude=xr.DataArray(target_lats, dims="y"), method="nearest")
        flag_merc = ds_flag.interp(latitude=xr.DataArray(target_lats, dims="y"), method="nearest")
        
        # Extract the variable
        rate = rate_merc[list(rate_merc.data_vars)[0]]
        flag = flag_merc[list(flag_merc.data_vars)[0]]

        # 5. Calculate Exact Bounds for Leaflet
        # Since we forced the interpolation to LAT_TOP/LAT_BOT exactly:
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]

        # Apply Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # Plotting
        height, width = rain.shape
        # Use simple aspect ratio since we manually warped the Y-axis
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # 'aspect="auto"' allows the pixels to fill the warped grid
        plot_args = dict(extent=extent, origin='upper', interpolation='none', aspect='auto')
        
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **plot_args)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **plot_args)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **plot_args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # Metadata
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
             utc_dt = datetime.strptime(timestamp_part.split('.')[0], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        # Leaflet bounds are now exactly what we requested
        meta = { "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": et_dt.strftime("%I:%M %p ET") }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"  Processed {index}: {meta['time']} (Mercator Warped)")
        
        ds_rate.close(); ds_flag.close(); gc.collect()

    except Exception as e:
        print(f"  Error on frame {index}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)
    
    processed_count = 0
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        if not rate_keys: continue
            
        target_frames = sorted(rate_keys)[::-1][:NUM_FRAMES]
        print(f"Processing latest {len(target_frames)} frames...")
        
        for idx, r_key in enumerate(target_frames):
            process_frame(idx, r_key, flag_keys)
            processed_count += 1
        if processed_count > 0: break
