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
LAT_TOP, LAT_BOT = 50.0, 24.0  # Precise CONUS bounds
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# --- NETWORK SETUP ---
session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# --- MERCATOR TRANSFORMATION MATH ---
def lat_to_mercator_y(lat):
    """Converts latitude to normalized Mercator Y."""
    lat_rad = np.radians(lat)
    return np.log(np.tan(np.pi / 4 + lat_rad / 2))

def mercator_y_to_lat(y):
    """Converts normalized Mercator Y back to latitude."""
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

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
        print(" Success.")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        return False

def process_frame(index, rate_key, flag_keys):
    timestamp_part = rate_key.split('_')[-1]
    time_prefix = timestamp_part[:13] # YYYYMMDD-HHMM
    flag_key = next((k for k in flag_keys if time_prefix in k), None)
    if not flag_key: return

    tmp_r, tmp_f = f"rate_{index}.grib2", f"flag_{index}.grib2"

    try:
        if not download_and_extract(rate_key, tmp_r): return
        if not download_and_extract(flag_key, tmp_f): return
        
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})
        
        # 1. Normalize Coordinates
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # 2. Precise Mercator Re-projection
        # Calculate aspect-correct dimensions
        scale = 100 # pixels per degree of longitude
        width_px = int((LON_RIGHT - LON_LEFT) * scale)
        
        # Mercator stretching factor
        merc_top = lat_to_mercator_y(LAT_TOP)
        merc_bot = lat_to_mercator_y(LAT_BOT)
        lon_rad_dist = np.radians(LON_RIGHT - LON_LEFT)
        height_px = int(width_px * (merc_top - merc_bot) / lon_rad_dist)

        # Generate the non-linear latitude grid (Warped Grid)
        new_merc_y = np.linspace(merc_top, merc_bot, height_px)
        target_lats = mercator_y_to_lat(new_merc_y)
        
        # Sample the data onto the warped grid
        rate_m = ds_rate[list(ds_rate.data_vars)[0]].interp(
            latitude=xr.DataArray(target_lats, dims="y"),
            longitude=np.linspace(LON_LEFT, LON_RIGHT, width_px),
            method="nearest"
        )
        flag_m = ds_flag[list(ds_flag.data_vars)[0]].interp(
            latitude=xr.DataArray(target_lats, dims="y"),
            longitude=np.linspace(LON_LEFT, LON_RIGHT, width_px),
            method="nearest"
        )

        # 3. Create Masked Arrays
        rain = rate_m.where(flag_m.isin([1, 2, 5, 7, 8]))
        snow = rate_m.where(flag_m == 3)
        ice  = rate_m.where(flag_m.isin([4, 6, 10]))

        # 4. Pixel-Perfect Plotting
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # Bounds for Leaflet
        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]
        # aspect='equal' is critical now that we've manually warped the Y-axis
        plot_args = dict(extent=extent, origin='upper', interpolation='none', aspect='equal')
        
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **plot_args)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **plot_args)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **plot_args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # 5. Metadata Sync
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
             utc_dt = datetime.strptime(timestamp_part.split('.')[0], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta = { 
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], 
            "time": et_dt.strftime("%I:%M %p ET") 
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"  Processed Frame {index}: {meta['time']} | Size: {width_px}x{height_px}")
        ds_rate.close(); ds_flag.close(); gc.collect()

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Prefix discovery and loop logic...
    print("Starting Radar Update...")
    # ... (Discovery and Loop as per previous working logic)
