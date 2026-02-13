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
import gc

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 24.0 
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# --- HELPERS ---
def lat_to_merc(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

def merc_to_lat(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

def get_colormap(p_type):
    if p_type == 'snow':
        return ListedColormap(['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff'])
    elif p_type == 'ice':
        return ListedColormap(['#ff00ff', '#d100d1', '#910091', '#4b0082'])
    else: # Rain
        return ListedColormap(['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100', '#ff0000', '#d20000', '#910000'])

def download_and_extract(key, filename):
    try:
        r = requests.get(f"{BUCKET_URL}/{key}", timeout=30)
        with open(filename + ".gz", "wb") as f: f.write(r.content)
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename + ".gz")
        return True
    except: return False

def process_frame(index, rate_key, flag_keys):
    timestamp = rate_key.split('_')[-1][:13]
    flag_key = next((k for k in flag_keys if timestamp in k), None)
    if not flag_key: return

    tmp_r, tmp_f = f"r_{index}.grib2", f"f_{index}.grib2"
    if not download_and_extract(rate_key, tmp_r) or not download_and_extract(flag_key, tmp_f): return

    try:
        ds_r = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath':''})
        ds_f = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath':''})
        
        # Normalize Lon and Sort
        for ds in [ds_r, ds_f]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        ds_r = ds_r.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_f = ds_f.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # --- HIGH RES RESOLUTION CALCULATION ---
        # 100 pixels per degree matches the 0.01 MRMS resolution
        res_scale = 100 
        width_px = int((LON_RIGHT - LON_LEFT) * res_scale)
        
        # Calculate height based on Mercator stretch to avoid "big pixels" in the North
        merc_y_range = lat_to_merc(LAT_TOP) - lat_to_merc(LAT_BOT)
        # The 60-ish factor relates the Mercator Y units to degrees at the equator
        height_px = int(width_px * (merc_y_range / np.radians(LON_RIGHT - LON_LEFT)))

        # Create the high-res warping grid
        target_y = np.linspace(lat_to_merc(LAT_TOP), lat_to_merc(LAT_BOT), height_px)
        target_lats = merc_to_lat(target_y)
        target_lons = np.linspace(LON_LEFT, LON_RIGHT, width_px)

        # Warp data to the new grid
        r_warp = ds_r[list(ds_r.data_vars)[0]].interp(latitude=target_lats, longitude=target_lons, method="nearest")
        f_warp = ds_f[list(ds_f.data_vars)[0]].interp(latitude=target_lats, longitude=target_lons, method="nearest")

        # Masks
        rain = r_warp.where(f_warp.isin([1, 2, 5, 7, 8]))
        snow = r_warp.where(f_warp == 3)
        ice  = r_warp.where(f_warp.isin([4, 6, 10]))

        # --- SHARP PLOTTING ---
        # We set DPI to 100 and figsize to the exact pixel count for 1:1 mapping
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        extent = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]
        # Use aspect='auto' because we already manually handled the Mercator ratio in height_px
        args = dict(extent=extent, origin='upper', interpolation='none', aspect='auto')
        
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **args)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **args)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # Metadata
        try:
            raw_time = ds_r.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
            utc_dt = datetime.now(timezone.utc)

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta = { "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]], "time": et_dt.strftime("%I:%M %p ET") }
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Sharp Frame {index} saved ({width_px}x{height_px})")
        ds_r.close(); ds_f.close(); gc.collect()

    except Exception as e: print(f"Error: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Logic to fetch keys and call process_frame(idx, key, flag_keys)...
