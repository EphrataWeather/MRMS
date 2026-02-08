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
# We use slightly wider bounds to ensure we capture the edges, then slice tightly
LAT_TOP, LAT_BOT = 50.0, 20.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

def discover_rate_prefix():
    # ... (Keep existing discover logic) ...
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

def process_frame(index, rate_key, flag_keys):
    timestamp_str = rate_key.split('_')[-1].split('.')[0]
    flag_key = next((k for k in flag_keys if timestamp_str in k), None)
    
    if not flag_key: return

    try:
        tmp_r, tmp_f = f"rate_{index}.grib2", f"flag_{index}.grib2"
        download_and_extract(rate_key, tmp_r)
        download_and_extract(flag_key, tmp_f)
        
        # Open dataset
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib")
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib")
        
        # 1. Normalize Longitude (-180 to 180)
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
            
        # 2. Strict Sort: North-to-South (Descending Lat)
        ds_rate = ds_rate.sortby("latitude", ascending=False)
        ds_flag = ds_flag.sortby("latitude", ascending=False)
        
        # 3. Slice Data
        # slice() works on the INDEX values. Since Lat is descending: Top -> Bot
        rate = ds_rate[list(ds_rate.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
        flag = ds_flag[list(ds_flag.data_vars)[0]].sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

        # --- THE FIX: PIXEL EDGE CALCULATION ---
        # MRMS grid is 0.01 deg. Coordinates are CENTER of pixel.
        # We need the EDGES for the extent.
        res = 0.01
        
        # Get the actual coordinate arrays
        lats = rate.latitude.values
        lons = rate.longitude.values
        
        # Calculate edges
        north = lats[0] + (res / 2)  # Top edge of top pixel
        south = lats[-1] - (res / 2) # Bottom edge of bottom pixel
        west  = lons[0] - (res / 2)  # Left edge of left pixel
        east  = lons[-1] + (res / 2) # Right edge of right pixel
        
        extent = [west, east, south, north]

        # 4. Create Masks
        rain = rate.where(flag.isin([1, 2, 5, 7, 8]))
        snow = rate.where(flag == 3)
        ice  = rate.where(flag.isin([4, 6, 10]))

        # --- HIGH RES PLOTTING ---
        height_px, width_px = rain.shape
        # Force high DPI (300) to stop blurriness
        fig = plt.figure(figsize=(width_px/100, height_px/100), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        # IMPORTANT: 'interpolation=none' prevents smoothing
        # 'origin=upper' aligns the array [0,0] to the Top-Left corner (North-West)
        plot_args = dict(extent=extent, origin='upper', interpolation='none')
        
        if np.nanmax(rain.values) > 0.1:
            ax.imshow(rain.values, cmap=get_colormap('rain'), vmin=0.1, vmax=15, **plot_args)
        if np.nanmax(snow.values) > 0.1:
            ax.imshow(snow.values, cmap=get_colormap('snow'), vmin=0.1, vmax=5, **plot_args)
        if np.nanmax(ice.values) > 0.1:
            ax.imshow(ice.values, cmap=get_colormap('ice'), vmin=0.1, vmax=5, **plot_args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # 5. Metadata (Use strict edges)
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray): raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except:
             utc_dt = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        
        # EXPORT BOUNDS: [[South, West], [North, East]]
        meta = {
            "bounds": [[south, west], [north, east]], 
            "time": et_dt.strftime("%I:%M %p ET")
        }
        
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Processed {img_name} | Bounds: {north:.3f}N - {south:.3f}S")

    except Exception as e:
        print(f"Error on frame {index}: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    # ... (Keep existing loop logic) ...
    now_utc = datetime.now(timezone.utc)
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        if len(rate_keys) >= NUM_FRAMES:
            latest = sorted(rate_keys)[-NUM_FRAMES:][::-1]
            for idx, r_key in enumerate(latest):
                process_frame(idx, r_key, flag_keys)
            break
