import os
import json
import datetime
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# Adjust these slightly if the radar is still too far North/South
# Decrease LAT values (e.g. 49.5, 19.5) to shift the image DOWN (South)
LAT_TOP = 50.0
LAT_BOT = 20.0
LON_LEFT = -130.0
LON_RIGHT = -60.0

OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_file_list(base_url):
    """Helper to fetch and parse file links from a URL."""
    try:
        print(f"Checking: {base_url}")
        r = requests.get(base_url, timeout=10)
        if r.status_code != 200:
            return []
        
        soup = BeautifulSoup(r.text, 'html.parser')
        # Look for both .grib2 and .grib2.gz files
        links = [a['href'] for a in soup.find_all('a', href=True) 
                 if 'grib2' in a['href'] and 'latest' not in a['href']]
        return sorted(links)
    except Exception as e:
        print(f"Connection error: {e}")
        return []

def get_latest_mrms_url():
    """Smart fetcher: Tries today, then falls back to yesterday."""
    now_utc = datetime.datetime.utcnow()
    
    # 1. Try TODAY
    date_str = now_utc.strftime("%Y%m%d")
    base = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/mrms/prod/mrms"
    product = "radar/MergedReflectivityQCComposite_00.50/"
    
    url_today = f"{base}.{date_str}/{product}"
    files = get_file_list(url_today)
    
    if len(files) > 0:
        return url_today + files[-1]

    # 2. If today is empty, try YESTERDAY
    print("Today's folder is empty. Checking yesterday...")
    yesterday = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")
    url_yesterday = f"{base}.{yesterday}/{product}"
    files = get_file_list(url_yesterday)
    
    if len(files) > 0:
        return url_yesterday + files[-1]

    return None

def process_data():
    # 1. Get the URL
    url = get_latest_mrms_url()
    if not url:
        print("CRITICAL ERROR: No radar data found for today or yesterday.")
        return

    print(f"Downloading: {url}")
    
    # 2. Download
    try:
        r = requests.get(url, stream=True)
        with open("temp.grib2", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # 3. Open Data
    try:
        # filter_by_keys prevents errors with multi-message grib files
        ds = xr.open_dataset("temp.grib2", engine="cfgrib", 
                             backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'heightAboveSea'}})
    except Exception:
        # Fallback for different GRIB structures
        try:
            ds = xr.open_dataset("temp.grib2", engine="cfgrib")
        except Exception as e:
            print(f"GRIB Read Error: {e}")
            return

    # 4. Process Coordinates (0-360 to -180/180)
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')
    
    # Select the correct variable (Reflectivity)
    # Usually 'unknown', 'paramId_0', or similar. We look for the data variable.
    var_name = list(ds.data_vars)[0]
    data = ds[var_name]

    # 5. Crop to US Bounds
    subset = data.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))
    
    # Filter noise
    radar_clean = subset.where(subset > 5)

    # 6. Plot High-Res Image
    # figsize=(40,20) ensures very high quality (4000x2000 pixels approx)
    fig = plt.figure(figsize=(40, 20), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    
    ax.imshow(radar_clean, 
              extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP],
              aspect='equal',  # THIS FIXES THE NORTH/SOUTH SQUASHING
              cmap='nipy_spectral', 
              vmin=0, vmax=75,
              interpolation='nearest') # 'nearest' keeps pixels sharp
    
    plt.axis('off')
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(out_path, transparent=True, dpi=100, pad_inches=0)
    plt.close()

    # 7. Metadata
    # Get accurate timestamp from the file
    valid_time = ds.time.values
    ts = datetime.datetime.utcfromtimestamp(valid_time.astype(int) * 1e-9)
    time_str = ts.strftime("%b %d, %H:%M UTC")

    meta = {
        "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
        "time": time_str
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

    print(f"Success. Image saved for {time_str}")

if __name__ == "__main__":
    process_data()
