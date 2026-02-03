import os
import json
import datetime
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# We use a tighter box for better resolution. 
# If you want the whole US, keep these; for a specific state, make the box smaller.
LAT_TOP = 50.0
LAT_BOT = 20.0
LON_LEFT = -130.0
LON_RIGHT = -60.0

OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_mrms_url():
    """Dynamically finds the latest GRIB2 file from NOAA."""
    # Get today's date in UTC for the URL
    now_utc = datetime.datetime.utcnow()
    date_str = now_utc.strftime("%Y%m%d")
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/mrms/prod/mrms.{date_str}/radar/MergedReflectivityQCComposite_00.50/"
    
    try:
        response = requests.get(base_url, timeout=15)
        if response.status_code != 200:
            # Try yesterday's folder if today's isn't populated yet
            yesterday = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")
            base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/mrms/prod/mrms.{yesterday}/radar/MergedReflectivityQCComposite_00.50/"
            response = requests.get(base_url, timeout=15)

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if 'grib2' in a['href']]
        # Get the very latest file
        return base_url + sorted(links)[-1]
    except Exception as e:
        print(f"URL Fetch Error: {e}")
        return None

def process_data():
    url = get_latest_mrms_url()
    if not url:
        print("Could not find a valid data URL.")
        return

    print(f"Downloading latest MRMS: {url}")
    r = requests.get(url)
    with open("temp.grib2", "wb") as f:
        f.write(r.content)

    # 1. Open and fix coordinates
    # We use filter_by_keys to ensure we only get the reflectivity layer
    ds = xr.open_dataset("temp.grib2", engine="cfgrib", filter_by_keys={'stepType': 'instant'})
    
    # Convert 0..360 longitude to -180..180
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby('longitude')

    # 2. Slice the data to our window
    subset = ds.unknown.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

    # 3. MASKING & SHARPENING
    # MRMS reflectivity is usually 0-80 dBZ. Mask anything below 5 to keep it clean.
    radar_data = subset.where(subset > 5)

    # 4. HIGH-RESOLUTION PLOT
    # We use a massive 40x20 inch figure at 100 DPI for a 4000px wide image.
    # This ensures it looks sharp even when zoomed in.
    fig = plt.figure(figsize=(40, 20), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    
    # Use 'nearest' to prevent the computer from "blurring" the radar pixels
    ax.imshow(radar_data, 
              extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP],
              aspect='equal', # Forces correct map proportions
              cmap='nipy_spectral', # A high-contrast radar-style map
              vmin=0, vmax=75,
              interpolation='nearest')

    plt.axis('off')
    
    # Save as master.png
    image_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(image_path, transparent=True, dpi=800, pad_inches=0)
    plt.close()

    # 5. GENERATE ACCURATE METADATA
    # Use the timestamp from the actual data file
    data_time = ds.time.values
    ts = datetime.datetime.utcfromtimestamp(data_time.astype(int) * 1e-9)
    # Format for the UI
    time_str = ts.strftime("%b %d, %H:%M UTC")

    meta = {
        "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
        "time": time_str
    }

    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(meta, f)

    print(f"Success! Processed data for {time_str}")

if __name__ == "__main__":
    process_data()
