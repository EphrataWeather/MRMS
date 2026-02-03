import os
import json
import datetime
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# These coordinates define the "canvas" for the image.
# If the rain is too far North, decrease both LAT values by 0.1 or 0.2.
LAT_TOP = 55.0
LAT_BOT = 20.0
LON_LEFT = -130.0
LON_RIGHT = -60.0

OUTPUT_DIR = "public/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_latest_mrms_url():
    """Finds the most recent GRIB2 file from the NCEP server."""
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/mrms/prod/mrms.20260203/radar/MergedReflectivityQCComposite_00.50/"
    # Note: In a real script, you'd dynamically generate today's date in the URL
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if 'grib2' in a['href']]
        return base_url + sorted(links)[-1]
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

def process_data():
    url = get_latest_mrms_url()
    if not url:
        return

    print(f"Downloading: {url}")
    r = requests.get(url)
    with open("temp.grib2", "wb") as f:
        f.write(r.content)

    # Load data using cfgrib
    ds = xr.open_dataset("temp.grib2", engine="cfgrib")
    
    # Extract reflectivity and handle coordinates
    # MRMS usually uses 0-360 for Longitude; we convert to -180 to 180
    data = ds.unknown  # reflectivity variable
    data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180))
    
    # Crop to the US window
    subset = data.sel(latitude=slice(LAT_TOP, LAT_BOT), longitude=slice(LON_LEFT, LON_RIGHT))

    # --- HIGH QUALITY RENDERING ---
    # We use a large figure size and high DPI to prevent blurriness
    fig = plt.figure(figsize=(20, 10), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    
    # Mask out the very low values (background noise)
    masked_data = subset.where(subset > 0)

    # Plotting with 'nearest' ensures the radar bins stay crisp
    im = ax.imshow(masked_data, 
                   extent=[LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP],
                   aspect='auto', 
                   cmap='pyart_NWSRef', # Requires a weather colormap or standard 'jet'/'nipy_spectral'
                   vmin=0, vmax=80,
                   interpolation='nearest')

    # Save the high-res image
    image_path = os.path.join(OUTPUT_DIR, "master.png")
    plt.savefig(image_path, transparent=True, dpi=300, pad_inches=0)
    plt.close()

    # --- ACCURATE TIMESTAMPS ---
    # We pull the time directly from the data file itself for 100% accuracy
    dt = ds.time.values
    timestamp = datetime.datetime.utcfromtimestamp(dt.astype(int) * 1e-9)
    # Adjust for your timezone if preferred, currently UTC
    formatted_time = timestamp.strftime("%I:%M %p UTC")

    metadata = {
        "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
        "time": formatted_time,
        "updated": datetime.datetime.utcnow().isoformat()
    }

    with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Successfully processed frame at {formatted_time}")

if __name__ == "__main__":
    process_data()
