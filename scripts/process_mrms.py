import os
import requests
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import glob
from bs4 import BeautifulSoup
from scipy.ndimage import gaussian_filter
from PIL import Image
from datetime import datetime

# --- CONFIGURATION ---
BASE_URL = "https://mrms.ncep.noaa.gov/data/2D/"
PROD_REF = "MergedReflectivityQCComposite"
PROD_TYPE = "PrecipFlag"
OUTPUT_DIR = "public/data"
TILE_DIR = "public/data/tiles"
FRAMES_TO_KEEP = 6  # Last ~1 hour (MRMS updates ~ every 10 mins)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TILE_DIR, exist_ok=True)

def get_file_list(product):
    """Scrapes the MRMS directory for the latest .grib2.gz files."""
    url = f"{BASE_URL}{product}/"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.grib2.gz')]
        links.sort()
        return links # Return just filenames to match easier
    except Exception as e:
        print(f"Error fetching {product}: {e}")
        return []

def download_and_extract(url, local_path):
    """Downloads and gunzips a file."""
    gz_path = local_path + ".gz"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        os.system(f"gunzip -f {gz_path}")
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False

def generate_color_composite(refl_data, type_data):
    """
    Combines Reflectivity (Intensity) and PrecipFlag (Type) into an RGBA image.
    Rules:
    - Snow (3): Blue Gradient
    - Rain (1,2): Green Gradient
    - Mix/Ice (4,6,7): Pink/Purple Gradient
    """
    # Normalize Reflectivity (0-75 dBZ)
    norm = mcolors.Normalize(vmin=5, vmax=70)
    
    # Create empty RGBA array
    rows, cols = refl_data.shape
    rgba_img = np.zeros((rows, cols, 4))
    
    # Masks
    # MRMS PrecipFlag: 0=None, 1=Warm Stratiform, 2=Warm Convective, 3=Snow, 4=Conv Snow, 6=Hail, 7=Mix
    mask_precip = (refl_data > 5) & (type_data >= 0)
    
    mask_snow = (type_data == 3) | (type_data == 4)
    mask_mix  = (type_data == 6) | (type_data == 7) | (type_data == 91)
    mask_rain = (type_data == 1) | (type_data == 2) | (~mask_snow & ~mask_mix) # Fallback to rain
    
    # Colormaps
    cmap_rain = plt.get_cmap('gist_ncar') # High contrast for rain
    cmap_snow = plt.get_cmap('winter')    # Blues/Cyans for snow
    cmap_mix  = plt.get_cmap('cool_r')    # Pinks/Purples for mix
    
    # Apply Colors
    # We use the normalized reflectivity to pick the intensity color from the cmap
    
    # 1. Rain (Standard Radar Colors)
    # We tweak the raw reflectivity to stick to standard NWS colors loosely
    valid_rain = mask_precip & mask_rain
    if np.any(valid_rain):
        # Using nipy_spectral for rain gives that classic rainbow look
        # Or use a custom lookup. Let's use nipy_spectral for rain as it's familiar.
        rgba_img[valid_rain] = plt.get_cmap('nipy_spectral')(norm(refl_data[valid_rain]))

    # 2. Snow (Blue/White)
    valid_snow = mask_precip & mask_snow
    if np.any(valid_snow):
        rgba_img[valid_snow] = cmap_snow(norm(refl_data[valid_snow]))
        
    # 3. Mix (Pink/Purple)
    valid_mix = mask_precip & mask_mix
    if np.any(valid_mix):
        rgba_img[valid_mix] = cmap_mix(norm(refl_data[valid_mix]))

    # Apply Alpha: Transparent where dBZ < 5
    rgba_img[refl_data < 5, 3] = 0.0
    
    return rgba_img

def slice_to_tiles(image_path, timestamp, rows=4, cols=4):
    """Slices master image into tiles and returns metadata."""
    img = Image.open(image_path)
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    tile_list = []
    
    for r in range(rows):
        for c in range(cols):
            left, upper = c * tile_w, r * tile_h
            right = left + tile_w if c < cols-1 else w
            lower = upper + tile_h if r < rows-1 else h
            
            tile = img.crop((left, upper, right, lower))
            # Unique filename per timestamp to allow caching/looping
            tile_name = f"tile_{timestamp}_{r}_{c}.png"
            tile.save(os.path.join(TILE_DIR, tile_name))
            tile_list.append({"row": r, "col": c, "url": f"data/tiles/{tile_name}"})
            
    return tile_list

def process_frame(timestamp, refl_file, type_file):
    """Process a single timestamp into a composite image and tiles."""
    local_refl = f"temp_refl_{timestamp}.grib2"
    local_type = f"temp_type_{timestamp}.grib2"
    
    try:
        # Download
        if not download_and_extract(f"{BASE_URL}{PROD_REF}/{refl_file}", local_refl): return None
        if not download_and_extract(f"{BASE_URL}{PROD_TYPE}/{type_file}", local_type): return None

        # Load Data
        ds_refl = xr.open_dataset(local_refl, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        ds_type = xr.open_dataset(local_type, engine='cfgrib', backend_kwargs={'errors': 'ignore'})
        
        refl = ds_refl[list(ds_refl.data_vars)[0]].values
        ptype = ds_type[list(ds_type.data_vars)[0]].values
        
        # Coordinates
        lats = ds_refl.latitude.values
        lons = ds_refl.longitude.values
        lons = np.where(lons > 180, lons - 360, lons)
        
        # Smooth Reflectivity only (don't smooth type flags!)
        refl = gaussian_filter(refl, sigma=0.5)

        # Generate Image
        rgba_data = generate_color_composite(refl, ptype)
        
        # Plotting
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        h_w_ratio = (lat_max - lat_min) / (lon_max - lon_min)
        
        fig = plt.figure(figsize=(20, 20 * h_w_ratio), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        
        ax.imshow(rgba_data, extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
        
        master_path = os.path.join(OUTPUT_DIR, f"master_{timestamp}.png")
        plt.savefig(master_path, transparent=True, pad_inches=0, dpi=300)
        plt.close()
        
        tiles = slice_to_tiles(master_path, timestamp)
        
        # Cleanup Master
        os.remove(master_path)
        
        return {
            "time": timestamp,
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
            "tiles": tiles
        }

    except Exception as e:
        print(f"Failed to process {timestamp}: {e}")
        return None
    finally:
        for f in [local_refl, local_type]:
            if os.path.exists(f): os.remove(f)

def main():
    # 1. Clear old tiles to prevent disk fill-up on runner
    for f in glob.glob(f"{TILE_DIR}/*.png"): os.remove(f)

    # 2. Get available files
    refl_files = get_file_list(PROD_REF)
    type_files = get_file_list(PROD_TYPE)
    
    # 3. Match them by timestamp
    # filenames look like: MRMS_MergedReflectivityQCComposite_00.50_20231027-120000.grib2.gz
    # Timestamp is the last part before .grib2.gz
    
    pairs = []
    type_map = {f.split('_')[-1]: f for f in type_files}
    
    for rf in refl_files:
        ts = rf.split('_')[-1]
        if ts in type_map:
            pairs.append((ts.split('.')[0], rf, type_map[ts]))
            
    # 4. Process last N frames
    # Sort by timestamp
    pairs.sort(key=lambda x: x[0])
    target_pairs = pairs[-FRAMES_TO_KEEP:]
    
    frames_data = []
    
    print(f"Processing {len(target_pairs)} frames...")
    
    for ts, rf, tf in target_pairs:
        print(f"Processing {ts}...")
        result = process_frame(ts, rf, tf)
        if result:
            frames_data.append(result)
            
    # 5. Save Metadata
    if frames_data:
        output = {
            "generated_at": datetime.now().isoformat(),
            "frames": frames_data
        }
        with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
            json.dump(output, f)

if __name__ == "__main__":
    main()
