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
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timezone, timedelta
import pytz
import gc

# --- CONFIGURATION ---
LAT_TOP, LAT_BOT = 50.0, 24.0
LON_LEFT, LON_RIGHT = -130.0, -60.0
OUTPUT_DIR = "public/data"
NUM_FRAMES = 15
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
FLAG_PREFIX = "CONUS/PrecipFlag_00.00"

# Physically-based rate cap for wintry precipitation (mm/hr liquid equivalent).
# Real snowfall tops out ~5 mm/hr liquid; real ice pellets ~8 mm/hr.
# Anything above this threshold is almost certainly misclassified hail or
# heavy convective rain — force those pixels to the rain layer.
WINTRY_RATE_MAX = 15.0  # mm/hr

# --- SESSION SETUP ---
session = requests.Session()
retry = Retry(connect=3, read=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# --- MERCATOR MATH ---
def lat_to_merc(lat):
    """Converts latitude to normalised Mercator Y."""
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

def merc_to_lat(y):
    """Converts normalised Mercator Y back to latitude."""
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

# --- COLOR TABLES ---
# BoundaryNorm with explicit mm/hr breaks calibrated to match NWS / standard
# MRMS viewers.  The old LogNorm(vmin=0.1, vmax=75) compressed the lower end of
# the scale so that moderate rain (3–6 mm/hr) rendered as heavy-orange/red.
# These explicit boundaries keep green through ~6 mm/hr, matching what users
# see on Weather.gov and other MRMS platforms.

RAIN_BOUNDS = [0.1, 0.3, 1.0, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0]  # mm/hr (8 intervals)
SNOW_BOUNDS = [0.1, 0.3, 0.5, 1.0, 2.5,  6.0]                       # mm/hr (5 intervals)
ICE_BOUNDS  = [0.1, 0.3, 0.5, 1.0, 2.5,  6.0]                       # mm/hr (5 intervals)

RAIN_COLORS = ['#00fb90', '#00bb00', '#008800', '#ffff00', '#ff9100',
               '#ff0000', '#d20000', '#910000']
SNOW_COLORS = ['#00ffff', '#80ffff', '#ffffff', '#adc5ff', '#5a82ff']
ICE_COLORS  = ['#ff00ff', '#d100d1', '#910091', '#4b0082', '#2d004b']

def get_cmap_norm(p_type):
    if p_type == 'snow':
        cmap = ListedColormap(SNOW_COLORS)
        norm = BoundaryNorm(SNOW_BOUNDS, cmap.N)
    elif p_type == 'ice':
        cmap = ListedColormap(ICE_COLORS)
        norm = BoundaryNorm(ICE_BOUNDS, cmap.N)
    else:  # rain
        cmap = ListedColormap(RAIN_COLORS)
        norm = BoundaryNorm(RAIN_BOUNDS, cmap.N)
    cmap.set_bad(alpha=0)  # NaN → fully transparent
    return cmap, norm

# --- HRRR PRECIP TYPE (model-based, avoids radar artefacts) ---
# HRRR categorical precip (CRAIN/CSNOW/CICEP/CFRZR) is fetched via the NCEP
# NOMADS subregion filter — only the 4 surface fields for our bounding box,
# so the download is a few KB rather than the full 200 MB HRRR file.
# Results are cached per HRRR analysis hour to avoid redundant downloads.

_hrrr_cache = {}  # cache key: YYYYMMDDHH string

def get_hrrr_precip_type(target_dt, tgt_lats_2d, tgt_lons_2d):
    """
    Fetch HRRR categorical precip type for our domain and regrid to the target
    Mercator grid.  Returns a dict with boolean 2-D arrays:
        {'rain': ..., 'snow': ..., 'ice': ...}
    Returns None if HRRR is unavailable (caller falls back to MRMS PrecipFlag).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    # Round to the nearest HRRR analysis hour for caching
    hrrr_hour_dt = target_dt.replace(minute=0, second=0, microsecond=0)
    cache_key = hrrr_hour_dt.strftime('%Y%m%d%H')
    if cache_key in _hrrr_cache:
        return _hrrr_cache[cache_key]

    # Try the most recent available HRRR analysis, working back up to 2 h
    for hours_back in range(3):
        run_dt   = hrrr_hour_dt - timedelta(hours=hours_back)
        date_str = run_dt.strftime('%Y%m%d')
        hour_str = run_dt.strftime('%H')

        # NOMADS subregion filter: downloads only CRAIN/CSNOW/CICEP/CFRZR at
        # surface for our lat/lon bounding box (~few KB instead of ~200 MB).
        url = (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
            f"?file=hrrr.t{hour_str}z.wrfsfcf00.grib2"
            f"&dir=%2Fhrrr.{date_str}%2Fconus"
            "&var_CRAIN=on&var_CSNOW=on&var_CICEP=on&var_CFRZR=on"
            "&lev_surface=on"
            f"&subregion=&leftlon={LON_LEFT}&rightlon={LON_RIGHT}"
            f"&toplat={LAT_TOP}&bottomlat={LAT_BOT}"
        )

        fname  = f"hrrr_cat_{hours_back}.grib2"
        all_ds = []
        try:
            resp = session.get(url, timeout=45)
            if resp.status_code != 200 or len(resp.content) < 500:
                continue
            with open(fname, 'wb') as f:
                f.write(resp.content)

            # cfgrib may split 4 categorical variables into separate datasets
            import cfgrib
            all_ds = cfgrib.open_datasets(fname, backend_kwargs={'indexpath': ''})

            # Collect variable arrays and lat/lon from all datasets
            vmap = {}
            hrrr_lat = hrrr_lon = None
            for ds in all_ds:
                for vname in ds.data_vars:
                    vmap[vname.lower()] = ds[vname].values
                if hrrr_lat is None:
                    lat_k = next((k for k in ['latitude', 'lat'] if k in ds.coords), None)
                    lon_k = next((k for k in ['longitude', 'lon'] if k in ds.coords), None)
                    if lat_k and lon_k:
                        hrrr_lat = ds[lat_k].values
                        hrrr_lon = ds[lon_k].values

            if hrrr_lat is None or not vmap:
                continue

            zero  = np.zeros_like(hrrr_lat)
            crain = vmap.get('crain', zero)
            csnow = vmap.get('csnow', zero)
            cicep = vmap.get('cicep', zero)
            cfrzr = vmap.get('cfrzr', zero)

            # Skip if all fields came back empty
            if not (np.any(crain) or np.any(csnow) or np.any(cicep) or np.any(cfrzr)):
                continue

            # Nearest-neighbour regrid: HRRR Lambert Conformal → our Mercator grid
            flat_lons = hrrr_lon.flatten()
            flat_lats = hrrr_lat.flatten()
            tree      = cKDTree(np.column_stack([flat_lons, flat_lats]))
            _, idxs   = tree.query(
                np.column_stack([tgt_lons_2d.flatten(), tgt_lats_2d.flatten()])
            )
            sh = tgt_lons_2d.shape

            def regrid(arr):
                return (arr.flatten()[idxs] >= 0.5).reshape(sh)

            result = {
                'rain': regrid(crain),
                'snow': regrid(csnow),
                'ice':  regrid(cicep) | regrid(cfrzr),
            }

            for ds in all_ds:
                ds.close()
            if os.path.exists(fname):
                os.remove(fname)

            print(f"  HRRR precip type: {hour_str}z (t-{hours_back}h)")
            _hrrr_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"  HRRR t-{hours_back}h failed: {e}")
            for ds in all_ds:
                try:
                    ds.close()
                except Exception:
                    pass
            if os.path.exists(fname):
                os.remove(fname)

    print("  HRRR unavailable — using MRMS PrecipFlag + safeguards")
    _hrrr_cache[cache_key] = None
    return None

# ---------------------------------------------------------------------------

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
    except:
        pass
    return "CONUS/SurfacePrecipRate_00.00"

def get_s3_keys(date_str, prefix):
    url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}/{date_str}/"
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        return sorted([
            e.text for e in root.iter()
            if e.tag.endswith('Key') and e.text.endswith('.grib2.gz')
        ])
    except:
        return []

def download_and_extract(key, filename):
    url = f"{BUCKET_URL}/{key}"
    print(f"  Downloading {key}...", end="", flush=True)
    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename + ".gz", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with gzip.open(filename + ".gz", "rb") as f_in, open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename + ".gz")
        print(" ok")
        return True
    except Exception as e:
        print(f" Failed: {e}")
        return False

def process_frame(index, rate_key, flag_keys):
    # Match rate and flag files by timestamp (YYYYMMDD-HHMM)
    timestamp_part = rate_key.split('_')[-1]
    time_prefix    = timestamp_part[:13]
    flag_key       = next((k for k in flag_keys if time_prefix in k), None)

    if not flag_key:
        print(f"  Skipping {time_prefix}: no matching flag file.")
        return

    tmp_r, tmp_f = f"rate_{index}.grib2", f"flag_{index}.grib2"

    try:
        if not download_and_extract(rate_key, tmp_r):
            return
        if not download_and_extract(flag_key, tmp_f):
            return

        # Load data
        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})

        # Normalise coordinates to [-180, 180]
        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        # --- 1. RESOLUTION & MERCATOR GRID ---
        # 100 pixels/degree matches the native 0.01° MRMS resolution
        res_scale = 100
        width_px  = int((LON_RIGHT - LON_LEFT) * res_scale)

        merc_top          = lat_to_merc(LAT_TOP)
        merc_bot          = lat_to_merc(LAT_BOT)
        merc_height_ratio = (merc_top - merc_bot) / np.radians(LON_RIGHT - LON_LEFT)
        height_px         = int(width_px * merc_height_ratio)

        # Mercator-spaced Y values mapped back to latitude for interpolation
        target_y    = np.linspace(merc_top, merc_bot, height_px)
        target_lats = merc_to_lat(target_y)
        target_lons = np.linspace(LON_LEFT, LON_RIGHT, width_px)

        r_warp = ds_rate[list(ds_rate.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest")
        f_warp = ds_flag[list(ds_flag.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest")

        rate_vals = r_warp.values   # mm/hr, float, NaN where no data
        flag_vals = f_warp.values   # categorical integer

        # --- 2. VALID TIME (needed for HRRR lookup) ---
        try:
            raw_time = ds_rate.valid_time.values
            if isinstance(raw_time, np.ndarray):
                raw_time = raw_time.flat[0]
            utc_dt = datetime.fromtimestamp(
                raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
        except Exception:
            utc_dt = datetime.strptime(
                timestamp_part.split('.')[0], "%Y%m%d-%H%M%S"
            ).replace(tzinfo=timezone.utc)

        # 2-D lat/lon grids for HRRR regridding (same shape as rate/flag)
        tgt_lons_2d, tgt_lats_2d = np.meshgrid(target_lons, target_lats)

        # --- 3. PRECIP TYPE CLASSIFICATION ---
        # Primary: HRRR categorical precip type (model-based, avoids MRMS radar
        # artefacts such as hail being tagged as snow/ice).
        hrrr = get_hrrr_precip_type(utc_dt, tgt_lats_2d, tgt_lons_2d)

        if hrrr is not None:
            has_precip = rate_vals > 0.1

            # Priority: rain > snow > ice (handles HRRR mixed-phase overlap)
            rain_mask = hrrr['rain'] & has_precip
            snow_mask = hrrr['snow'] & has_precip & ~hrrr['rain']
            ice_mask  = hrrr['ice']  & has_precip & ~hrrr['rain'] & ~hrrr['snow']

            # Where HRRR shows no type but MRMS reports precip, fall back to
            # MRMS PrecipFlag for those pixels so nothing is silently dropped.
            hrrr_typed = hrrr['rain'] | hrrr['snow'] | hrrr['ice']
            fallback   = has_precip & ~hrrr_typed
            if np.any(fallback):
                rain_mask |= fallback & np.isin(flag_vals, [1, 2, 91, 96])
                snow_mask |= fallback & np.isin(flag_vals, [3])
                ice_mask  |= fallback & np.isin(flag_vals, [4, 5, 6])
        else:
            # Fallback: MRMS PrecipFlag with enhanced safeguards
            #
            # Flag values:
            #   1      = Rain (warm stratiform)
            #   2      = Rain + hail / convective rain
            #   3      = Snow (dry, pure)
            #   4      = Wet snow  → classified as MIX, not snow.
            #            Wet snow is a transitional melting-layer product that
            #            is frequently triggered by melting hail and bright-band
            #            contamination.  Other MRMS platforms show it as mix.
            #   5      = Sleet / ice pellets
            #   6      = Freezing rain / drizzle
            #   7, 10  = Unknown / no classification — excluded to avoid false
            #            wintry-mix pixels
            #  91, 96  = Multi-sensor / radar-only QPE rain estimates
            rain_mask = np.isin(flag_vals, [1, 2, 91, 96])
            snow_mask = np.isin(flag_vals, [3])           # pure dry snow only
            ice_mask  = np.isin(flag_vals, [4, 5, 6])    # wet snow + sleet + frzr

        # --- 4. RATE SAFEGUARD (applied regardless of classification source) ---
        # Real snow: typically < 5 mm/hr liquid equivalent.
        # Real ice pellets: typically < 8 mm/hr.
        # Anything above WINTRY_RATE_MAX is almost certainly misclassified
        # hail or heavy convective rain.  Force those pixels into the rain layer.
        high_rate  = rate_vals > WINTRY_RATE_MAX
        rain_mask |= (snow_mask | ice_mask) & high_rate
        snow_mask &= ~high_rate
        ice_mask  &= ~high_rate

        # Build float arrays (NaN = transparent)
        rain = np.where(rain_mask, rate_vals, np.nan)
        snow = np.where(snow_mask, rate_vals, np.nan)
        ice  = np.where(ice_mask,  rate_vals, np.nan)

        # --- 5. PLOTTING ---
        fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
        ax  = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        extent    = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]
        plot_args = dict(extent=extent, origin='upper', interpolation='none', aspect='auto')

        rain_cmap, rain_norm = get_cmap_norm('rain')
        snow_cmap, snow_norm = get_cmap_norm('snow')
        ice_cmap,  ice_norm  = get_cmap_norm('ice')

        if np.any(rain > 0.1):
            ax.imshow(rain, cmap=rain_cmap, norm=rain_norm, **plot_args)
        if np.any(snow > 0.1):
            ax.imshow(snow, cmap=snow_cmap, norm=snow_norm, **plot_args)
        if np.any(ice > 0.1):
            ax.imshow(ice,  cmap=ice_cmap,  norm=ice_norm,  **plot_args)

        img_name = "master.png" if index == 0 else f"master_{index}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, img_name), transparent=True, pad_inches=0)
        plt.close()

        # --- 6. METADATA ---
        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta  = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time":   et_dt.strftime("%I:%M %p ET"),
        }
        with open(os.path.join(OUTPUT_DIR, f"metadata_{index}.json"), "w") as f:
            json.dump(meta, f)

        print(f"  Frame {index} saved: {meta['time']} ({width_px}x{height_px})")

        ds_rate.close()
        ds_flag.close()
        gc.collect()

    except Exception as e:
        print(f"  Error on frame {index}: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)

    processed_count = 0
    # Check today (0) and yesterday (1) to handle midnight UTC rollovers
    for d in range(2):
        date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        print(f"--- Checking Date: {date_str} ---")

        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)

        if not rate_keys:
            print("No keys found.")
            continue

        # Newest first
        target_frames = sorted(rate_keys)[::-1][:NUM_FRAMES]
        print(f"Processing {len(target_frames)} frames...")

        for idx, r_key in enumerate(target_frames):
            process_frame(idx, r_key, flag_keys)
            processed_count += 1

        if processed_count > 0:
            print("Batch complete.")
            break

    if processed_count == 0:
        print("NO FRAMES PROCESSED.")
