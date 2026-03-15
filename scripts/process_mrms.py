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

# --- COLOR TABLES: PRECIPITATION RATE ---
RAIN_BOUNDS = [0.1, 0.5, 1.5, 3.0, 5.1, 12.7, 25.4, 38.1, 50.8, 76.2, 127.0]  # mm/hr
SNOW_BOUNDS = [0.1, 0.3, 0.5, 1.0, 2.5, 6.0]                                    # mm/hr
ICE_BOUNDS  = [0.1, 0.3, 0.5, 1.0, 2.5, 6.0]                                    # mm/hr

RAIN_COLORS = [
    '#00fb90',  # 0.1  – 0.5   mm/hr : mint / light green
    '#00cc00',  # 0.5  – 1.5   mm/hr : medium-light green
    '#009900',  # 1.5  – 3.0   mm/hr : medium green
    '#006600',  # 3.0  – 5.1   mm/hr : dark green
    '#ffff00',  # 5.1  – 12.7  mm/hr : bright yellow
    '#ffcc00',  # 12.7 – 25.4  mm/hr : amber
    '#ff9100',  # 25.4 – 38.1  mm/hr : orange
    '#ff5500',  # 38.1 – 50.8  mm/hr : red-orange
    '#ff0000',  # 50.8 – 76.2  mm/hr : red
    '#cc0000',  # 76.2 – 127   mm/hr : dark red
]
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

# --- COLOR TABLES: SINGLE-FIELD PRODUCTS ---
# Each entry: prefix, bounds (N+1 values for N color intervals), colors (N values),
# min_val (display threshold), max_val (fill-value cap).
SINGLE_PRODUCTS = {
    'mesh': {
        'prefix': 'CONUS/MESH_00.00',
        'bounds': [5, 10, 15, 20, 25, 30, 40, 50, 65],   # mm hail diameter
        'colors': [
            '#c8f500',  # 5-10 mm  : lime-yellow (pea hail)
            '#ffff00',  # 10-15 mm : bright yellow
            '#ffd700',  # 15-20 mm : gold
            '#ff8c00',  # 20-25 mm : dark orange
            '#ff4500',  # 25-30 mm : red-orange
            '#ff0000',  # 30-40 mm : red
            '#cc0000',  # 40-50 mm : dark red
            '#7f0000',  # 50-65 mm : deep red (baseball-size)
        ],
        'min_val': 5.0,
        'max_val': 200.0,
        'label': 'Max Estimated Hail Size (mm)',
    },
    'qpe6h': {
        'prefix': 'CONUS/RadarOnly_QPE_06H_00.00',
        'bounds': [1, 5, 10, 25, 50, 75, 100, 150, 200],  # mm accumulated
        'colors': [
            '#00fb90',  # 1-5 mm
            '#00cc00',  # 5-10 mm
            '#009900',  # 10-25 mm
            '#006600',  # 25-50 mm
            '#ffff00',  # 50-75 mm
            '#ffcc00',  # 75-100 mm
            '#ff9100',  # 100-150 mm
            '#ff0000',  # 150-200 mm
        ],
        'min_val': 1.0,
        'max_val': 1000.0,
        'label': '6-Hour QPE (mm)',
    },
    'qpe24h': {
        'prefix': 'CONUS/RadarOnly_QPE_24H_00.00',
        'bounds': [5, 25, 50, 75, 100, 150, 200, 300, 400],  # mm accumulated
        'colors': [
            '#00fb90',  # 5-25 mm
            '#00cc00',  # 25-50 mm
            '#009900',  # 50-75 mm
            '#006600',  # 75-100 mm
            '#ffff00',  # 100-150 mm
            '#ffcc00',  # 150-200 mm
            '#ff9100',  # 200-300 mm
            '#ff0000',  # 300-400 mm
        ],
        'min_val': 5.0,
        'max_val': 2000.0,
        'label': '24-Hour QPE (mm)',
    },
    'refl': {
        'prefix': 'CONUS/MergedBaseReflectivityQC_00.00',
        'bounds': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75],  # dBZ
        'colors': [
            '#646464',  # 0-5  dBZ : dark gray
            '#04e9e7',  # 5-10      : cyan
            '#019ff4',  # 10-15     : light blue
            '#0300f4',  # 15-20     : blue
            '#02fd02',  # 20-25     : bright green
            '#01c501',  # 25-30     : green
            '#008e00',  # 30-35     : dark green
            '#fdf802',  # 35-40     : yellow
            '#e5bc00',  # 40-45     : amber
            '#fd9500',  # 45-50     : orange
            '#fd0000',  # 50-55     : red
            '#d40000',  # 55-60     : dark red
            '#bc0000',  # 60-65     : darker red
            '#f800fd',  # 65-75     : magenta
        ],
        'min_val': 0.0,
        'max_val': 80.0,
        'label': 'Base Reflectivity (dBZ)',
    },
}

# --- HRRR PRECIP TYPE (model-based, avoids radar artefacts) ---
HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

_hrrr_cache = {}  # cache key: YYYYMMDDHH string

def _fetch_hrrr_vars_s3(date_str, hour_str, hours_back):
    import cfgrib  # noqa: F401

    base_url = (
        f"{HRRR_BUCKET}/hrrr.{date_str}/conus"
        f"/hrrr.t{hour_str}z.wrfsfcf00.grib2"
    )
    idx_url = base_url + ".idx"

    resp = session.get(idx_url, timeout=15)
    if resp.status_code != 200:
        raise ValueError(f"idx fetch failed: HTTP {resp.status_code}")

    target_vars = {'CRAIN', 'CSNOW', 'CICEP', 'CFRZR'}
    lines = resp.text.strip().split('\n')
    ranges = []
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) < 4:
            continue
        var_name = parts[3]
        if var_name in target_vars:
            start = int(parts[1])
            end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ''
            ranges.append((var_name, start, end))

    if not ranges:
        raise ValueError("No CRAIN/CSNOW/CICEP/CFRZR found in .idx")

    fname = f"hrrr_cat_{hours_back}.grib2"
    with open(fname, 'wb') as f_out:
        for var_name, start, end in ranges:
            rng_header = f"bytes={start}-{end}" if end != '' else f"bytes={start}-"
            r = session.get(base_url, headers={'Range': rng_header}, timeout=30)
            if r.status_code not in (200, 206):
                raise ValueError(
                    f"Range request failed for {var_name}: HTTP {r.status_code}"
                )
            f_out.write(r.content)

    return fname


def get_hrrr_precip_type(target_dt, tgt_lats_2d, tgt_lons_2d):
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None

    hrrr_hour_dt = target_dt.replace(minute=0, second=0, microsecond=0)
    cache_key = hrrr_hour_dt.strftime('%Y%m%d%H')
    if cache_key in _hrrr_cache:
        return _hrrr_cache[cache_key]

    for hours_back in range(3):
        run_dt   = hrrr_hour_dt - timedelta(hours=hours_back)
        date_str = run_dt.strftime('%Y%m%d')
        hour_str = run_dt.strftime('%H')

        fname  = None
        all_ds = []
        try:
            fname = _fetch_hrrr_vars_s3(date_str, hour_str, hours_back)

            import cfgrib
            all_ds = cfgrib.open_datasets(fname, backend_kwargs={'indexpath': ''})

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

            if not (np.any(crain) or np.any(csnow) or np.any(cicep) or np.any(cfrzr)):
                continue

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
            if fname and os.path.exists(fname):
                os.remove(fname)

            print(f"  HRRR precip type (S3): {hour_str}z (t-{hours_back}h)")
            _hrrr_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"  HRRR S3 t-{hours_back}h failed: {e}")
            for ds in all_ds:
                try:
                    ds.close()
                except Exception:
                    pass
            if fname and os.path.exists(fname):
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

def _get_mercator_grid():
    """Return (width_px, height_px, target_lats, target_lons) for the standard grid."""
    res_scale = 100
    width_px  = int((LON_RIGHT - LON_LEFT) * res_scale)
    merc_top  = lat_to_merc(LAT_TOP)
    merc_bot  = lat_to_merc(LAT_BOT)
    merc_height_ratio = (merc_top - merc_bot) / np.radians(LON_RIGHT - LON_LEFT)
    height_px = int(width_px * merc_height_ratio)
    target_y    = np.linspace(merc_top, merc_bot, height_px)
    target_lats = merc_to_lat(target_y)
    target_lons = np.linspace(LON_LEFT, LON_RIGHT, width_px)
    return width_px, height_px, target_lats, target_lons

def _parse_valid_time(ds, fallback_key):
    """Extract valid_time from dataset, falling back to filename timestamp."""
    try:
        raw_time = ds.valid_time.values
        if isinstance(raw_time, np.ndarray):
            raw_time = raw_time.flat[0]
        return datetime.fromtimestamp(
            raw_time.astype('datetime64[s]').astype(int), tz=timezone.utc)
    except Exception:
        ts_part = fallback_key.split('_')[-1]
        return datetime.strptime(
            ts_part.split('.')[0], "%Y%m%d-%H%M%S"
        ).replace(tzinfo=timezone.utc)

def process_frame(rate_key, flag_keys):
    """Process one precipitation rate frame and save as frame 0 (master.png)."""
    timestamp_part = rate_key.split('_')[-1]
    time_prefix    = timestamp_part[:13]
    flag_key       = next((k for k in flag_keys if time_prefix in k), None)

    if not flag_key:
        print(f"  Skipping {time_prefix}: no matching flag file.")
        return

    tmp_r, tmp_f = "rate_0.grib2", "flag_0.grib2"

    try:
        if not download_and_extract(rate_key, tmp_r):
            return
        if not download_and_extract(flag_key, tmp_f):
            return

        ds_rate = xr.open_dataset(tmp_r, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds_flag = xr.open_dataset(tmp_f, engine="cfgrib", backend_kwargs={'indexpath': ''})

        for ds in [ds_rate, ds_flag]:
            ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        ds_rate = ds_rate.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
        ds_flag = ds_flag.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        width_px, height_px, target_lats, target_lons = _get_mercator_grid()

        r_warp = ds_rate[list(ds_rate.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest")
        f_warp = ds_flag[list(ds_flag.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest")

        rate_vals = r_warp.values
        flag_vals = f_warp.values

        utc_dt = _parse_valid_time(ds_rate, rate_key)
        tgt_lons_2d, tgt_lats_2d = np.meshgrid(target_lons, target_lats)

        hrrr = get_hrrr_precip_type(utc_dt, tgt_lats_2d, tgt_lons_2d)

        if hrrr is not None:
            has_precip = rate_vals > 0.1

            rain_mask = hrrr['rain'] & has_precip
            snow_mask = hrrr['snow'] & has_precip & ~hrrr['rain']
            ice_mask  = hrrr['ice']  & has_precip & ~hrrr['rain'] & ~hrrr['snow']

            hrrr_typed = hrrr['rain'] | hrrr['snow'] | hrrr['ice']
            fallback   = has_precip & ~hrrr_typed
            if np.any(fallback):
                rain_mask |= fallback & np.isin(flag_vals, [1, 2, 91, 96])
                snow_mask |= fallback & np.isin(flag_vals, [3])
                ice_mask  |= fallback & np.isin(flag_vals, [4, 5, 6])
                rain_mask |= fallback & np.isin(flag_vals, [7, 10])

            mrms_rain_hail = np.isin(flag_vals, [2]) & has_precip
            ice_mask  &= ~mrms_rain_hail
            snow_mask &= ~mrms_rain_hail
            rain_mask |=  mrms_rain_hail

            still_untyped = has_precip & ~(rain_mask | snow_mask | ice_mask)
            rain_mask |= still_untyped

        else:
            has_precip = rate_vals > 0.1
            rain_mask = np.isin(flag_vals, [1, 2, 91, 96])
            snow_mask = np.isin(flag_vals, [3])
            ice_mask  = np.isin(flag_vals, [4, 5, 6])
            untyped_with_rate = has_precip & np.isin(flag_vals, [7, 10])
            rain_mask |= untyped_with_rate

        high_rate  = rate_vals > WINTRY_RATE_MAX
        rain_mask |= (snow_mask | ice_mask) & high_rate
        snow_mask &= ~high_rate
        ice_mask  &= ~high_rate

        rain = np.where(rain_mask, rate_vals, np.nan)
        snow = np.where(snow_mask, rate_vals, np.nan)
        ice  = np.where(ice_mask,  rate_vals, np.nan)

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

        plt.savefig(os.path.join(OUTPUT_DIR, "master.png"), transparent=True, pad_inches=0)
        plt.close()

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta  = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time":   et_dt.strftime("%I:%M %p ET"),
        }
        with open(os.path.join(OUTPUT_DIR, "metadata_0.json"), "w") as f:
            json.dump(meta, f)

        print(f"  Precip rate frame 0 saved: {meta['time']} ({width_px}x{height_px})")

        ds_rate.close()
        ds_flag.close()
        gc.collect()

    except Exception as e:
        print(f"  Error on precip rate frame: {e}")
    finally:
        for f in [tmp_r, tmp_f]:
            if os.path.exists(f):
                os.remove(f)


def process_single_field_frame(key, product_key, product_cfg):
    """Download, process, and render one frame of a single-field MRMS product.

    Saves output as {product_key}_0.png and metadata_{product_key}_0.json.
    The GitHub Actions workflow shifts these to _1, _2, ... before each run,
    maintaining a rolling NUM_FRAMES archive.
    """
    tmp_file = f"tmp_{product_key}.grib2"

    try:
        if not download_and_extract(key, tmp_file):
            return

        ds = xr.open_dataset(tmp_file, engine="cfgrib", backend_kwargs={'indexpath': ''})
        ds.coords['longitude'] = ((ds.longitude + 180) % 360) - 180
        ds = ds.sortby("latitude", ascending=False).sortby("longitude", ascending=True)

        width_px, height_px, target_lats, target_lons = _get_mercator_grid()

        warped = ds[list(ds.data_vars)[0]].interp(
            latitude=target_lats, longitude=target_lons, method="nearest"
        )
        vals = warped.values.astype(float)

        # Mask fill values and below-threshold values; anything unrealistically
        # large is a fill value (MRMS uses 9.99e+20 for missing data).
        min_val = product_cfg['min_val']
        max_val = product_cfg['max_val']
        vals[(vals < min_val) | (vals > max_val)] = np.nan

        utc_dt = _parse_valid_time(ds, key)

        cmap = ListedColormap(product_cfg['colors'])
        norm = BoundaryNorm(product_cfg['bounds'], cmap.N)
        cmap.set_bad(alpha=0)

        fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
        ax  = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()

        extent    = [LON_LEFT, LON_RIGHT, LAT_BOT, LAT_TOP]
        plot_args = dict(extent=extent, origin='upper', interpolation='none', aspect='auto')

        if np.any(~np.isnan(vals)):
            ax.imshow(vals, cmap=cmap, norm=norm, **plot_args)

        img_path  = os.path.join(OUTPUT_DIR, f"{product_key}_0.png")
        meta_path = os.path.join(OUTPUT_DIR, f"metadata_{product_key}_0.json")

        plt.savefig(img_path, transparent=True, pad_inches=0)
        plt.close()

        et_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))
        meta  = {
            "bounds": [[LAT_BOT, LON_LEFT], [LAT_TOP, LON_RIGHT]],
            "time":   et_dt.strftime("%I:%M %p ET"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        print(f"  {product_key} frame 0 saved: {meta['time']} ({width_px}x{height_px})")

        ds.close()
        gc.collect()

    except Exception as e:
        print(f"  Error processing {product_key}: {e}")
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


if __name__ == "__main__":
    RATE_PREFIX = discover_rate_prefix()
    now_utc = datetime.now(timezone.utc)

    # --- Precipitation Rate: process only the single latest frame ---
    print("=== Precipitation Rate ===")
    precip_done = False
    for d in range(2):
        date_str  = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
        rate_keys = get_s3_keys(date_str, RATE_PREFIX)
        flag_keys = get_s3_keys(date_str, FLAG_PREFIX)
        if not rate_keys:
            print(f"  No rate keys for {date_str}.")
            continue
        latest_rate_key = sorted(rate_keys)[-1]
        print(f"  Latest key: {latest_rate_key}")
        process_frame(latest_rate_key, flag_keys)
        precip_done = True
        break
    if not precip_done:
        print("  No precipitation rate data found.")

    # --- Single-field products: process only the single latest frame each ---
    for product_key, product_cfg in SINGLE_PRODUCTS.items():
        print(f"\n=== {product_cfg['label']} ===")
        product_done = False
        for d in range(2):
            date_str = (now_utc - timedelta(days=d)).strftime("%Y%m%d")
            keys     = get_s3_keys(date_str, product_cfg['prefix'])
            if not keys:
                print(f"  No keys for {date_str}.")
                continue
            latest_key = sorted(keys)[-1]
            print(f"  Latest key: {latest_key}")
            process_single_field_frame(latest_key, product_key, product_cfg)
            product_done = True
            break
        if not product_done:
            print(f"  No data found for {product_key}.")
