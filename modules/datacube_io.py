"""
datacube_io.py — File discovery, lazy dataset loading, pixel extraction,
and coordinate reprojection for the VI Phenology Dashboard.

Supports both local paths and Google Cloud Storage (gs://bucket/path).

Memory contract
---------------
- The full datacube array is NEVER loaded into memory.
- Basemap: computed via Dask lazy reduce + spatial coarsening.
- Pixel time series: direct HDF5 hyperslab read (local) or xarray isel (GCS).
- Dataset handles: lru_cache so files are opened once per session.

GCS notes
---------
- Set VI_DATACUBE_ROOT=gs://your-bucket/path in your environment.
- Authenticate via Application Default Credentials (gcloud auth application-default login)
  or a service-account key (GOOGLE_APPLICATION_CREDENTIALS env var).
- ZARR format is strongly recommended on GCS — pixel reads require only 1-4 chunks
  vs streaming an entire NetCDF file.  Use tools/convert_to_zarr.py locally, then
  upload the .zarr directory to GCS.
- Install required extras: pip install gcsfs h5netcdf
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATACUBE_ROOT,
    GCS_PROJECT,
    DEFAULT_VI_VAR,
    DATACUBE_CRS_EPSG,
    TARGET_CRS_EPSG,
    BASEMAP_MAX_DIM,
    BASEMAP_MAX_DIM_PRECOMPUTED,
    VI_VALID_RANGE,
)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _is_gcs(path: str) -> bool:
    """Return True if path is a Google Cloud Storage URI (gs://)."""
    return str(path).startswith("gs://")


def _gcs_token():
    """
    Resolve the GCS authentication token from environment variables.

    Priority order:
    1. GOOGLE_SERVICE_ACCOUNT_JSON — JSON key content pasted as an env var.
       Recommended for Plotly Cloud / any non-GCP host with a private bucket.
    2. GCS_TOKEN=anon — use anonymous access for publicly readable buckets.
    3. GCS_TOKEN=google_default (or unset) — Application Default Credentials.
       Works on GCP (Cloud Run, GKE with Workload Identity, local gcloud auth).
    """
    import json as _json
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        return _json.loads(sa_json)
    token = os.environ.get("GCS_TOKEN", "google_default")
    return token


@lru_cache(maxsize=1)
def _gcs_fs():
    """Return a cached gcsfs.GCSFileSystem instance."""
    import warnings
    try:
        import gcsfs
    except ImportError:
        raise ImportError(
            "gcsfs is required for GCS support.  Install with: pip install gcsfs"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "GCS project not set", UserWarning)
        return gcsfs.GCSFileSystem(token=_gcs_token(), project=GCS_PROJECT)


def _gcs_storage_options() -> dict:
    """Return fsspec storage_options dict for GCS (used by xarray/zarr)."""
    opts: dict = {"token": _gcs_token()}
    if GCS_PROJECT:
        opts["project"] = GCS_PROJECT
    return opts


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegionPaths:
    """Paths (local or gs:// URIs) and VI variable name for one datacube region."""
    region_id: str
    nc_path: str | None        # None when only a Zarr store is available
    zarr_path: str | None      # None if not yet converted
    metrics_path: str | None   # None if pixel_metrics.nc not present
    vi_var: str = DEFAULT_VI_VAR


class PixelTimeSeries(NamedTuple):
    """Raw time series data for a single pixel."""
    dates: np.ndarray       # datetime64[D], shape (n_time,)
    raw_vi: np.ndarray      # float32, shape (n_time,); NaN where masked
    valid_mask: np.ndarray  # bool, shape (n_time,)
    x_coord: float          # UTM easting (m)
    y_coord: float          # UTM northing (m)
    lon: float              # WGS84 longitude
    lat: float              # WGS84 latitude


# ---------------------------------------------------------------------------
# Region discovery
# ---------------------------------------------------------------------------

def _parse_nc_stem(stem: str) -> tuple[str, str]:
    """
    Extract (region_id, vi_var) from a NetCDF filename stem.

    Handles two common naming conventions (and anything in between):

        T34HDG_NDVI          →  ("T34HDG", "NDVI")   — trailing VI suffix
        NDVI_G5_1_datacube   →  ("G5_1",   "NDVI")   — leading VI prefix
        SomeName_datacube    →  ("SomeName", DEFAULT_VI_VAR)
        Anything             →  (stem,      DEFAULT_VI_VAR)

    Known VI names are taken from VI_VALID_RANGE in config.py.
    The optional "_datacube" suffix is stripped before matching.
    """
    known_vis: tuple[str, ...] = tuple(VI_VALID_RANGE.keys())

    # Strip optional "_datacube" suffix
    base = stem[: -len("_datacube")] if stem.endswith("_datacube") else stem

    # Trailing _{VI} suffix  (e.g. T34HDG_NDVI)
    for vi in known_vis:
        if base.endswith(f"_{vi}"):
            return base[: -len(f"_{vi}")], vi

    # Leading {VI}_ prefix  (e.g. NDVI_G5_1)
    for vi in known_vis:
        if base.startswith(f"{vi}_"):
            return base[len(f"{vi}_") :], vi

    # No VI found in name
    return base, DEFAULT_VI_VAR


def _natural_sort_key(name: str):
    """Sort 'G5_10' after 'G5_9' (numeric-aware)."""
    import re
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def discover_regions(root: str = DATACUBE_ROOT) -> dict[str, RegionPaths]:
    """
    Recursively scan the datacube root for NetCDF datacubes (*.nc).

    Supports both local filesystem paths and GCS URIs (gs://bucket/path).
    See module docstring for GCS setup instructions.

    No assumptions are made about subdirectory names, nesting depth,
    or filename conventions.  For each datacube found:

    - Region ID and VI variable are parsed from the filename stem via
      _parse_nc_stem().  Examples:
          T34HDG_NDVI.nc         →  region "T34HDG",  vi "NDVI"
          NDVI_G5_1_datacube.nc  →  region "G5_1",    vi "NDVI"
    - Files whose stem contains "pixel_metrics" are skipped (those are
      companion output files, not input datacubes).
    - Companion files are looked up in the same directory as the .nc:
          <nc_stem>.zarr/                  (optional — fast pixel reads)
          <nc_stem>_pixel_metrics.nc       (optional — precomputed metrics)
    - If two files in different directories produce the same region ID,
      the parent directory name is prepended ("parent/region").

    Returns a dict keyed by region_id, sorted naturally.
    Raises FileNotFoundError if root does not exist or no datacubes are found.
    """
    if _is_gcs(root):
        return _discover_regions_gcs(root)
    return _discover_regions_local(root)


def _discover_regions_local(root: str) -> dict[str, RegionPaths]:
    """Local filesystem implementation of region discovery."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Datacube directory not found: {root}\n"
            f"Set VI_DATACUBE_ROOT env var or edit config.py."
        )

    nc_files = sorted(
        (p for p in root_path.rglob("*.nc") if "pixel_metrics" not in p.stem),
        key=lambda p: _natural_sort_key(p.name),
    )

    if not nc_files:
        raise FileNotFoundError(
            f"No *.nc datacube files found under: {root}\n"
            f"Set VI_DATACUBE_ROOT env var or edit config.py."
        )

    # First pass: parse candidate region IDs and VI variable names
    candidates: list[tuple[str, str, Path]] = []
    for nc_path in nc_files:
        region_id, vi_var = _parse_nc_stem(nc_path.stem)
        candidates.append((region_id, vi_var, nc_path))

    # Detect collisions and qualify with parent directory name
    seen: dict[str, int] = {}
    for rid, _, _ in candidates:
        seen[rid] = seen.get(rid, 0) + 1

    regions: dict[str, RegionPaths] = {}
    for region_id, vi_var, nc_path in candidates:
        if seen[region_id] > 1:
            region_id = f"{nc_path.parent.name}/{region_id}"

        parent = nc_path.parent
        zarr_path    = parent / (nc_path.stem + ".zarr")
        # pixel_phenology_extract.py writes NDVI_<region_id>_pixel_metrics.nc;
        # fall back to <nc_stem>_pixel_metrics.nc for any alternative naming.
        _metrics_canonical = parent / f"{vi_var}_{region_id}_pixel_metrics.nc"
        _metrics_fallback  = parent / (nc_path.stem + "_pixel_metrics.nc")
        metrics_path = (
            _metrics_canonical if _metrics_canonical.exists()
            else _metrics_fallback
        )

        regions[region_id] = RegionPaths(
            region_id=region_id,
            nc_path=str(nc_path),
            zarr_path=str(zarr_path) if zarr_path.exists() else None,
            metrics_path=str(metrics_path) if metrics_path.exists() else None,
            vi_var=vi_var,
        )

    return regions


def _discover_regions_gcs(root: str) -> dict[str, RegionPaths]:
    """
    GCS implementation of region discovery using gcsfs.

    Discovers regions from Zarr stores (*.zarr/.zmetadata) and/or NetCDF
    datacubes (*.nc, excluding pixel_metrics files).  Either is sufficient
    — Zarr is preferred for performance, NC is the fallback.
    """
    fs = _gcs_fs()
    root_no_scheme = root.removeprefix("gs://").rstrip("/")

    # --- Collect Zarr stores ---
    # Zarr v3 stores have zarr.json at the root.
    # Zarr v2 stores have .zmetadata (consolidated) or .zgroup (always present).
    # Try all three markers; stop as soon as any are found.
    zarr_rels: list[str] = []
    for marker in ("zarr.json", ".zmetadata", ".zgroup"):
        raw: list[str] = fs.glob(f"{root_no_scheme}/**/{marker}")
        found = [m[: -len(f"/{marker}")] for m in raw
                 if m.endswith(f"/{marker}")
                 and m[: -len(f"/{marker}")].endswith(".zarr")]
        zarr_rels = list(dict.fromkeys(zarr_rels + found))  # dedup, preserve order
        if zarr_rels:
            break

    # --- Collect NetCDF datacubes (optional, used when no Zarr present) ---
    all_nc: list[str] = [
        p for p in fs.glob(f"{root_no_scheme}/**/*.nc")
        if "pixel_metrics" not in Path(p).stem
    ]

    if not zarr_rels and not all_nc:
        raise FileNotFoundError(
            f"No Zarr stores or *.nc datacube files found under: {root}\n"
            f"Set VI_DATACUBE_ROOT env var or edit config.py."
        )

    # Build a lookup: stem → nc_rel_path (for pairing with zarr stores)
    nc_by_stem: dict[str, str] = {Path(p).stem: p for p in all_nc}

    # Build candidate list from zarr stores first, then any unpaired nc files
    # Each candidate: (region_id, vi_var, zarr_rel_or_None, nc_rel_or_None)
    Candidate = tuple[str, str, str | None, str | None]
    candidates: list[Candidate] = []
    seen_stems: set[str] = set()

    for zarr_rel in sorted(zarr_rels, key=lambda p: _natural_sort_key(Path(p).name)):
        stem = Path(zarr_rel).stem  # e.g. "NDVI_G5_1_datacube"
        region_id, vi_var = _parse_nc_stem(stem)
        nc_rel = nc_by_stem.get(stem)
        candidates.append((region_id, vi_var, zarr_rel, nc_rel))
        seen_stems.add(stem)

    for nc_rel in sorted(all_nc, key=lambda p: _natural_sort_key(Path(p).name)):
        stem = Path(nc_rel).stem
        if stem in seen_stems:
            continue  # already covered by the zarr entry above
        region_id, vi_var = _parse_nc_stem(stem)
        candidates.append((region_id, vi_var, None, nc_rel))
        seen_stems.add(stem)

    # Detect collisions and qualify with parent directory name
    seen_ids: dict[str, int] = {}
    for rid, _, _, _ in candidates:
        seen_ids[rid] = seen_ids.get(rid, 0) + 1

    regions: dict[str, RegionPaths] = {}
    for region_id, vi_var, zarr_rel, nc_rel in candidates:
        src_rel   = zarr_rel or nc_rel
        parent_rel = str(Path(src_rel).parent)

        if seen_ids[region_id] > 1:
            region_id = f"{Path(src_rel).parent.name}/{region_id}"

        nc_uri   = f"gs://{nc_rel}"   if nc_rel   else None
        zarr_uri = f"gs://{zarr_rel}" if zarr_rel  else None

        # Metrics: look for <VI>_<region>_pixel_metrics.nc next to the data
        metrics_canonical_rel = f"{parent_rel}/{vi_var}_{region_id}_pixel_metrics.nc"
        metrics_fallback_rel  = f"{parent_rel}/{Path(src_rel).stem}_pixel_metrics.nc"
        if fs.exists(metrics_canonical_rel):
            metrics_uri: str | None = f"gs://{metrics_canonical_rel}"
        elif fs.exists(metrics_fallback_rel):
            metrics_uri = f"gs://{metrics_fallback_rel}"
        else:
            metrics_uri = None

        regions[region_id] = RegionPaths(
            region_id=region_id,
            nc_path=nc_uri,
            zarr_path=zarr_uri,
            metrics_path=metrics_uri,
            vi_var=vi_var,
        )

    return dict(sorted(regions.items(), key=lambda x: _natural_sort_key(x[0])))


# ---------------------------------------------------------------------------
# Dataset handles — lru_cache so files are opened only once per session
# ---------------------------------------------------------------------------

@lru_cache(maxsize=18)
def _open_datacube_cached(nc_path_str: str) -> xr.Dataset:
    """
    Open a NetCDF4 datacube with xarray + Dask using the file's native
    HDF5 chunk layout (chunks={}).

    For local files, uses engine="netcdf4".
    For GCS URIs (gs://), uses engine="h5netcdf" with gcsfs storage_options.

    String argument (not Path) required for lru_cache hashability.
    maxsize=18 covers all current LVIS regions so handles are never evicted.
    """
    if _is_gcs(nc_path_str):
        return xr.open_dataset(
            nc_path_str,
            engine="h5netcdf",
            chunks={},
            mask_and_scale=True,
            storage_options=_gcs_storage_options(),
        )
    return xr.open_dataset(
        nc_path_str,
        engine="netcdf4",
        chunks={},
        mask_and_scale=True,
    )


@lru_cache(maxsize=18)
def _open_zarr_cached(zarr_path_str: str) -> xr.Dataset:
    """
    Open a Zarr store with xarray (lazy).  Cached so the store is opened
    only once per session per region, and reused for both basemap display
    and fast pixel extraction.

    Supports both local paths and GCS URIs (gs://).

    String argument (not Path) required for lru_cache hashability.
    """
    if _is_gcs(zarr_path_str):
        return xr.open_zarr(
            zarr_path_str,
            storage_options=_gcs_storage_options(),
        )
    return xr.open_zarr(zarr_path_str)


def get_dataset(paths: RegionPaths) -> xr.Dataset:
    """
    Return a lazily opened dataset for the region.
    Prefers ZARR (faster pixel access) when available.
    """
    if paths.zarr_path is not None:
        return _open_zarr_cached(paths.zarr_path)
    if paths.nc_path is not None:
        return _open_datacube_cached(paths.nc_path)
    raise FileNotFoundError(
        f"Region '{paths.region_id}' has neither a Zarr store nor a NetCDF file."
    )


# ---------------------------------------------------------------------------
# Coordinate reprojection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _get_transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True
    )


def utm_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    src_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert UTM easting/northing arrays to WGS84 (lon, lat)."""
    tf = _get_transformer(src_epsg, TARGET_CRS_EPSG)
    lon, lat = tf.transform(x, y)
    return lon, lat


def latlon_to_utm(
    lon: float,
    lat: float,
    dst_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[float, float]:
    """Convert WGS84 lon/lat to UTM easting/northing."""
    tf = _get_transformer(TARGET_CRS_EPSG, dst_epsg)
    x, y = tf.transform(lon, lat)
    return float(x), float(y)


def detect_crs_epsg(ds: xr.Dataset) -> int:
    """
    Read the EPSG code from the CF grid_mapping 'spatial_ref' variable.
    Falls back to DATACUBE_CRS_EPSG if not present or unparseable.
    """
    try:
        from pyproj import CRS
        sr = ds["spatial_ref"]
        wkt = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref", "")
        if wkt:
            epsg = CRS.from_wkt(wkt).to_epsg()
            if epsg:
                return int(epsg)
    except Exception:
        pass
    return DATACUBE_CRS_EPSG


def build_display_coords(
    ds: xr.Dataset,
    src_epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lon/lat 2-D arrays from the dataset's x/y coordinate vectors.
    Used to label Plotly heatmap axes in geographic coordinates.

    Returns (lon_2d, lat_2d), each shape (ny, nx).
    Only reads coordinate arrays — never the data variable.
    """
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)
    x_vals = ds["x"].values  # (nx,)
    y_vals = ds["y"].values  # (ny,)
    xx, yy = np.meshgrid(x_vals, y_vals)
    lon, lat = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    return lon.reshape(yy.shape), lat.reshape(yy.shape)


# ---------------------------------------------------------------------------
# Date cache — shared structure for all pixels in a datacube
# ---------------------------------------------------------------------------

def build_date_cache(ds: xr.Dataset) -> dict:
    """
    Build the date_cache dict required by _extract_pixel_metrics().
    Reads only the time coordinate — never the data array.

    Returns a dict with keys:
        n_days      : int — calendar days from first to last observation
        year_arr    : np.int32 (n_days,) — year of each day on the daily grid
        doy_arr     : np.int16 (n_days,) — day-of-year on the daily grid
        years       : np.ndarray — unique years present
        year_masks  : dict[int, np.ndarray] — boolean masks per year
        day_offsets : np.int32 (n_time,) — index of each obs on the daily grid
    """
    # Decode time from "days since 1970-01-01" to pd.DatetimeIndex
    time_raw = ds["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        times = pd.DatetimeIndex(pd.to_datetime(time_raw))
    else:
        # int32 days since 1970-01-01
        origin = pd.Timestamp("1970-01-01")
        times = pd.DatetimeIndex(
            [origin + pd.Timedelta(days=int(d)) for d in time_raw]
        )

    n_days = (times[-1] - times[0]).days + 1
    all_dates = pd.date_range(start=times[0], periods=n_days, freq="D")
    year_arr  = all_dates.year.values.astype(np.int32)
    doy_arr   = all_dates.dayofyear.values.astype(np.int16)
    years     = np.unique(year_arr)

    day_offsets = np.array(
        [(t - times[0]).days for t in times],
        dtype=np.int32,
    )

    return {
        "n_days":      n_days,
        "year_arr":    year_arr,
        "doy_arr":     doy_arr,
        "years":       years,
        "year_masks":  {int(yr): (year_arr == yr) for yr in years},
        "day_offsets": day_offsets,
    }


def build_date_cache_from_dates(obs_dates: np.ndarray) -> dict:
    """
    Build a date_cache dict from a datetime64[D] array of observation dates.
    Equivalent to build_date_cache() but takes a date array instead of a Dataset.
    Used when the active date range has been clipped (e.g. by the year-range slider).
    """
    times = pd.DatetimeIndex(obs_dates.astype("datetime64[ns]"))
    n_days = (times[-1] - times[0]).days + 1
    all_dates = pd.date_range(start=times[0], periods=n_days, freq="D")
    year_arr  = all_dates.year.values.astype(np.int32)
    doy_arr   = all_dates.dayofyear.values.astype(np.int16)
    years     = np.unique(year_arr)
    day_offsets = np.array(
        [(t - times[0]).days for t in times],
        dtype=np.int32,
    )
    return {
        "n_days":      n_days,
        "year_arr":    year_arr,
        "doy_arr":     doy_arr,
        "years":       years,
        "year_masks":  {int(yr): (year_arr == yr) for yr in years},
        "day_offsets": day_offsets,
    }


# ---------------------------------------------------------------------------
# Web Mercator regridding helper
# ---------------------------------------------------------------------------

def _regrid_to_mercator(
    z: np.ndarray,
    lon_2d: np.ndarray,
    lat_2d: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    src_epsg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-project z values from a regular UTM grid onto a regular Web Mercator
    (EPSG:3857) grid, returning WGS84 lon/lat equivalents of each cell centre.

    Why Mercator instead of WGS84
    ------------------------------
    Leaflet renders dl.ImageOverlay by converting the WGS84 corner bounds to
    EPSG:3857 and stretching the PNG linearly in Mercator space.  Generating
    the PNG on a regular Mercator grid makes that stretch geometrically exact
    in both axes.  A plain WGS84 grid cannot account for the ~0.5–1° rotation
    of UTM Zone 34S relative to geographic north, which causes 50–200 m pixel
    drift at flight-box edges.

    Return value
    ------------
    z_reg     : reprojected metric values on the Mercator grid
    lon_grid  : WGS84 longitude of each grid cell (for bounds calculation)
    lat_grid  : WGS84 latitude  of each grid cell (for bounds calculation)
    """
    from scipy.interpolate import RegularGridInterpolator

    ny, nx = z.shape

    # --- build interpolator on the UTM source grid ---
    # RegularGridInterpolator requires strictly increasing axis arrays;
    # y_c (UTM northing) is often decreasing in raster convention.
    y_order = np.argsort(y_c)
    x_order = np.argsort(x_c)
    z_sorted = z[y_order, :][:, x_order]

    interp = RegularGridInterpolator(
        (y_c[y_order], x_c[x_order]),
        z_sorted,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    # --- WGS84 data extent → Web Mercator extent ---
    tf_wgs_to_merc = _get_transformer(TARGET_CRS_EPSG, 3857)
    lon_min, lon_max = float(lon_2d.min()), float(lon_2d.max())
    lat_min, lat_max = float(lat_2d.min()), float(lat_2d.max())
    # Use all four corners to capture any skew in the UTM→WGS84 conversion.
    cx, cy = tf_wgs_to_merc.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max],
    )
    merc_x_min, merc_x_max = float(np.min(cx)), float(np.max(cx))
    merc_y_min, merc_y_max = float(np.min(cy)), float(np.max(cy))

    # --- regular Web Mercator target grid ---
    merc_x = np.linspace(merc_x_min, merc_x_max, nx)
    merc_y = np.linspace(merc_y_min, merc_y_max, ny)
    merc_x_grid, merc_y_grid = np.meshgrid(merc_x, merc_y)

    # --- Mercator → WGS84 (for the return value / ImageOverlay bounds) ---
    tf_merc_to_wgs = _get_transformer(3857, TARGET_CRS_EPSG)
    lon_flat, lat_flat = tf_merc_to_wgs.transform(
        merc_x_grid.ravel(), merc_y_grid.ravel()
    )
    lon_grid = lon_flat.reshape(ny, nx)
    lat_grid = lat_flat.reshape(ny, nx)

    # --- Mercator → UTM (for interpolator lookup) ---
    tf_merc_to_utm = _get_transformer(3857, src_epsg)
    x_tgt, y_tgt = tf_merc_to_utm.transform(
        merc_x_grid.ravel(), merc_y_grid.ravel()
    )
    z_reg = interp(
        np.column_stack([y_tgt, x_tgt])  # interpolator axis order: (northing, easting)
    ).reshape(ny, nx)

    return z_reg, lon_grid, lat_grid


# ---------------------------------------------------------------------------
# Basemap disk cache helpers
# ---------------------------------------------------------------------------

def basemap_cache_path(data_path: str, metric: str, max_dim: int) -> str:
    """
    Return the path (local) or URI (GCS) of the on-disk basemap cache file.
    Accepts either an nc_path or a zarr_path (stem is used for naming either way).

    Naming: <stem>_basemap_<metric>_d<max_dim>.npz
    e.g.    NDVI_G5_14_datacube_basemap_mean_ndvi_d500.npz
    """
    if _is_gcs(data_path):
        # Strip trailing slash, then take everything after the last /
        name   = data_path.rstrip("/").rsplit("/", 1)[-1]
        stem   = name.rsplit(".", 1)[0]  # removes .nc or .zarr
        parent = data_path.rstrip("/").rsplit("/", 1)[0]
        return f"{parent}/{stem}_basemap_{metric}_d{max_dim}.npz"
    p = Path(data_path.rstrip("/"))
    return str(p.parent / f"{p.stem}_basemap_{metric}_d{max_dim}.npz")


def load_basemap_cache(
    cache_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load a cached (z, lon, lat) triple from disk or GCS.
    Returns None if the file does not exist or cannot be read.
    """
    try:
        if _is_gcs(cache_path):
            fs = _gcs_fs()
            cache_rel = cache_path.removeprefix("gs://")
            with fs.open(cache_rel, "rb") as f:
                data = np.load(f)
                return data["z"], data["lon"], data["lat"]
        data = np.load(cache_path)
        return data["z"], data["lon"], data["lat"]
    except Exception:
        return None


def save_basemap_cache(
    cache_path: str,
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:
    """
    Persist (z, lon, lat) to a compressed .npz file on disk or GCS.
    Silently ignores write errors (e.g. read-only volume or missing permissions).
    """
    try:
        if _is_gcs(cache_path):
            fs = _gcs_fs()
            cache_rel = cache_path.removeprefix("gs://")
            buf = io.BytesIO()
            np.savez_compressed(buf, z=z, lon=lon, lat=lat)
            buf.seek(0)
            with fs.open(cache_rel, "wb") as f:
                f.write(buf.read())
        else:
            np.savez_compressed(cache_path, z=z, lon=lon, lat=lat)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Basemap spatial metrics — lazy Dask compute, downsampled
# ---------------------------------------------------------------------------

def compute_basemap_metric(
    ds: xr.Dataset,
    metric: str,
    vi_var: str = DEFAULT_VI_VAR,
    max_dim: int = BASEMAP_MAX_DIM,
    src_epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple spatial summary metric using Dask and return downsampled
    arrays suitable for Plotly heatmap display.

    The full data array is NEVER loaded into memory — Dask reads only the
    chunks needed for the spatial coarsening + reduce operation.

    Supported metric keys:
        "peak_ndvi_mean"  → max over time axis
        "mean_ndvi"       → mean over time axis
        "std_ndvi"        → std  over time axis
        "data_coverage"   → fraction of non-NaN observations

    Returns (z_2d, lon_2d, lat_2d) numpy arrays at display resolution.
    """
    da = ds[vi_var]  # Dask-backed DataArray (time, y, x)
    ny, nx = da.sizes["y"], da.sizes["x"]

    # Rechunk time to -1 so each spatial tile holds the full time series.
    # This makes the subsequent temporal max/mean a single in-memory reduce
    # per tile rather than a multi-chunk aggregation.
    da = da.chunk({"time": -1})

    # Compute coarsening factors to stay within max_dim × max_dim.
    # Equalize so both axes use the same factor → square display pixels
    # (unequal factors produce elongated rectangles for tall/narrow regions).
    cf_y = max(1, ny // max_dim)
    cf_x = max(1, nx // max_dim)
    cf_y = cf_x = max(cf_y, cf_x)

    # Coarsen spatially first (still lazy), then reduce over time
    da_c = da.coarsen(y=cf_y, x=cf_x, boundary="trim").mean()

    # scheduler="synchronous" avoids thread-pool/SIGALRM conflicts with gunicorn
    # sync workers: the threaded scheduler spawns threads that can be killed
    # mid-lock when gunicorn's timeout signal fires, causing the
    # "SystemExit inside threading.Condition.wait" crash seen in production.
    if metric == "peak_ndvi_mean":
        z = da_c.max(dim="time").compute(scheduler="synchronous").values
    elif metric == "mean_ndvi":
        z = da_c.mean(dim="time").compute(scheduler="synchronous").values
    elif metric == "std_ndvi":
        z = da_c.std(dim="time").compute(scheduler="synchronous").values
    elif metric == "data_coverage":
        z = da_c.notnull().mean(dim="time").compute(scheduler="synchronous").values
        z[z == 0.0] = np.nan  # pixels with no valid obs → transparent
    else:
        raise ValueError(f"Unknown on-the-fly basemap metric: {metric!r}")

    # Build WGS84 coordinates and reproject onto a regular grid so that
    # go.Heatmap axes (lon[0,:] / lat[:,0]) are truly axis-aligned.
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)
    x_c = da_c["x"].values
    y_c = da_c["y"].values
    xx, yy = np.meshgrid(x_c, y_c)
    lon_2d, lat_2d = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    lon_2d = lon_2d.reshape(yy.shape)
    lat_2d = lat_2d.reshape(yy.shape)

    return _regrid_to_mercator(z, lon_2d, lat_2d, x_c, y_c, src_epsg)


def load_metrics_for_basemap(
    metrics_path: str,
    metric: str,
    max_dim: int = BASEMAP_MAX_DIM_PRECOMPUTED,
    src_epsg: int = DATACUBE_CRS_EPSG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast path: load one 2-D metric slice from a precomputed pixel_metrics.nc.
    Returns (z_2d, lon_2d, lat_2d) at display resolution.

    Supports both local paths and GCS URIs.
    """
    open_kwargs: dict = {"mask_and_scale": True, "decode_timedelta": False}
    if _is_gcs(metrics_path):
        open_kwargs["engine"] = "h5netcdf"
        open_kwargs["storage_options"] = _gcs_storage_options()

    with xr.open_dataset(metrics_path, **open_kwargs) as ds_m:
        if metric not in ds_m:
            raise KeyError(
                f"Metric {metric!r} not found in {metrics_path}. "
                f"Available: {list(ds_m.data_vars)}"
            )
        da = ds_m[metric]
        ny, nx = da.sizes["y"], da.sizes["x"]
        cf_y = max(1, ny // max_dim)
        cf_x = max(1, nx // max_dim)
        cf_y = cf_x = max(cf_y, cf_x)   # equalize → square display pixels
        if cf_y > 1:
            da = da.coarsen(y=cf_y, x=cf_x, boundary="trim").mean()
        z = da.values
        x_c = da["x"].values
        y_c = da["y"].values

    xx, yy = np.meshgrid(x_c, y_c)
    lon_2d, lat_2d = utm_to_latlon(xx.ravel(), yy.ravel(), src_epsg)
    lon_2d = lon_2d.reshape(yy.shape)
    lat_2d = lat_2d.reshape(yy.shape)
    return _regrid_to_mercator(z, lon_2d, lat_2d, x_c, y_c, src_epsg)


# ---------------------------------------------------------------------------
# Single-pixel time series — direct HDF5 hyperslab read OR Zarr isel
# ---------------------------------------------------------------------------

def _extract_pixel_timeseries_zarr(
    zarr_path: str,
    yi: int,
    xi: int,
    vi_var: str = DEFAULT_VI_VAR,
) -> PixelTimeSeries:
    """
    Fast pixel extraction from a Zarr store (local or GCS).

    With chunks [time=-1, y=10, x=10] exactly one Zarr chunk is read
    (worst case 4 chunks at a spatial chunk boundary), vs decompressing the
    entire NetCDF file with the HDF5 hyperslab approach.
    """
    ds = _open_zarr_cached(zarr_path)

    # Pull only the pixel's time series (triggers single-chunk read)
    pixel_da = ds[vi_var].isel(y=yi, x=xi)
    vi_arr = pixel_da.values.astype(np.float32)

    # Decode time coordinate
    time_raw = ds["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        times = pd.DatetimeIndex(pd.to_datetime(time_raw))
        dates = np.array(times, dtype="datetime64[D]")
    else:
        origin = np.datetime64("1970-01-01", "D")
        dates = origin + time_raw.astype("timedelta64[D]")

    # Replace fill values with NaN
    fill = ds[vi_var].encoding.get("_FillValue", None)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        vi_arr[vi_arr == fill] = np.nan

    x_val = float(ds["x"].values[xi])
    y_val = float(ds["y"].values[yi])

    vi_min, vi_max = VI_VALID_RANGE.get(vi_var, (-1.0, 2.0))
    valid_mask = (
        ~np.isnan(vi_arr)
        & (vi_arr >= vi_min)
        & (vi_arr <= vi_max)
    )

    lon_arr, lat_arr = utm_to_latlon(np.array([x_val]), np.array([y_val]))

    return PixelTimeSeries(
        dates=dates,
        raw_vi=vi_arr,
        valid_mask=valid_mask,
        x_coord=x_val,
        y_coord=y_val,
        lon=float(lon_arr[0]),
        lat=float(lat_arr[0]),
    )


def _extract_pixel_timeseries_xarray(
    nc_path: str,
    yi: int,
    xi: int,
    vi_var: str = DEFAULT_VI_VAR,
) -> PixelTimeSeries:
    """
    Pixel extraction via xarray isel — used for GCS NetCDF files where
    the netCDF4 library cannot open remote paths directly.

    Note: this triggers a Dask compute for the single pixel's time series.
    Converting to ZARR (tools/convert_to_zarr.py) is strongly recommended
    for better GCS performance.
    """
    ds = _open_datacube_cached(nc_path)
    pixel_da = ds[vi_var].isel(y=yi, x=xi)
    vi_arr = pixel_da.compute().values.astype(np.float32)

    time_raw = ds["time"].values
    if np.issubdtype(time_raw.dtype, np.datetime64):
        times = pd.DatetimeIndex(pd.to_datetime(time_raw))
        dates = np.array(times, dtype="datetime64[D]")
    else:
        origin = np.datetime64("1970-01-01", "D")
        dates = origin + time_raw.astype("timedelta64[D]")

    fill = ds[vi_var].encoding.get("_FillValue", None)
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        vi_arr[vi_arr == fill] = np.nan

    x_val = float(ds["x"].values[xi])
    y_val = float(ds["y"].values[yi])

    vi_min, vi_max = VI_VALID_RANGE.get(vi_var, (-1.0, 2.0))
    valid_mask = (
        ~np.isnan(vi_arr)
        & (vi_arr >= vi_min)
        & (vi_arr <= vi_max)
    )

    lon_arr, lat_arr = utm_to_latlon(np.array([x_val]), np.array([y_val]))

    return PixelTimeSeries(
        dates=dates,
        raw_vi=vi_arr,
        valid_mask=valid_mask,
        x_coord=x_val,
        y_coord=y_val,
        lon=float(lon_arr[0]),
        lat=float(lat_arr[0]),
    )


def extract_pixel_timeseries(
    nc_path: str,
    yi: int,
    xi: int,
    vi_var: str = DEFAULT_VI_VAR,
    zarr_path: str | None = None,
) -> PixelTimeSeries:
    """
    Extract the full time series for pixel (yi, xi).

    Fast path (preferred): when zarr_path is provided, uses xarray isel on
    the Zarr store.  With Zarr chunks [time=-1, y=10, x=10] this reads exactly
    one chunk (≈ 580 KB) regardless of spatial extent — vs decompressing the
    entire NetCDF file with the HDF5 hyperslab path.

    GCS fallback: when nc_path is a GCS URI and no zarr_path is provided,
    uses xarray isel via h5netcdf.  This is slower than ZARR — converting
    your datacubes to ZARR (tools/convert_to_zarr.py) is strongly recommended.

    Local fallback: direct netCDF4 hyperslab read (reads only T floats but must
    decompress every spatial chunk due to the [1, full_y, full_x] layout).

    Parameters
    ----------
    nc_path   : local path or gs:// URI to the .nc datacube file
    yi, xi    : zero-based array indices (row = y direction, col = x direction)
    vi_var    : name of the VI variable in the file
    zarr_path : optional local path or gs:// URI to the companion .zarr store

    Returns
    -------
    PixelTimeSeries namedtuple
    """
    if zarr_path is not None:
        return _extract_pixel_timeseries_zarr(zarr_path, yi, xi, vi_var)
    if _is_gcs(nc_path):
        return _extract_pixel_timeseries_xarray(nc_path, yi, xi, vi_var)
    # Local NetCDF: use direct HDF5 hyperslab read (fast)
    with nc4.Dataset(nc_path, mode="r") as ds:
        # Decode time to datetime64
        time_var = ds.variables["time"]
        time_vals = np.array(time_var[:])
        units = getattr(time_var, "units", "days since 1970-01-01")

        if "days since" in units.lower():
            origin_str = units.lower().replace("days since", "").strip().split()[0]
            origin = np.datetime64(origin_str, "D")
            dates = origin + time_vals.astype("timedelta64[D]")
        else:
            # Fall back to netCDF4 num2date
            import cftime
            dts = nc4.num2date(time_vals, units, only_use_cftime_datetimes=False)
            dates = np.array(
                [np.datetime64(str(d)[:10], "D") for d in dts]
            )

        # Hyperslab read — only the pixel's time series
        vi_data = ds.variables[vi_var][:, yi, xi]
        vi_arr = np.array(vi_data, dtype=np.float32)

        # Mask fill value → NaN
        fill = getattr(ds.variables[vi_var], "_FillValue", None)
        if fill is not None and not np.isnan(fill):
            vi_arr[vi_arr == fill] = np.nan

        # Spatial coordinates
        x_val = float(ds.variables["x"][xi])
        y_val = float(ds.variables["y"][yi])

    # Apply valid-range mask
    vi_min, vi_max = VI_VALID_RANGE.get(vi_var, (-1.0, 2.0))
    valid_mask = (
        ~np.isnan(vi_arr)
        & (vi_arr >= vi_min)
        & (vi_arr <= vi_max)
    )

    lon_arr, lat_arr = utm_to_latlon(np.array([x_val]), np.array([y_val]))

    return PixelTimeSeries(
        dates=dates,
        raw_vi=vi_arr,
        valid_mask=valid_mask,
        x_coord=x_val,
        y_coord=y_val,
        lon=float(lon_arr[0]),
        lat=float(lat_arr[0]),
    )


# ---------------------------------------------------------------------------
# Click coordinate → array index mapping
# ---------------------------------------------------------------------------

def click_to_array_index(
    click_lon: float,
    click_lat: float,
    ds: xr.Dataset,
    src_epsg: int | None = None,
) -> tuple[int, int]:
    """
    Convert a Plotly heatmap click's (lon, lat) display coordinates to
    zero-based (yi, xi) array indices via nearest-neighbour lookup.

    Parameters
    ----------
    click_lon, click_lat : WGS84 coordinates from the Plotly click event
    ds                   : lazily opened xarray Dataset for the region
    src_epsg             : UTM EPSG of the datacube; auto-detected if None

    Returns
    -------
    (yi, xi) clamped to valid array bounds
    """
    if src_epsg is None:
        src_epsg = detect_crs_epsg(ds)

    x_click, y_click = latlon_to_utm(click_lon, click_lat, src_epsg)

    x_vals = ds["x"].values  # (nx,) — coordinate arrays are tiny
    y_vals = ds["y"].values  # (ny,)

    xi = int(np.argmin(np.abs(x_vals - x_click)))
    yi = int(np.argmin(np.abs(y_vals - y_click)))

    xi = max(0, min(xi, len(x_vals) - 1))
    yi = max(0, min(yi, len(y_vals) - 1))

    return yi, xi
