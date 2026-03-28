"""
cache_basemaps.py — Offline utility: pre-compute and cache basemap arrays.

Why?
----
When the dashboard first displays a region its basemap is computed on the fly
via a full Dask temporal reduction over the Zarr store.  For large regions
(G5_14: ~23 GB uncompressed) this can take 5–30 seconds and freezes the
browser until it finishes.

Running this script once writes a small .npz cache file next to each datacube
(local or GCS):

    NDVI_G5_14_datacube_basemap_mean_ndvi_d500.npz   (~300 KB)

The dashboard detects these files and loads them in milliseconds instead of
running the Dask compute.

Cache file naming
-----------------
    <zarr_stem>_basemap_<metric>_d<max_dim>.npz

    zarr_stem : name of the zarr store without extension
                  e.g. NDVI_G5_14_datacube
    metric    : internal metric key (mean_ndvi, peak_ndvi_mean, std_ndvi,
                  data_coverage)
    max_dim   : maximum display pixels per axis (matches config.BASEMAP_MAX_DIM,
                  default 500)

For GCS targets the .npz is written to the same bucket path as the zarr store.

GCS authentication
------------------
Set the GOOGLE_SERVICE_ACCOUNT_JSON environment variable to the path of your
service account key file (same variable the dashboard uses):

    export GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/key.json
    python tools/cache_basemaps.py --all

Usage
-----
    # Activate the vi_phenology_dashboard conda environment first.

    # Cache all 4 fast metrics for every region:
    python tools/cache_basemaps.py --all

    # Cache one specific region:
    python tools/cache_basemaps.py --region G5_14

    # Force-recompute even if cache already exists:
    python tools/cache_basemaps.py --all --force

    # Cache only specific metrics:
    python tools/cache_basemaps.py --all --metrics mean_ndvi peak_ndvi_mean

    # Override the max display resolution (default: config.BASEMAP_MAX_DIM = 500):
    python tools/cache_basemaps.py --all --max-dim 300

    # Dry run — show what would be computed without writing anything:
    python tools/cache_basemaps.py --all --dry-run

    # Override data root:
    VI_DATACUBE_ROOT=gs://my-bucket/data python tools/cache_basemaps.py --all

Expected metrics
----------------
    mean_ndvi        Mean VI over time (default basemap)
    peak_ndvi_mean   Maximum VI over time
    std_ndvi         Standard deviation over time
    data_coverage    Fraction of non-NaN observations
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a standalone script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASEMAP_MAX_DIM, FAST_BASEMAP_METRICS
from modules.datacube_io import (
    RegionPaths,
    basemap_cache_path,
    compute_basemap_metric,
    discover_regions,
    get_dataset,
    load_basemap_cache,
    save_basemap_cache,
)

_ALL_FAST_METRICS: list[str] = list(FAST_BASEMAP_METRICS.values())


def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _path_label(path: str) -> str:
    """Short display name for a data path (last two components)."""
    parts = path.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else path


# ---------------------------------------------------------------------------
# Per-region caching
# ---------------------------------------------------------------------------

def cache_region(
    paths: RegionPaths,
    metrics: list[str],
    max_dim: int,
    force: bool,
    dry_run: bool,
) -> None:
    """
    Compute and cache basemap arrays for one region.

    Parameters
    ----------
    paths   : RegionPaths for the region
    metrics : list of metric keys to compute
    max_dim : maximum display pixels per axis
    force   : if True, recompute even if a cache file already exists
    dry_run : if True, print what would happen but don't write
    """
    data_path = paths.zarr_path or paths.nc_path
    if data_path is None:
        print(f"\n[{paths.region_id}]  SKIP — no zarr or nc path")
        return

    print(f"\n[{paths.region_id}]  {_path_label(data_path)}")

    ds = None  # opened lazily on first cache miss

    for metric in metrics:
        cache = basemap_cache_path(data_path, metric, max_dim)

        if not force:
            hit = load_basemap_cache(cache)
            if hit is not None:
                z, _, _ = hit
                print(f"  {metric:20s}  [cached]  shape={z.shape}")
                continue

        cache_name = cache.rsplit("/", 1)[-1] if "/" in cache else cache
        if dry_run:
            print(f"  {metric:20s}  [would compute]  → {cache_name}")
            continue

        print(f"  {metric:20s}  computing … ", end="", flush=True)
        t0 = time.perf_counter()

        if ds is None:
            ds = get_dataset(paths)

        try:
            z, lon, lat = compute_basemap_metric(
                ds, metric, vi_var=paths.vi_var, max_dim=max_dim
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        save_basemap_cache(cache, z, lon, lat)
        print(f"done  ({elapsed:.1f}s)  shape={z.shape}  → {cache_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache basemap arrays for the VI Phenology Dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--region",
        metavar="REGION_ID",
        help="Cache basemaps for a single region (e.g. G5_14)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Cache basemaps for all discovered regions",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        metavar="METRIC",
        default=_ALL_FAST_METRICS,
        choices=_ALL_FAST_METRICS,
        help=(
            "Metric keys to cache (default: all 4 fast metrics). "
            f"Choices: {_ALL_FAST_METRICS}"
        ),
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=BASEMAP_MAX_DIM,
        metavar="N",
        help=f"Max display pixels per axis (default: {BASEMAP_MAX_DIM}, matches config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing cache files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be computed without writing any files",
    )
    args = parser.parse_args()

    try:
        regions = discover_regions()
    except Exception as e:
        print(f"Error discovering regions: {e}")
        sys.exit(1)

    if not regions:
        print("No regions found. Check VI_DATACUBE_ROOT and GCS credentials.")
        sys.exit(1)

    if args.all:
        target_regions = list(regions.values())
        print(f"Found {len(target_regions)} regions: {', '.join(r.region_id for r in target_regions)}")
    else:
        if args.region not in regions:
            print(
                f"Region '{args.region}' not found. "
                f"Available: {', '.join(regions)}"
            )
            sys.exit(1)
        target_regions = [regions[args.region]]

    print(f"Metrics : {args.metrics}")
    print(f"Max dim : {args.max_dim}")
    if args.force:
        print("Force   : yes — existing cache files will be overwritten")
    if args.dry_run:
        print("Dry run : no files will be written.\n")

    t_total = time.perf_counter()
    for paths in target_regions:
        cache_region(paths, args.metrics, args.max_dim, args.force, args.dry_run)

    elapsed_total = time.perf_counter() - t_total
    print(f"\nFinished in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
