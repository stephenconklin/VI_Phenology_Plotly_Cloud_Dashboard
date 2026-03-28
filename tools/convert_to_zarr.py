"""
convert_to_zarr.py — Offline utility: rechunk NetCDF4 datacubes to ZARR.

Why ZARR?
---------
The netCDF4 datacubes use HDF5 compression with default or spatial chunking.
For single-pixel time-series access (the dashboard's core operation), HDF5
must decompress many spatial chunks to read one pixel's full time series.

Rechunking to ZARR with chunks={'time': -1, 'y': 10, 'x': 10} places each
10×10 pixel block's full time series in a single chunk.  Reading one pixel
decompresses at most 4 chunks (worst case, pixel at chunk boundary).

This reduces pixel read latency from potentially hundreds of chunk reads
down to 1–4.  Run this script once per region before using the dashboard
with large files (G5_7, G5_10, G5_14 are most in need).

After conversion the dashboard auto-detects the .zarr directory and uses it
without any configuration change.

Usage
-----
    # Activate the vi_phenology_dashboard conda environment first.

    # Convert one region:
    python tools/convert_to_zarr.py --region G5_14

    # Convert all regions:
    python tools/convert_to_zarr.py --all

    # Override data root:
    VI_DATACUBE_ROOT=/path/to/data python tools/convert_to_zarr.py --all

    # Dry run (show what would be converted):
    python tools/convert_to_zarr.py --all --dry-run

Disk space
----------
ZARR with blosc/lz4 compression typically uses 1–1.5× the NC file size.
Ensure sufficient disk space before converting large regions.

Chunk sizes
-----------
Default: time=-1 (full axis), y=10, x=10.
For very large regions (G5_14: 2222 rows × 409 cols) this creates
~91,000 spatial chunks.  Total chunk count is manageable for zarr.

Adjust --chunk-y and --chunk-x if your access pattern differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xarray as xr

from modules.datacube_io import discover_regions, RegionPaths


# ---------------------------------------------------------------------------
# Conversion function
# ---------------------------------------------------------------------------

def convert_region(
    paths: RegionPaths,
    chunk_y: int = 10,
    chunk_x: int = 10,
    dry_run: bool = False,
) -> None:
    """
    Rechunk one region's .nc file to .zarr with time-first chunking.

    Parameters
    ----------
    paths    : RegionPaths for the region
    chunk_y  : spatial chunk size along the y axis
    chunk_x  : spatial chunk size along the x axis
    dry_run  : if True, print what would happen but don't write
    """
    zarr_path = paths.nc_path.parent / (paths.nc_path.stem + ".zarr")

    if zarr_path.exists():
        print(f"[{paths.region_id}] ZARR already exists — skipping: {zarr_path}")
        return

    nc_size_mb = paths.nc_path.stat().st_size / (1024 ** 2)
    print(
        f"[{paths.region_id}] NC size: {nc_size_mb:.0f} MB "
        f"→ ZARR: {zarr_path}"
    )

    if dry_run:
        print(f"[{paths.region_id}] (dry run — no files written)")
        return

    print(f"[{paths.region_id}] Opening NC with dask (read chunks: time=100, y=200, x=200)...")
    ds = xr.open_dataset(
        str(paths.nc_path),
        engine="netcdf4",
        chunks={"time": 100, "y": 200, "x": 200},
        mask_and_scale=True,
    )

    print(
        f"[{paths.region_id}] Rechunking to time=-1, y={chunk_y}, x={chunk_x}..."
    )
    ds_rechunked = ds.chunk({"time": -1, "y": chunk_y, "x": chunk_x})

    print(f"[{paths.region_id}] Writing ZARR (this may take several minutes)...")
    ds_rechunked.to_zarr(str(zarr_path), mode="w")

    zarr_size_mb = sum(
        f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
    ) / (1024 ** 2)
    print(
        f"[{paths.region_id}] Done. "
        f"ZARR size: {zarr_size_mb:.0f} MB "
        f"(ratio {zarr_size_mb / nc_size_mb:.2f}× vs NC)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rechunk VI Phenology NetCDF4 datacubes to ZARR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--region",
        metavar="REGION_ID",
        help="Convert a single region (e.g. G5_14)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Convert all discovered regions",
    )
    parser.add_argument(
        "--chunk-y",
        type=int,
        default=10,
        metavar="N",
        help="Spatial chunk size along y axis (default: 10)",
    )
    parser.add_argument(
        "--chunk-x",
        type=int,
        default=10,
        metavar="N",
        help="Spatial chunk size along x axis (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing files",
    )
    args = parser.parse_args()

    try:
        regions = discover_regions()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not regions:
        print("No regions found. Check VI_DATACUBE_ROOT.")
        sys.exit(1)

    if args.all:
        print(f"Found {len(regions)} regions: {', '.join(regions)}")
        for paths in regions.values():
            convert_region(paths, args.chunk_y, args.chunk_x, args.dry_run)
    else:
        if args.region not in regions:
            print(
                f"Region '{args.region}' not found. "
                f"Available: {', '.join(regions)}"
            )
            sys.exit(1)
        convert_region(
            regions[args.region], args.chunk_y, args.chunk_x, args.dry_run
        )


if __name__ == "__main__":
    main()
