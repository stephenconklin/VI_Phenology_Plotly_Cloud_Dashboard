"""
config.py — VI Phenology Dashboard configuration.

All tuneable constants live here. Override DATACUBE_ROOT via the
VI_DATACUBE_ROOT environment variable to avoid editing this file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

# Supports local paths and Google Cloud Storage URIs (gs://bucket/path).
# Override via the VI_DATACUBE_ROOT environment variable.
DATACUBE_ROOT: str = os.environ.get(
    "VI_DATACUBE_ROOT",
    # "/Volumes/ConklinGeospatialData/Data/BioSCape_SA_LVIS/VI_Phenology/netcdf_datacube",
    # Public URL for Google Coud Storage Bucket: https://storage.googleapis.com/bioscape_phenology_data/
    "gs://bioscape_phenology_data/"
)

# GCS project ID — required when using Application Default Credentials with a
# billing-enabled project.  Set via the GCS_PROJECT environment variable.
# Leave unset (None) when using a service-account key file or Workload Identity.
GCS_PROJECT: str | None = os.environ.get("BioSCape", None)

# ---------------------------------------------------------------------------
# Shapefile overlays (optional — set to None to disable)
# ---------------------------------------------------------------------------

# Space-separated list of shapefile paths to display over the basemap.
# Set to None to disable all overlays.
# Example: "data/regions.shp data/plots.shp"
# SHAPEFILE_PATHS: str | None = None
# SHAPEFILE_PATHS: str = "/Users/stephenconklin/Documents/ConklinGeospatial/Projects/GITHub/VI_Phenology_Dashboard/shapefiles/LVIS_Flightboxes.geojson \
#                         /Users/stephenconklin/Documents/ConklinGeospatial/Projects/GITHub/VI_Phenology_Dashboard/shapefiles/BioSCape_HLS_Tiles.geojson"
_PROJECT_ROOT = Path(__file__).parent
SHAPEFILE_PATHS: str = str(_PROJECT_ROOT / "shapefiles" / "LVIS_Flightboxes.geojson")

# Space-separated list of attribute field names used as labels, one per file.
# Must match the order of SHAPEFILE_PATHS.
# Example: "NAME plot_id"
# SHAPEFILE_LABEL_FIELDS: str = "box_nr Name"
SHAPEFILE_LABEL_FIELDS: str = "box_nr"

# ---------------------------------------------------------------------------
# Spatial / coordinate reference
# ---------------------------------------------------------------------------

# CRS of the datacubes (UTM Zone 34S). Auto-detected from spatial_ref at runtime
# but this default avoids re-reading the file for reprojection setup.
DATACUBE_CRS_EPSG: int = 32734

# Display CRS (WGS84 geographic, for Plotly axes).
TARGET_CRS_EPSG: int = 4326

# ---------------------------------------------------------------------------
# VI settings
# ---------------------------------------------------------------------------

DEFAULT_VI_VAR = "NDVI"

# Valid observation range per VI.
VI_VALID_RANGE: dict[str, tuple[float, float]] = {
    "NDVI": (-0.1, 1.0),
    "EVI2": (-1.0, 2.0),
    "NIRv": (-0.5, 1.0),
}

# ---------------------------------------------------------------------------
# Whittaker smoothing
# ---------------------------------------------------------------------------

LAMBDA_MIN: float = 10.0
LAMBDA_MAX: float = 1000.0
LAMBDA_DEFAULT: float = 500.0
LAMBDA_STEP: int = 10

# ---------------------------------------------------------------------------
# Basemap display
# ---------------------------------------------------------------------------

# Maximum pixel count per axis for the on-the-fly Dask basemap path.
# Kept small for performance (Dask compute over full time axis).
BASEMAP_MAX_DIM: int = 500

# Maximum pixel count per axis for the precomputed pixel_metrics.nc path.
# Data is already reduced to 2D so no Dask cost — use a higher limit to
# display at or near native 30 m resolution for all current LVIS regions
# (largest is G5_14 at 2 222 × 409 px).
BASEMAP_MAX_DIM_PRECOMPUTED: int = 2000

# Fast basemap metrics always available (no precomputed file required).
# Keys are the internal metric IDs used in compute_basemap_metric().
FAST_BASEMAP_METRICS: dict[str, str] = {
    "Peak VI": "peak_ndvi_mean",
    "Mean VI": "mean_ndvi",
    "Std Dev VI": "std_ndvi",
    "Data Coverage": "data_coverage",
}

# ---------------------------------------------------------------------------
# Per-pixel metric computation defaults
# ---------------------------------------------------------------------------

PIXEL_METRIC_CONFIG: dict = {
    "vi_min": VI_VALID_RANGE["NDVI"][0],
    "vi_max": VI_VALID_RANGE["NDVI"][1],
    "min_valid_obs": 20,
    "min_valid_obs_per_year": 5,
    "peak_prominence": 0.05,
    "peak_min_distance_days": 45,
    "season_threshold": 0.20,
}

# ---------------------------------------------------------------------------
# 19 phenological metric names (exact order from pixel_phenology_extract.py)
# ---------------------------------------------------------------------------

ALL_19_METRICS: list[str] = [
    "peak_ndvi_mean",
    "peak_ndvi_std",
    "peak_doy_mean",
    "peak_doy_std",
    "integrated_ndvi_mean",
    "integrated_ndvi_std",
    "greenup_rate_mean",
    "greenup_rate_std",
    "floor_ndvi_mean",
    "ceiling_ndvi_mean",
    "season_length_mean",
    "season_length_std",
    "cv",
    "interannual_peak_range",
    "interannual_peak_std",
    "n_peaks_mean",
    "peak_separation_mean",
    "relative_peak_amplitude_mean",
    "valley_depth_mean",
]

# Display labels and units for each metric.
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "peak_ndvi_mean":               ("Peak NDVI (mean)",             "NDVI"),
    "peak_ndvi_std":                ("Peak NDVI (std)",              "NDVI"),
    "peak_doy_mean":                ("Peak DOY (mean)",              "DOY"),
    "peak_doy_std":                 ("Peak DOY (std)",               "days"),
    "integrated_ndvi_mean":         ("Integrated NDVI (mean)",       "NDVI·days yr⁻¹"),
    "integrated_ndvi_std":          ("Integrated NDVI (std)",        "NDVI·days"),
    "greenup_rate_mean":            ("Green-up Rate (mean)",         "NDVI day⁻¹"),
    "greenup_rate_std":             ("Green-up Rate (std)",          "NDVI day⁻¹"),
    "floor_ndvi_mean":              ("Floor NDVI",                   "NDVI"),
    "ceiling_ndvi_mean":            ("Ceiling NDVI",                 "NDVI"),
    "season_length_mean":           ("Season Length (mean)",         "days"),
    "season_length_std":            ("Season Length (std)",          "days"),
    "cv":                           ("Coeff. of Variation",          "ratio"),
    "interannual_peak_range":       ("Interannual Peak Range",       "NDVI"),
    "interannual_peak_std":         ("Interannual Peak Std",         "NDVI"),
    "n_peaks_mean":                 ("N Peaks (mean)",               "peaks yr⁻¹"),
    "peak_separation_mean":         ("Peak Separation (mean)",       "days"),
    "relative_peak_amplitude_mean": ("Relative Peak Amplitude",      "ratio"),
    "valley_depth_mean":            ("Valley Depth (mean)",          "normalized"),
}

# Grouped for sidebar display.
METRIC_GROUPS: dict[str, list[str]] = {
    "Peak": [
        "peak_ndvi_mean", "peak_ndvi_std",
        "peak_doy_mean", "peak_doy_std",
    ],
    "Productivity": [
        "integrated_ndvi_mean", "integrated_ndvi_std",
        "greenup_rate_mean", "greenup_rate_std",
    ],
    "Seasonality": [
        "floor_ndvi_mean", "ceiling_ndvi_mean",
        "season_length_mean", "season_length_std",
    ],
    "Variability": [
        "cv", "interannual_peak_range", "interannual_peak_std",
    ],
    "Bimodality": [
        "n_peaks_mean", "peak_separation_mean",
        "relative_peak_amplitude_mean", "valley_depth_mean",
    ],
}
