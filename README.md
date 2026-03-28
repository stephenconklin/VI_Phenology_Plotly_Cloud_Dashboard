# VI Phenology Plotly Cloud Dashboard

An interactive web dashboard for exploring vegetation index (VI) phenology from BioSCape airborne imaging spectroscopy data. Built with Plotly Dash and Dash-Leaflet, deployable to Plotly Cloud or any Gunicorn-compatible host.

## Features

- **Interactive map** with LVIS flight box overlays — click a polygon to select a region
- **Basemap rendering** of spatial phenology metrics (peak NDVI, mean VI, data coverage, and 19 precomputed metrics) at near-native 30 m resolution
- **Per-pixel time series** extraction with Whittaker smoother
- **Phenology charts**: time series, annual cycle, metrics table, and interannual scatter
- **Lazy data loading** via Dask — full datacubes are never loaded into memory
- **GCS-native** — reads NetCDF and Zarr stores directly from Google Cloud Storage

## Data

Data are hosted in the `bioscape_phenology_data` GCS bucket. Each region corresponds to an LVIS flight box (e.g. `G5_8`) and expects one or more of:

| File | Description |
|---|---|
| `<region>_<VI>_datacube.nc` | Full time-series datacube (NetCDF4/HDF5) |
| `<region>_<VI>_datacube.zarr/` | Rechunked Zarr store (faster pixel reads) |
| `<region>_pixel_metrics.nc` | Precomputed 19-metric phenology summary |

See [`tools/convert_to_zarr.py`](tools/convert_to_zarr.py) and [`tools/pixel_phenology_extract.py`](tools/pixel_phenology_extract.py) for offline data preparation utilities.

## Setup

### Requirements

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

### Configuration

All settings are in [`config.py`](config.py). The most common overrides via environment variables:

| Variable | Description | Default |
|---|---|---|
| `VI_DATACUBE_ROOT` | Data root — local path or `gs://` URI | `gs://bioscape_phenology_data/` |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full JSON content of a GCP service account key (for private buckets) | — |
| `GCS_TOKEN` | Auth method: `anon` or `google_default` | `google_default` |
| `BioSCape` | GCP project ID (for billing attribution) | — |

### Running Locally

```bash
python app.py
```

The app will be available at `http://localhost:8050`.

### Deployment (Plotly Cloud / Gunicorn)

```bash
gunicorn app:server
```

Set `GOOGLE_SERVICE_ACCOUNT_JSON` in your platform's environment variables with the full contents of a service account JSON key that has **Storage Object Viewer** access to the data bucket.

## Project Structure

```
app.py                        # Dash application, layout, and callbacks
config.py                     # All tuneable constants and environment variable overrides
requirements.txt              # Python dependencies
Procfile                      # Gunicorn entry point for cloud deployment
modules/
  datacube_io.py              # Region discovery, dataset loading, pixel extraction
  visualization.py            # Map helpers, shapefile loading, chart rendering
  phenology_metrics.py        # Whittaker smoothing and phenological metric computation
shapefiles/
  LVIS_Flightboxes.geojson    # LVIS flight box polygons (box_nr field = region ID)
  BioSCape_HLS_Tiles.geojson  # HLS tile boundaries (optional overlay)
tools/
  convert_to_zarr.py          # Rechunk NetCDF → Zarr for faster GCS pixel reads
  pixel_phenology_extract.py  # Batch compute per-pixel phenology metrics
  cache_basemaps.py           # Pre-render basemap cache files
```

## License

MIT — see [LICENSE](LICENSE).
