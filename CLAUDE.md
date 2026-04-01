# VI Phenology Plotly Cloud Dashboard — Claude Code Context

## Project Overview

Plotly Dash version of the BioSCape Phenology Explorer, designed for Dash Enterprise /
Cloud deployment.  Functionally mirrors the Shiny dashboard but uses Dash callbacks
instead of Shiny reactives, and dash-leaflet instead of ipyleaflet.

## Running

```bash
python app.py                   # http://127.0.0.1:8050
gunicorn app:server             # Dash Enterprise / gunicorn deployment
```

## Key Files

| File | Role |
|---|---|
| `app.py` | Dash app entry point; all Dash callbacks |
| `config.py` | All constants — data paths, VI ranges, shapefile paths, metric config |
| `modules/datacube_io.py` | File discovery, lazy Dask loading, pixel extraction, basemap cache |
| `modules/phenology_metrics.py` | Whittaker smoothing + 19 per-pixel phenological metrics |
| `modules/visualization.py` | Plotly figure factories, dash-leaflet helpers, metrics table HTML |
| `tools/pixel_phenology_extract.py` | Batch per-pixel metric extraction → `pixel_metrics.nc` |
| `tools/convert_to_zarr.py` | One-time CLI: rechunk NC → Zarr |
| `tools/cache_basemaps.py` | One-time CLI: pre-compute basemap `.npz` caches |
| `shapefiles/` | LVIS_Flightboxes.geojson |

## Architecture Notes

### State management
- Shiny's `reactive.Value` / `reactive.Calc` → `dcc.Store` components.
- Pixel result (Whittaker solve) is computed once in `compute_pixel_result` callback
  and stored in `pixel-result-store`; all four chart callbacks read from that store
  to avoid redundant computation.

### Colorscale limits — important caveat
`_compute_colorscale_limits(z, sel, metric)` in `app.py` computes SD-clipped bounds.
For metrics in `NONNEGATIVE_METRICS` (config.py), `zmin` is floored to `max(0.0, zmin)`.
Without this floor, metrics with high spatial variability (e.g., `peak_doy_std`,
`season_length_mean`) produce a negative lower colorscale bound, which makes the
colorbar display physically impossible values.

### Std band in annual metric trends — same floor applies
`make_metrics_annual_figure` in `visualization.py` draws ±1 std shaded bands.
The lower bound `lo = mean - std` is floored to `max(0.0, lo)` for metrics in
`NONNEGATIVE_METRICS` to prevent the shaded region from extending below zero.

### Basemap amplitude filter
The VI amplitude filter (colorscale zmin/zmax → valid observation gating) is only
applied when the active basemap metric is a "fast" VI-scaled metric (one of the four
in `FAST_BASEMAP_METRICS`).  Phenology metrics (DOY, days, etc.) have colorscale
values in different units and must never gate VI observations.  See `_FAST_METRIC_KEYS`
guard in `compute_pixel_result`.

### pixel_metrics.nc
Same naming convention as Shiny: `{vi_var}_{region_id}_pixel_metrics.nc`.
Same 19 metrics, same repair tool (`tools/repair_pixel_metrics.py` in the Shiny repo
or an equivalent copy here).

## Config Reference

```python
# NONNEGATIVE_METRICS — metrics physically bounded below by zero.
# Used to floor SD-clipped colorscale zmin and std band lower bound.
# Excluded (can be negative): peak_ndvi_mean, integrated_ndvi_mean,
#                              floor_ndvi_mean, ceiling_ndvi_mean
NONNEGATIVE_METRICS: frozenset[str] = frozenset({...})
```

## Known Display Issues (Fixed)

| Issue | Root cause | Fix |
|---|---|---|
| Season Length, Peak DOY std, Peak Separation showing wildly wrong values (~-9.2e18) | xarray decodes variables with `units="days"` as `timedelta64`; float32 fill value (9.97e36 days) overflows to int64 min | Add `decode_timedelta=False` to `xr.open_dataset` in `load_metrics_for_basemap` (datacube_io.py) |
| Negative lower colorscale bound for Season Length, Peak DOY std, etc. | `_compute_colorscale_limits` returned `mean - N*std < 0` | Floor `zmin` to 0 for `NONNEGATIVE_METRICS` in `app.py` |
| Std band shading extends below zero in annual metric trends | `lo = mean_val - std_val` unclamped | Floor `lo` to 0 for `NONNEGATIVE_METRICS` in `visualization.py:make_metrics_annual_figure` |
