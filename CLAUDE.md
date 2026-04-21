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

### Font scaling system
A-/A+ buttons in the sidebar header adjust a `--fs-scale` CSS custom property on
`:root` via a clientside callback. All `font-size` values in `custom.css` use
`calc(Xpx * var(--fs-scale))` so one JS variable change cascades everywhere.
Scale steps: `[0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]`. Current scale stored
in `dcc.Store(id="font-scale-store")`.

### Colorbar as map overlay
`make_colorbar_component()` in `visualization.py` returns an `html.Div` tree (not
an HTML string — `dcc.Markdown(dangerously_allow_html=True)` is unreliable in Dash 4).
The `#colorbar-div` is placed inside `#map-wrapper` (which has `position: relative`)
and positioned with `position: absolute; bottom: 30px; right: 10px; z-index: 1000`.

### Metric layer opacity
`dl.ImageOverlay.opacity` prop controls user-visible transparency. The PNG alpha
channel is **not** used for opacity — valid pixels are always fully opaque (alpha=255)
in the PNG, and NaN pixels are fully transparent (alpha=0). This avoids double-opacity.
Use `float(opacity if opacity is not None else 0.75)` — never `float(opacity or 0.75)`
because `0.0 or 0.75` evaluates to `0.75` in Python.

### Metric Trends tab — scrollable chart area
The chart renders at a fixed pixel height (`n_rows * 260 + 120` px, set via
`autosize=False` in the figure layout). The `render_metrics_annual` callback outputs
both `figure` and `style={"height": f"{fig.layout.height}px"}` to the `dcc.Graph`
element.

The wrapper `#metrics-annual-chart-wrapper` uses an explicit CSS variable height —
**not** `height: 100%` — because `dbc.Tabs` may render as a fragment or wrapper div
depending on version, making percentage height propagation unreliable.

`resize.js` computes `--charts-h` (panel height minus map height minus 6px divider)
on every drag and on init, setting it on `#main-panel-col`. The wrapper CSS:
```css
#metrics-annual-chart-wrapper {
  height: calc(var(--charts-h, 50vh) - 42px);  /* 42px = Bootstrap nav-tabs bar */
  overflow-y: auto;
}
```
This gives a concrete pixel height the browser measures overflow against, activating
the scrollbar when the figure is taller than the available space.

### Phenology Scatter — per-year discrete colors
`make_phenology_scatter_figure` creates one `go.Scatter` trace per unique year using
`_year_color(i_yr)` from `_YEAR_PALETTE`. This gives a discrete legend entry per year
instead of a continuous colorscale colorbar. The palette is 12 bright colors chosen
for visibility on the dark (`#060c12`) background.

### Dash 4.x CSS class names (breaking change)
Dash 4 uses completely different CSS class names from older versions. Do not use
old react-select or Bootstrap form-check class names. Correct Dash 4 names:

| Component | CSS selector |
|---|---|
| Dropdown container | `.dash-dropdown-wrapper` |
| Dropdown value | `.dash-dropdown-value` |
| Dropdown placeholder | `.dash-dropdown-placeholder` |
| Dropdown options list | `.dash-dropdown-options` |
| Dropdown option item | `.dash-dropdown-option` |
| Dropdown search input | `.dash-dropdown-search` |
| Slider tooltip | `.dash-slider-tooltip` |
| Range slider input | `.dash-input-container`, `.dash-range-slider-input` |
| Checklist option text | `.dash-options-list-option-text` |
| Checklist option wrapper | `.dash-options-list-option-wrapper` |
| Checklist checkbox | `.dash-options-list-option-checkbox` |

Find actual names by grepping the installed JS bundles:
```bash
grep -o 'className:"[^"]*"' .venv/lib/*/site-packages/dash/deps/async-dropdown.js | sort -u
```

### Matplotlib RGBA in Python
Matplotlib does **not** accept CSS `rgba()` strings. Use normalized float tuples:
```python
cb.outline.set_edgecolor((0.357, 0.890, 1.0, 0.3))  # correct
cb.outline.set_edgecolor("rgba(91,227,255,0.3)")      # ValueError
```

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
| SyntaxError: keyword argument repeated (xaxis, yaxis, font) | `go.Layout()` calls had duplicate kwargs from Claude Design session | Merge all duplicate dict keys in `go.Layout()` calls in `visualization.py` (3 locations) |
| White-on-white text in dropdowns, sliders, checklists | Dash 4 uses different CSS class names; old react-select/Bootstrap selectors don't apply | Rewrite `custom.css` with Dash 4 class names (see table above) |
| Colorbar not visible on map | `dcc.Markdown(dangerously_allow_html=True)` unreliable in Dash 4 | Replace with `make_colorbar_component()` returning `html.Div` tree |
| Opacity slider inverted / double-opacity | PNG alpha channel × `ImageOverlay.opacity` both encoding opacity | PNG alpha always 255 for valid pixels; only `ImageOverlay.opacity` prop controls transparency |
| Opacity=0 still shows 75% | `float(0.0 or 0.75)` → `0.75` (Python falsy) | Use `float(opacity if opacity is not None else 0.75)` |
| No scrollbar in Metric Trends tab | `dbc.Tabs` wrapper div not a flex container; `.tab-pane height: 100%` resolves to `auto` | Add `#charts-wrapper > div { flex: 1 1 0; display: flex; flex-direction: column; }` to `resize.css` |
| Mean line in Metric Trends black/invisible | Hard-coded `color="#000000"` on dark background | Changed to `rgba(91,227,255,0.75)` (theme cyan); std fill to `rgba(91,227,255,0.10)` |
| Phenology Scatter uses continuous colorscale for Year | Single trace with `colorscale="Plasma"` | One trace per year using `_year_color()` palette; discrete legend entries |
