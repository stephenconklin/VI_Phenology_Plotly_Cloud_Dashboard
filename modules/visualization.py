"""
visualization.py — Plotly Figure factory functions for the Dash version.

Key differences from the Shiny version:
- go.Figure is used instead of go.FigureWidget (Dash handles all interactivity
  via callbacks; Python-side on_click() is not needed).
- ipyleaflet is replaced by dash-leaflet helper functions that return
  component props dicts for use in Dash callbacks.
- make_leaflet_map() / update_leaflet_map() / add_shapefile_overlay() are
  replaced by get_tile_layer_props(), get_overlay_url_and_bounds(), and
  get_shapefile_geojson_data() which return plain dicts consumed by callbacks.
"""

from __future__ import annotations

import base64
import io
import math
from typing import NamedTuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure
from PIL import Image as PILImage

import pandas as pd
from plotly.subplots import make_subplots as _make_subplots

from config import METRIC_LABELS, METRIC_GROUPS, VI_VALID_RANGE
from modules.phenology_metrics import _build_whittaker_system, _whittaker_smooth_pixel


class SatelliteImage(NamedTuple):
    """
    A satellite raster tile fetched from a tile service, ready for
    use as a Plotly layout.images entry in geographic (lon/lat) coordinates.
    """
    data_uri: str    # "data:image/png;base64,..." — base64-encoded PNG
    lon_min: float   # west edge of the fetched image (WGS84)
    lon_max: float   # east edge
    lat_min: float   # south edge
    lat_max: float   # north edge


def _z_to_json_safe(arr: np.ndarray) -> list:
    """
    Convert a 2-D numpy array to a nested list with NaN/Inf replaced by None.
    Plotly renders None as a gap in heatmaps (correct visual for masked pixels).
    """
    return [
        [
            None if (v is not None and isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
            for v in row
        ]
        for row in arr.tolist()
    ]


# ---------------------------------------------------------------------------
# Colour scales
# ---------------------------------------------------------------------------

_NDVI_COLORSCALE   = "RdYlGn"
_METRIC_COLORSCALE = "Viridis"
_COVERAGE_COLORSCALE = "Blues"


def _choose_colorscale(metric_key: str) -> str:
    if "coverage" in metric_key:
        return _COVERAGE_COLORSCALE
    if "ndvi" in metric_key or "integrated" in metric_key:
        return _NDVI_COLORSCALE
    return _METRIC_COLORSCALE


# ---------------------------------------------------------------------------
# Tile services and map helpers
# ---------------------------------------------------------------------------

LEAFLET_TILE_SERVICES: dict[str, dict] = {
    "World_Imagery": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, Maxar, Earthstar Geographics",
        "max_zoom":    18,
        "label":       "Satellite",
    },
    "World_Topo_Map": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, HERE, Garmin, FAO, USGS, NGA",
        "max_zoom":    18,
        "label":       "Topographic",
    },
    "World_Shaded_Relief": {
        "url":         "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, USGS, NOAA",
        "max_zoom":    13,
        "label":       "Shaded Relief",
    },
    "OpenStreetMap": {
        "url":         "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "max_zoom":    19,
        "label":       "OpenStreetMap",
    },
}


def _auto_zoom(lat_range: float, lon_range: float) -> int:
    """Estimate an initial Leaflet zoom level that fits the bounding box."""
    extent = max(lat_range, lon_range)
    if extent <= 0:
        return 10
    return max(6, min(14, int(math.log2(360.0 / extent)) + 1))


def get_tile_layer_props(tile_service: str) -> dict:
    """
    Return a dict with url, attribution, maxZoom for a dash-leaflet TileLayer.
    """
    svc = LEAFLET_TILE_SERVICES.get(tile_service, LEAFLET_TILE_SERVICES["World_Imagery"])
    return {
        "url":         svc["url"],
        "attribution": svc["attribution"],
        "maxZoom":     svc.get("max_zoom", 18),
    }


def get_map_center_and_zoom(lon: np.ndarray, lat: np.ndarray) -> tuple[list[float], int]:
    """Return [lat_center, lon_center] and initial zoom level."""
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
    zoom   = _auto_zoom(lat_max - lat_min, lon_max - lon_min)
    return center, zoom


def get_overlay_url_and_bounds(
    z: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    metric_key: str,
    opacity: float = 0.75,
    zmin: float | None = None,
    zmax: float | None = None,
) -> tuple[str, list]:
    """
    Compute the PNG data URI and bounds for a dash-leaflet ImageOverlay.

    Returns
    -------
    url    : base64-encoded RGBA PNG data URI
    bounds : [[lat_min, lon_min], [lat_max, lon_max]]
    """
    url = make_metric_overlay_png(z, lat, metric_key, zmin, zmax, opacity)
    bounds = [
        [float(lat.min()), float(lon.min())],
        [float(lat.max()), float(lon.max())],
    ]
    return url, bounds


def get_shapefile_geojson_data(shapefile_path: str) -> dict | None:
    """
    Load a shapefile / GeoJSON and return the GeoJSON dict (WGS84) for
    use as the `data` prop of a dash-leaflet GeoJSON component.
    Returns None if the file is missing or geopandas is unavailable.
    """
    try:
        import geopandas as gpd
    except ImportError:
        return None

    from pathlib import Path as _Path
    path = _Path(shapefile_path)
    if not path.exists():
        return None

    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.__geo_interface__


# ---------------------------------------------------------------------------
# Metric overlay PNG + colorbar HTML
# ---------------------------------------------------------------------------

def _mpl_cmap_for(metric_key: str) -> str:
    """Return the matplotlib colormap name that matches the Plotly colorscale."""
    if "coverage" in metric_key:
        return "Blues"
    if "ndvi" in metric_key or "integrated" in metric_key:
        return "RdYlGn"
    return "viridis"


def make_metric_overlay_png(
    z: np.ndarray,
    lat: np.ndarray,
    metric_key: str,
    zmin: float | None = None,
    zmax: float | None = None,
    opacity: float = 0.75,
) -> str:
    """
    Render a 2-D metric array as a georeferenced RGBA PNG data URI for
    a dash-leaflet ImageOverlay.  NaN pixels are fully transparent.

    Row 0 of the PNG must be the northernmost row (top of image).
    If lat[0, 0] < lat[-1, 0] (south-to-north storage), the array is flipped.
    """
    cmap = matplotlib.colormaps[_mpl_cmap_for(metric_key)]

    valid = z[~np.isnan(z)]
    if zmin is None:
        zmin = float(valid.min()) if len(valid) else 0.0
    if zmax is None:
        zmax = float(valid.max()) if len(valid) else 1.0
    if zmax <= zmin:
        zmax = zmin + 1e-6

    nan_mask = np.isnan(z)
    norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
    rgba = cmap(norm(z), bytes=True).copy()  # (ny, nx, 4) uint8
    rgba[..., 3] = np.where(nan_mask, 0, int(np.clip(opacity, 0.0, 1.0) * 255))

    # Ensure north-first row order (top of PNG = north)
    if lat[0, 0] < lat[-1, 0]:
        rgba = rgba[::-1, :, :]

    img = PILImage.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def make_colorbar_html(
    metric_key: str,
    zmin: float,
    zmax: float,
) -> str:
    """
    Render a horizontal colorbar as a tiny matplotlib PNG and return HTML
    for display in the Dash sidebar.
    """
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    title = f"{label} ({units})" if units else label

    fig = MplFigure(figsize=(3.2, 0.32))
    ax  = fig.add_axes([0.02, 0.05, 0.96, 0.85])
    matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=matplotlib.colormaps[_mpl_cmap_for(metric_key)],
        norm=mcolors.Normalize(vmin=zmin, vmax=zmax),
        orientation="horizontal",
    ).ax.tick_params(labelsize=6)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=96, bbox_inches="tight", transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    import matplotlib.pyplot as _plt; _plt.close(fig)  # noqa: E702

    return (
        f'<div style="font-size:0.72em;color:#444;margin-bottom:1px">{title}</div>'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:100%;max-width:280px;height:auto"/>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.68em;color:#666">'
        f'<span>{zmin:.4g}</span><span>{zmax:.4g}</span></div>'
    )


# ---------------------------------------------------------------------------
# Time series plot
# ---------------------------------------------------------------------------

def _ndvi_compatible(metric_key: str) -> bool:
    return any(s in metric_key for s in ("ndvi", "integrated"))


def make_timeseries_figure(
    ts,
    smoothed_daily: np.ndarray,
    daily_dates: np.ndarray,
    region_id: str,
    vi_var: str = "NDVI",
    basemap_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    obs_dates = ts.dates[ts.valid_mask].astype("datetime64[ms]").astype(str)
    obs_vi    = ts.raw_vi[ts.valid_mask].tolist()

    daily_date_strs = daily_dates.astype("datetime64[ms]").astype(str)
    smooth_vi = smoothed_daily.tolist()

    raw_trace = go.Scatter(
        x=obs_dates,
        y=obs_vi,
        mode="markers",
        marker=dict(color="#888888", size=4, opacity=0.65),
        name="Raw observations",
        hovertemplate=f"Date: %{{x}}<br>{vi_var}: %{{y:.4f}}<extra></extra>",
    )

    smooth_trace = go.Scatter(
        x=daily_date_strs,
        y=smooth_vi,
        mode="lines",
        line=dict(color="#2ca02c", width=2),
        name="Whittaker smoothed",
        hovertemplate=f"Date: %{{x}}<br>{vi_var}: %{{y:.4f}}<extra></extra>",
        connectgaps=False,
    )

    subtitle = (
        f"Lat {ts.lat:.4f}°, Lon {ts.lon:.4f}° | "
        f"n={int(ts.valid_mask.sum())} valid obs"
    )

    all_values = obs_vi + [v for v in smooth_vi if v is not None and not math.isnan(v)]
    if all_values:
        _dmin, _dmax = min(all_values), max(all_values)
        _pad = max((_dmax - _dmin) * 0.08, 0.02)
        y_range = [_dmin - _pad, _dmax + _pad]
    else:
        _vi_lo, _vi_hi = VI_VALID_RANGE.get(vi_var, (-0.15, 1.05))
        y_range = [_vi_lo - 0.05, _vi_hi + 0.05]

    fig = go.Figure(
        data=[raw_trace, smooth_trace],
        layout=go.Layout(
            title=dict(
                text=(
                    f"{region_id} — Pixel time series"
                    f"<br><sup style='font-size:11px'>{subtitle}</sup>"
                ),
                font=dict(size=13),
            ),
            xaxis=dict(title="Date", showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(
                title=vi_var,
                range=y_range,
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            autosize=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=60, r=20, t=70, b=50),
            uirevision=f"timeseries-{zmin}-{zmax}",
            plot_bgcolor="#f8f8f8",
        ),
    )
    return fig


def make_empty_timeseries_figure() -> go.Figure:
    """Placeholder shown before a pixel is selected."""
    return go.Figure(
        layout=go.Layout(
            title=dict(
                text="Click a pixel on the map to view its time series",
                font=dict(size=13, color="#888888"),
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            autosize=True,
            plot_bgcolor="#f8f8f8",
            paper_bgcolor="#f8f8f8",
            annotations=[
                dict(
                    text="← Select a pixel",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#aaaaaa"),
                )
            ],
        )
    )


# ---------------------------------------------------------------------------
# Annual-cycle and per-metric trend figures
# ---------------------------------------------------------------------------

_ANNUAL_MEAN_METRICS: list[str] = [
    "peak_ndvi_mean",
    "peak_doy_mean",
    "integrated_ndvi_mean",
    "greenup_rate_mean",
    "floor_ndvi_mean",
    "ceiling_ndvi_mean",
    "season_length_mean",
    "n_peaks_mean",
    "peak_separation_mean",
    "relative_peak_amplitude_mean",
    "valley_depth_mean",
]

_ANNUAL_KEY: dict[str, str] = {
    "peak_ndvi_mean":               "peak_ndvi",
    "peak_doy_mean":                "peak_doy",
    "integrated_ndvi_mean":         "integrated",
    "greenup_rate_mean":            "greenup",
    "floor_ndvi_mean":              "floor",
    "ceiling_ndvi_mean":            "ceiling",
    "season_length_mean":           "season_len",
    "n_peaks_mean":                 "n_peaks",
    "peak_separation_mean":         "peak_sep",
    "relative_peak_amplitude_mean": "rel_amp",
    "valley_depth_mean":            "valley",
}

_STD_KEY: dict[str, str] = {
    "peak_ndvi_mean":       "peak_ndvi_std",
    "peak_doy_mean":        "peak_doy_std",
    "integrated_ndvi_mean": "integrated_ndvi_std",
    "greenup_rate_mean":    "greenup_rate_std",
    "season_length_mean":   "season_length_std",
}

_YEAR_PALETTE: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _year_color(idx: int) -> str:
    return _YEAR_PALETTE[idx % len(_YEAR_PALETTE)]


def _short_metric_label(metric_key: str) -> str:
    label, units = METRIC_LABELS.get(metric_key, (metric_key, ""))
    for suffix in (" (mean)", " Mean", "(mean)"):
        label = label.replace(suffix, "")
    label = label.strip()
    return f"{label} [{units}]" if units else label


def make_annual_cycle_figure(
    smoothed: np.ndarray,
    daily_dates: np.ndarray,
    region_id: str,
    vi_var: str = "NDVI",
    basemap_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    dates_dt  = pd.DatetimeIndex(daily_dates.astype("datetime64[ns]"))
    years_arr = dates_dt.year.to_numpy()
    doys_arr  = dates_dt.day_of_year.to_numpy()

    unique_years = sorted(set(years_arr.tolist()))
    traces: list[go.BaseTraceType] = []
    doy_rows: list[pd.DataFrame] = []

    for i_yr, yr in enumerate(unique_years):
        mask   = years_arr == yr
        doy_yr = doys_arr[mask]
        val_yr = smoothed[mask]
        color  = _year_color(i_yr)
        traces.append(go.Scatter(
            x=doy_yr.tolist(),
            y=val_yr.tolist(),
            mode="lines",
            name=str(yr),
            legendgroup=str(yr),
            line=dict(color=color, width=1.5),
            opacity=0.75,
            hovertemplate=f"{yr}, DOY %{{x}}: %{{y:.4f}}<extra></extra>",
        ))
        doy_rows.append(pd.DataFrame({"doy": doy_yr, "ndvi": val_yr}))

    if doy_rows:
        mean_doy = (
            pd.concat(doy_rows)
            .groupby("doy")["ndvi"]
            .mean()
            .reset_index()
        )
        traces.append(go.Scatter(
            x=mean_doy["doy"].tolist(),
            y=mean_doy["ndvi"].tolist(),
            mode="lines",
            name="Mean",
            legendgroup="mean",
            line=dict(color="#000000", width=2.5),
            hovertemplate="Mean, DOY %{x}: %{y:.4f}<extra></extra>",
        ))

    valid_vals = smoothed[~np.isnan(smoothed)]
    if len(valid_vals) > 0:
        _dmin, _dmax = float(valid_vals.min()), float(valid_vals.max())
        _pad = max((_dmax - _dmin) * 0.08, 0.02)
        y_range = [_dmin - _pad, _dmax + _pad]
    else:
        _vi_lo, _vi_hi = VI_VALID_RANGE.get(vi_var, (-0.15, 1.05))
        y_range = [_vi_lo - 0.05, _vi_hi + 0.05]

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=f"{region_id} — Annual {vi_var} Cycles", font=dict(size=13)),
            xaxis=dict(title="Day of Year", showgrid=True, gridcolor="#e0e0e0",
                       range=[1, 366]),
            yaxis=dict(
                title=vi_var,
                range=y_range,
                showgrid=True,
                gridcolor="#e0e0e0",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                        font=dict(size=10)),
            autosize=True,
            margin=dict(l=60, r=20, t=70, b=50),
            uirevision=f"annual_cycle-{zmin}-{zmax}",
            plot_bgcolor="#f8f8f8",
        ),
    )


def make_phenology_scatter_figure(
    ts,
    vi_var: str,
    lam: float,
    region_id: str,
    basemap_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    valid_dates = ts.dates[ts.valid_mask]
    vi_vals = ts.raw_vi[ts.valid_mask].astype(np.float64)

    if len(vi_vals) == 0:
        return make_empty_timeseries_figure()

    dates_dt = pd.DatetimeIndex(valid_dates.astype("datetime64[ns]"))
    doy_arr  = dates_dt.day_of_year.to_numpy()
    year_arr = dates_dt.year.to_numpy()

    scatter = go.Scatter(
        x=doy_arr.tolist(),
        y=vi_vals.tolist(),
        mode="markers",
        marker=dict(
            color=year_arr.tolist(),
            colorscale="Plasma",
            colorbar=dict(title="Year", thickness=14, len=0.7),
            size=4,
            opacity=0.55,
        ),
        name="Observations",
        hovertemplate="DOY %{x}, %{marker.color}: %{y:.4f}<extra></extra>",
    )

    daily_y = np.zeros(366, dtype=np.float64)
    daily_w = np.zeros(366, dtype=np.float64)
    day_count = np.zeros(366, dtype=np.float64)
    idx = doy_arr - 1
    np.add.at(daily_y, idx, vi_vals)
    np.add.at(day_count, idx, 1.0)
    hit = day_count > 0
    daily_y[hit] /= day_count[hit]
    daily_w[hit] = 1.0

    lam_DTD = _build_whittaker_system(366, lam)
    smoothed_doy = _whittaker_smooth_pixel(daily_y, daily_w, lam_DTD)

    mean_line = go.Scatter(
        x=list(range(1, 367)),
        y=smoothed_doy.tolist(),
        mode="lines",
        name="Mean fit",
        line=dict(color="#000000", width=2.5),
        hovertemplate="Mean fit, DOY %{x}: %{y:.4f}<extra></extra>",
    )

    _dmin, _dmax = float(vi_vals.min()), float(vi_vals.max())
    _pad = max((_dmax - _dmin) * 0.08, 0.02)
    y_range = [_dmin - _pad, _dmax + _pad]

    return go.Figure(
        data=[scatter, mean_line],
        layout=go.Layout(
            title=dict(
                text=f"{region_id} — {vi_var} Phenology Scatter",
                font=dict(size=13),
            ),
            xaxis=dict(title="Day of Year", range=[1, 366],
                       showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(title=vi_var, range=y_range,
                       showgrid=True, gridcolor="#e0e0e0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                        font=dict(size=10)),
            autosize=True,
            margin=dict(l=60, r=20, t=70, b=50),
            uirevision=f"phenology_scatter-{zmin}-{zmax}",
            plot_bgcolor="#f8f8f8",
        ),
    )


def make_metrics_annual_figure(
    valid_years: list[int],
    annual_data: dict[str, list],
    metrics: dict[str, float],
    region_id: str,
) -> go.Figure:
    mean_keys = [k for k in _ANNUAL_MEAN_METRICS if k in _ANNUAL_KEY]
    n_plots   = len(mean_keys)
    n_cols    = 2
    n_rows    = math.ceil(n_plots / n_cols)

    subplot_titles = [_short_metric_label(k) for k in mean_keys]
    subplot_titles += [""] * (n_rows * n_cols - n_plots)

    fig_base = _make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=max(0.06, 0.55 / max(n_rows, 1)),
        horizontal_spacing=0.14,
    )

    x_span = (
        [min(valid_years) - 0.5, max(valid_years) + 0.5]
        if valid_years else [0.5, 1.5]
    )

    for plot_idx, metric_key in enumerate(mean_keys):
        row  = plot_idx // n_cols + 1
        col  = plot_idx %  n_cols + 1
        akey = _ANNUAL_KEY[metric_key]
        yr_vals = annual_data.get(akey, [])

        for i_yr, yr in enumerate(valid_years):
            val = yr_vals[i_yr] if i_yr < len(yr_vals) else float("nan")
            y_val = None if (isinstance(val, float) and np.isnan(val)) else val
            fig_base.add_trace(
                go.Scatter(
                    x=[yr],
                    y=[y_val],
                    mode="markers",
                    name=str(yr),
                    legendgroup=str(yr),
                    showlegend=(plot_idx == 0),
                    marker=dict(color=_year_color(i_yr), size=10, symbol="circle"),
                    hovertemplate=f"{yr}: %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )

        mean_val = metrics.get(metric_key)
        std_key  = _STD_KEY.get(metric_key)
        std_val  = metrics.get(std_key) if std_key else None

        def _valid(v) -> bool:
            return v is not None and not np.isnan(float(v))

        if _valid(mean_val):
            if std_key and _valid(std_val):
                hi = float(mean_val) + float(std_val)
                lo = float(mean_val) - float(std_val)
                fig_base.add_trace(
                    go.Scatter(
                        x=x_span + x_span[::-1],
                        y=[hi, hi, lo, lo],
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )
            fig_base.add_trace(
                go.Scatter(
                    x=x_span,
                    y=[float(mean_val), float(mean_val)],
                    mode="lines",
                    name="Mean",
                    legendgroup="mean",
                    showlegend=(plot_idx == 0),
                    line=dict(color="#000000", width=1.8, dash="dash"),
                    hovertemplate=f"Mean: {float(mean_val):.4f}<extra></extra>",
                ),
                row=row, col=col,
            )

    fig = go.Figure(fig_base)
    fig.update_layout(
        title=dict(text=f"{region_id} — Annual Metric Trends", font=dict(size=13),
                   y=0.99, yanchor="top"),
        height=n_rows * 260 + 120,
        legend=dict(orientation="h", yanchor="top", y=0.97, x=0,
                    font=dict(size=10), traceorder="normal"),
        plot_bgcolor="#f8f8f8",
        paper_bgcolor="white",
        margin=dict(l=65, r=25, t=110, b=50),
        uirevision="metrics_annual",
    )
    # Reduce subplot title font so it fits within the inter-row gap
    fig.update_annotations(font_size=11)
    fig.update_xaxes(
        tickformat="d",
        tickmode="array",
        tickvals=valid_years if valid_years else [],
        showgrid=True,
        gridcolor="#e0e0e0",
        tickfont=dict(size=10),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", tickfont=dict(size=9))
    return fig


# ---------------------------------------------------------------------------
# Pixel metrics sidebar table
# ---------------------------------------------------------------------------

def _metric_swatch_html(val: float, metric_key: str, zmin: float, zmax: float) -> str:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return ""
    if np.isnan(fval) or zmax <= zmin:
        return ""
    t = max(0.0, min(1.0, (fval - zmin) / (zmax - zmin)))
    r, g, b, _ = matplotlib.colormaps[_mpl_cmap_for(metric_key)](t)
    hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return (
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'background:{hex_color};border:1px solid #888;'
        f'margin-right:4px;vertical-align:middle"></span>'
    )


def make_metrics_table(
    metrics: dict[str, float],
    selected_metric: str | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> str:
    rows: list[str] = []
    for group_name, metric_keys in METRIC_GROUPS.items():
        rows.append(
            f'<tr><td colspan="3" style="'
            f'font-weight:bold;background:#e8f0e8;'
            f'padding:4px 6px;font-size:0.8em;letter-spacing:0.05em;">'
            f'{group_name.upper()}</td></tr>'
        )
        for key in metric_keys:
            val = metrics.get(key, np.nan)
            label, units = METRIC_LABELS.get(key, (key, ""))
            val_str = f"{val:.4f}" if (val is not None and not np.isnan(float(val) if val is not None else float("nan"))) else "N/A"
            is_selected = key == selected_metric
            highlight = "font-weight:bold;background:#fff9c4;" if is_selected else ""

            swatch = ""
            if is_selected and zmin is not None and zmax is not None and val_str != "N/A":
                swatch = _metric_swatch_html(val, key, zmin, zmax)

            rows.append(
                f'<tr style="{highlight}">'
                f'<td style="padding:2px 6px;font-size:0.78em">{label}</td>'
                f'<td style="padding:2px 6px;font-size:0.78em;text-align:right">'
                f'{swatch}{val_str}</td>'
                f'<td style="padding:2px 4px;font-size:0.72em;color:#666">{units}</td>'
                f'</tr>'
            )

    return (
        '<table style="width:100%;border-collapse:collapse;'
        'font-family:monospace">'
        + "".join(rows)
        + "</table>"
    )
