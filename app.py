"""
app.py — VI Phenology Dashboard (Plotly Dash / Dash Enterprise)

Run with:
    python app.py                       # http://127.0.0.1:8050

Dash Enterprise deployment:
    gunicorn app:server                 # Procfile: web: gunicorn app:server

Environment variables:
    VI_DATACUBE_ROOT   — override the root data directory
    VI_PHENOLOGY_SRC   — override path to VI_Phenology/src

Architecture notes
------------------
- All state is managed via dcc.Store components (JSON-serialisable dicts).
  Shiny's reactive.Value / reactive.Calc → dcc.Store + @callback.
- Pixel selection is driven by dl.Map clickData (lat/lng), not Python-side
  on_click() callbacks (which required go.FigureWidget + shinywidgets).
- go.Figure is used throughout — go.FigureWidget is not needed in Dash.
- The pixel-result Store is computed once per selection and consumed by all
  four chart callbacks, avoiding redundant Whittaker solves.
- server = app.server exposes the Flask/WSGI app for gunicorn.
"""

from __future__ import annotations

# Suppress grpcio BlockingIOError spam that occurs when gunicorn sync workers
# are used alongside gcsfs/google-cloud-storage (which pulls in gRPC).
import os as _os
_os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")
_os.environ.setdefault("GRPC_POLL_STRATEGY", "epoll1")

import base64
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dash import (
    Dash, Input, Output, State,
    callback, ctx, dcc, html, no_update,
)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_leaflet as dl

from config import (
    ALL_19_METRICS,
    BASEMAP_MAX_DIM,
    BASEMAP_MAX_DIM_PRECOMPUTED,
    DEFAULT_VI_VAR,
    FAST_BASEMAP_METRICS,
    LAMBDA_DEFAULT,
    LAMBDA_MAX,
    LAMBDA_MIN,
    LAMBDA_STEP,
    METRIC_GROUPS,
    METRIC_LABELS,
    PIXEL_METRIC_CONFIG,
    NONNEGATIVE_METRICS,
    SHAPEFILE_LABEL_FIELDS,
    SHAPEFILE_PATHS,
    VI_VALID_RANGE,
)
from modules.datacube_io import (
    PixelTimeSeries,
    RegionPaths,
    basemap_cache_path,
    build_date_cache,
    build_date_cache_from_dates,
    click_to_array_index,
    compute_basemap_metric,
    detect_crs_epsg,
    discover_regions,
    extract_pixel_timeseries,
    get_dataset,
    load_basemap_cache,
    load_metrics_for_basemap,
    utm_to_latlon,
)
from modules.phenology_metrics import (
    compute_pixel_with_annual,
    source_available,
    source_error,
)
from modules.visualization import (
    LEAFLET_TILE_SERVICES,
    get_map_center_and_zoom,
    get_overlay_url_and_bounds,
    get_shapefile_geojson_data,
    get_tile_layer_props,
    make_annual_cycle_figure,
    make_colorbar_component,
    make_empty_timeseries_figure,
    make_metrics_annual_figure,
    make_metrics_table,
    make_phenology_scatter_figure,
    make_timeseries_figure,
)


# ---------------------------------------------------------------------------
# Startup: region discovery
# ---------------------------------------------------------------------------

_STARTUP_ERROR: str | None = None
try:
    ALL_REGIONS: dict[str, RegionPaths] = discover_regions()

    def _region_label(p: RegionPaths) -> str:
        fmt   = "zarr" if p.zarr_path else "nc"
        extra = " + metrics" if p.metrics_path else ""
        return f"{p.region_id}  [{fmt}{extra}]"

    REGION_OPTIONS = [
        {"label": _region_label(v), "value": k}
        for k, v in ALL_REGIONS.items()
    ]
    DEFAULT_REGION = next(iter(ALL_REGIONS), None)
except FileNotFoundError as _e:
    ALL_REGIONS    = {}
    REGION_OPTIONS = []
    DEFAULT_REGION = None
    _STARTUP_ERROR = str(_e)

# Tile service dropdown options
TILE_OPTIONS = [
    {"label": v["label"], "value": k}
    for k, v in LEAFLET_TILE_SERVICES.items()
] + [{"label": "No Basemap", "value": "none"}]

# Basemap metric options (flat list with disabled group headers)
_fast_group  = [{"label": lbl, "value": FAST_BASEMAP_METRICS[lbl]}
                for lbl in FAST_BASEMAP_METRICS]
_pheno_group = [{"label": METRIC_LABELS[k][0], "value": k}
                for k in ALL_19_METRICS if k in METRIC_LABELS]
BASEMAP_OPTIONS = (
    [{"label": "── Quick metrics ──", "value": "__hdr1__", "disabled": True}]
    + _fast_group
    + [{"label": "── Phenology metrics (precomputed .nc required) ──",
        "value": "__hdr2__", "disabled": True}]
    + _pheno_group
)
_DEFAULT_BASEMAP_KEY: str = FAST_BASEMAP_METRICS["Mean VI"]

# Shapefile overlay list (static — not reactive)
_SHAPEFILE_LIST: list[str] = SHAPEFILE_PATHS.split() if SHAPEFILE_PATHS else []
_SHAPEFILE_OPTIONS = [
    {"label": Path(p).stem, "value": str(i)}
    for i, p in enumerate(_SHAPEFILE_LIST)
]
# Load shapefile GeoJSON at startup (avoids per-request file I/O)
_SHAPEFILE_GEOJSON: list[dict | None] = [
    get_shapefile_geojson_data(p) for p in _SHAPEFILE_LIST
]


def _flatten_geojson_coords(geom: dict) -> list[tuple[float, float]]:
    """Recursively extract all (lon, lat) pairs from a GeoJSON geometry."""
    result: list[tuple[float, float]] = []
    def _r(c):
        if not c:
            return
        if isinstance(c[0], (int, float)):
            result.append((float(c[0]), float(c[1])))
        else:
            for sub in c:
                _r(sub)
    _r(geom.get("coordinates", []))
    return result


def _compute_initial_map_view() -> tuple[list[float], int]:
    """Return (center, zoom) from the combined extent of all loaded shapefiles."""
    all_lons: list[float] = []
    all_lats: list[float] = []
    for gj in _SHAPEFILE_GEOJSON:
        if gj is None:
            continue
        for feat in gj.get("features", []):
            for lon, lat in _flatten_geojson_coords(feat.get("geometry", {})):
                all_lons.append(lon)
                all_lats.append(lat)
    if not all_lons:
        return [-27.0, 26.0], 6
    return get_map_center_and_zoom(np.array(all_lons), np.array(all_lats))


_INITIAL_CENTER, _INITIAL_ZOOM = _compute_initial_map_view()

_NORTH_ARROW_SRC = "data:image/svg+xml;base64," + base64.b64encode((
    '<svg viewBox="0 0 24 36" width="24" height="36" xmlns="http://www.w3.org/2000/svg">'
    '<polygon points="12,2 7,22 12,18" fill="rgba(91,227,255,0.85)"/>'
    '<polygon points="12,2 12,18 17,22" fill="rgba(255,255,255,0.18)"/>'
    '<text x="12" y="34" text-anchor="middle" font-size="10"'
    ' font-family="Space Mono,monospace" fill="rgba(91,227,255,0.9)" font-weight="bold">N</text>'
    '</svg>'
).encode()).decode()
_FAST_METRIC_KEYS: frozenset[str] = frozenset(FAST_BASEMAP_METRICS.values())


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
)
server = app.server   # Dash Enterprise / gunicorn WSGI entry point


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_SIDEBAR_STYLE = {
    "background":  "#07101a",
    "borderRight": "1px solid rgba(91,227,255,0.13)",
    "padding":     "18px 14px 16px",
    "overflowY":   "auto",
    "height":      "100vh",
    "fontSize":    "0.9em",
    "fontFamily":  "'Inter', sans-serif",
    "color":       "rgba(255,255,255,0.92)",
}

_MAP_STYLE = {
    "height":         "100%",   # wrapper (#map-wrapper) drives height via CSS var
    "width":          "100%",
    "imageRendering": "pixelated",
}


def _initial_map_children() -> list:
    """Build the initial dash-leaflet children list (before any data loads)."""
    children: list = [
        dl.TileLayer(
            id="tile-layer",
            url=LEAFLET_TILE_SERVICES["World_Imagery"]["url"],
            attribution=LEAFLET_TILE_SERVICES["World_Imagery"]["attribution"],
            maxZoom=18,
        ),
        dl.ImageOverlay(
            id="metric-overlay",
            url="",
            bounds=[[-90, -180], [90, 180]],
            opacity=0.75,  # updated by callback; matches slider default
        ),
        dl.Marker(
            id="pixel-marker",
            position=[0, 0],
            opacity=0,
        ),
        dl.ScaleControl(position="bottomleft", imperial=False),
    ]
    # Static shapefile layers — visibility toggled by callback, not re-created
    for i, gj_data in enumerate(_SHAPEFILE_GEOJSON):
        if gj_data is not None:
            children.append(
                dl.GeoJSON(
                    id=f"shapefile-layer-{i}",
                    data=gj_data,
                    style={
                        "color": "#ffffff",
                        "weight": 2,
                        "fillOpacity": 0.0,
                    },
                    hoverStyle={"color": "#ffff00", "weight": 3},
                    zoomToBoundsOnClick=False,
                )
            )
    return children


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_source_warning = (
    dbc.Alert(
        [html.B("⚠ VI_Phenology source not found. "),
         "Per-pixel metrics unavailable. ",
         f"Error: {source_error()}"],
        color="warning",
        style={
            "fontSize": "0.82em", "padding": "6px", "marginTop": "6px",
            "background": "rgba(255,138,61,0.10)",
            "border": "1px solid rgba(255,138,61,0.3)",
            "color": "#ff8a3d",
            "borderRadius": "2px",
        },
    )
    if not source_available() else None
)

_sidebar_content = [
    # Product label + font-size controls (top-right)
    html.Div([
        html.Div("BioSCape · ESA / NASA", style={
            "fontFamily": "'Space Mono', monospace",
            "fontSize": "9px",
            "letterSpacing": "0.12em",
            "textTransform": "uppercase",
            "color": "#5be3ff",
        }),
        html.Div([
            html.Button("A-", id="font-size-dec", n_clicks=0, className="font-size-btn",
                        title="Decrease font size"),
            html.Button("A+", id="font-size-inc", n_clicks=0, className="font-size-btn",
                        title="Increase font size"),
        ], style={"display": "flex", "gap": "4px", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "3px"}),
    html.H5("Phenology Explorer", style={
        "fontFamily": "'Space Mono', monospace",
        "fontSize": "14px",
        "fontWeight": "700",
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "color": "rgba(255,255,255,0.92)",
        "marginBottom": "2px",
        "lineHeight": "1.2",
    }),
    html.Div(
        [
            "Stephen Conklin · ",
            html.A(
                "GitHub",
                href="https://github.com/stephenconklin/VI_Phenology_Plotly_Cloud_Dashboard",
                target="_blank",
                style={"color": "rgba(91,227,255,0.45)", "textDecoration": "none"},
            ),
        ],
        style={"fontSize": "10px", "color": "rgba(255,255,255,0.28)", "marginBottom": "4px", "letterSpacing": "0.04em"},
    ),
    html.Hr(style={"margin": "8px 0", "borderColor": "rgba(91,227,255,0.13)"}),

    html.Label("Region", className="form-label fw-semibold"),
    dcc.Dropdown(id="region-dropdown", options=REGION_OPTIONS,
                 value=DEFAULT_REGION, clearable=False, searchable=False,
                 style={"marginBottom": "10px"},
                 className="hud-dropdown"),

    html.Label("Basemap metric", className="form-label fw-semibold"),
    dcc.Dropdown(id="metric-select", options=BASEMAP_OPTIONS,
                 value=_DEFAULT_BASEMAP_KEY, clearable=False, searchable=False,
                 style={"marginBottom": "10px"},
                 className="hud-dropdown"),

    html.Label("Basemap style", className="form-label fw-semibold"),
    dcc.Dropdown(id="basemap-style", options=TILE_OPTIONS,
                 value="World_Imagery", clearable=False, searchable=False,
                 style={"marginBottom": "10px"},
                 className="hud-dropdown"),

    html.Label("Metric layer opacity", className="form-label fw-semibold"),
    dcc.Slider(id="opacity-slider", min=0.0, max=1.0, value=0.75, step=0.05,
               marks={0: "0", 0.5: "0.5", 1: "1"},
               tooltip={"placement": "bottom"}),

    html.Label("Data range", className="form-label fw-semibold",
               style={"marginTop": "8px"}),
    dcc.Dropdown(
        id="colorscale-range",
        options=[
            {"label": "Full range (min – max)", "value": "full"},
            {"label": "Mean ± 3 SD  (~99.7 %)", "value": "3sd"},
            {"label": "Mean ± 2 SD  (~95 %)",   "value": "2sd"},
            {"label": "Mean ± 1 SD  (~68 %)",   "value": "1sd"},
        ],
        value="3sd", clearable=False, searchable=False,
        style={"marginBottom": "10px"},
        className="hud-dropdown",
    ),

    html.Label("Year range", className="form-label fw-semibold"),
    dcc.RangeSlider(
        id="year-range",
        min=2000, max=2030, value=[2000, 2030], step=1,
        marks={}, allowCross=True, pushable=0,
        tooltip={"placement": "bottom", "always_visible": True},
    ),

    html.Hr(style={"margin": "8px 0", "borderColor": "rgba(91,227,255,0.13)"}),

    html.Label("Whittaker λ (smoothing)", className="form-label fw-semibold"),
    dcc.Slider(
        id="lambda-slider",
        min=LAMBDA_MIN, max=LAMBDA_MAX, value=LAMBDA_DEFAULT, step=LAMBDA_STEP,
        marks={LAMBDA_MIN: str(LAMBDA_MIN), LAMBDA_MAX: str(LAMBDA_MAX)},
        tooltip={"placement": "bottom"},
    ),

    # Shapefile toggles (only rendered when layers exist)
    (html.Div([
        html.Label("Overlay layers", className="form-label fw-semibold",
                   style={"marginTop": "8px"}),
        dcc.Checklist(
            id="shapefile-visible",
            options=_SHAPEFILE_OPTIONS,
            value=[o["value"] for o in _SHAPEFILE_OPTIONS],
            labelStyle={"display": "block"},
        ),
    ]) if _SHAPEFILE_OPTIONS else html.Div(id="shapefile-visible")),

    html.Hr(style={"margin": "8px 0", "borderColor": "rgba(91,227,255,0.13)"}),

    html.Div(id="pixel-info"),

    html.Hr(style={"margin": "8px 0", "borderColor": "rgba(91,227,255,0.13)"}),

    html.H6("Pixel Phenology Metrics", style={"marginBottom": "2px"}),
    html.Div(id="metrics-table"),

    _source_warning or html.Div(),

    # Hidden state stores
    dcc.Store(id="dataset-info"),
    dcc.Store(id="basemap-info"),
    dcc.Store(id="selected-pixel"),
    dcc.Store(id="pixel-result"),
    dcc.Store(id="shapefile-region-locked"),   # region ID locked after shapefile click
    dcc.Store(id="font-scale-store", data=1.0),
]

_sidebar = dbc.Col(_sidebar_content, width=3, style=_SIDEBAR_STYLE)

_main_panel = dbc.Col(
    [
        html.Div(
            [
                dl.Map(
                    id="main-map",
                    center=_INITIAL_CENTER,
                    zoom=_INITIAL_ZOOM,
                    style=_MAP_STYLE,
                    children=_initial_map_children(),
                ),
                html.Div(id="colorbar-div"),
                html.Div(
                    html.Img(src=_NORTH_ARROW_SRC,
                             style={"display": "block", "margin": "0 auto",
                                    "width": "24px", "height": "36px"}),
                    id="north-arrow",
                ),
            ],
            id="map-wrapper",
            style={"position": "relative"},
        ),
        html.Div(id="resize-divider"),
        html.Div(
            dbc.Tabs(
                [
                    dbc.Tab(
                        dcc.Graph(id="timeseries-chart",
                                  figure=make_empty_timeseries_figure(),
                                  style={"height": "100%"},
                                  config={"responsive": True}),
                        label="Raw VI",
                    ),
                    dbc.Tab(
                        dcc.Graph(id="annual-cycle-chart",
                                  figure=make_empty_timeseries_figure(),
                                  style={"height": "100%"},
                                  config={"responsive": True}),
                        label="Annual Cycles",
                    ),
                    dbc.Tab(
                        html.Div(
                            dcc.Graph(id="metrics-annual-chart",
                                      figure=make_empty_timeseries_figure(),
                                      config={"responsive": False}),
                            id="metrics-annual-chart-wrapper",
                        ),
                        label="Metric Trends",
                    ),
                    dbc.Tab(
                        dcc.Graph(id="phenology-scatter-chart",
                                  figure=make_empty_timeseries_figure(),
                                  style={"height": "100%"},
                                  config={"responsive": True}),
                        label="Phenology Scatter",
                    ),
                ],
            ),
            id="charts-wrapper",
        ),
    ],
    width=9,
    id="main-panel-col",
)

if _STARTUP_ERROR:
    app.layout = dbc.Container(
        dbc.Alert(
            [
                html.H4("⚠ Data directory not found"),
                html.P(_STARTUP_ERROR),
                html.P(["Set the ", html.Code("VI_DATACUBE_ROOT"),
                        " environment variable, or edit ", html.Code("config.py"), "."]),
            ],
            color="danger",
        ),
        fluid=True,
    )
else:
    app.layout = dbc.Container(
        [
            dbc.Row([_sidebar, _main_panel], className="g-0", style={"height": "100vh"}),
            dbc.Toast(
                id="no-data-toast",
                header="No data available",
                children="",
                is_open=False,
                dismissable=True,
                duration=4000,
                icon="warning",
                style={
                    "position": "fixed", "top": 10, "right": 10,
                    "width": 320, "zIndex": 9999,
                    "background": "#0d1c2e",
                    "border": "1px solid rgba(255,138,61,0.35)",
                    "color": "#ff8a3d",
                    "borderRadius": "2px",
                },
            ),
        ],
        fluid=True,
        style={"padding": 0, "overflow": "hidden", "background": "#060c12"},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_colorscale_limits(
    z: np.ndarray,
    sel: str,
    metric: str = "",
) -> tuple[float | None, float | None]:
    """Return (zmin, zmax) based on the SD-clipping selection.

    For metrics in NONNEGATIVE_METRICS, zmin is floored to 0 — SD-clipping
    can otherwise produce a negative lower bound (e.g. peak_doy_std with
    mean=8 days and sd=10 days at 2σ → zmin = -12 days) even though the
    metric value cannot be negative by construction.
    """
    if sel == "full":
        return None, None
    z_mean = float(np.nanmean(z))
    z_std  = float(np.nanstd(z))
    n_sd   = {"1sd": 1, "2sd": 2, "3sd": 3}.get(sel, 2)
    zmin = z_mean - n_sd * z_std
    zmax = z_mean + n_sd * z_std
    if metric in NONNEGATIVE_METRICS:
        zmin = max(0.0, zmin)
    return zmin, zmax


def _safe_float(v) -> float | None:
    """Convert to float; return None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _serialize_pixel_result(
    result,
    narrowed: PixelTimeSeries,
    full_ts: PixelTimeSeries,
    region: str,
    vi_var: str,
) -> dict:
    """Serialise compute_pixel_with_annual() output to a JSON-safe dict."""
    metrics, valid_years, annual_data, smoothed_daily, daily_dates = result

    def _arr(arr):
        return [_safe_float(v) for v in arr.tolist()]

    n_total    = len(full_ts.valid_mask)
    n_valid    = int(full_ts.valid_mask.sum())
    n_in_range = int(narrowed.valid_mask.sum())

    return {
        "metrics":        {k: _safe_float(v) for k, v in metrics.items()},
        "valid_years":    valid_years,
        "annual_data":    {k: [_safe_float(v) for v in vals]
                           for k, vals in annual_data.items()},
        "smoothed_daily": _arr(smoothed_daily),
        "daily_dates":    daily_dates.astype("datetime64[ms]").astype(str).tolist(),
        # Already-filtered observations for chart callbacks
        "obs_dates":      narrowed.dates[narrowed.valid_mask]
                          .astype("datetime64[ms]").astype(str).tolist(),
        "obs_vi":         narrowed.raw_vi[narrowed.valid_mask].tolist(),
        "ts_lat":         float(narrowed.lat),
        "ts_lon":         float(narrowed.lon),
        "ts_n_valid":     n_valid,
        "ts_n_total":     n_total,
        "ts_n_in_range":  n_in_range,
        "vi_var":         vi_var,
        "region_id":      region,
    }


def _restore_nan(d: dict) -> dict:
    """Replace None → float('nan') in a metrics dict."""
    return {k: float("nan") if v is None else float(v) for k, v in d.items()}


def _restore_annual_nan(d: dict) -> dict:
    """Replace None → float('nan') in annual_data lists."""
    return {k: [float("nan") if v is None else v for v in vals]
            for k, vals in d.items()}


# ---------------------------------------------------------------------------
# Clientside callback: font-size scale
# ---------------------------------------------------------------------------
_FONT_SCALES = [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]

app.clientside_callback(
    """
    function(n_dec, n_inc, current_scale) {
        const scales = [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5];
        let idx = scales.indexOf(current_scale);
        if (idx === -1) idx = 2;
        const trig = dash_clientside.callback_context.triggered;
        if (trig && trig.length > 0) {
            const prop = trig[0].prop_id;
            if (prop.includes('font-size-dec')) idx = Math.max(0, idx - 1);
            if (prop.includes('font-size-inc')) idx = Math.min(scales.length - 1, idx + 1);
        }
        const newScale = scales[idx];
        document.documentElement.style.setProperty('--fs-scale', newScale);
        return newScale;
    }
    """,
    Output("font-scale-store", "data"),
    Input("font-size-dec", "n_clicks"),
    Input("font-size-inc", "n_clicks"),
    State("font-scale-store", "data"),
    prevent_initial_call=True,
)

# Relay drag_value → value so the server callback fires even when Plotly
# fails to emit a value change when both handles land on the same integer.
app.clientside_callback(
    "function(v) { return v || window.dash_clientside.no_update; }",
    Output("year-range", "value", allow_duplicate=True),
    Input("year-range", "drag_value"),
    prevent_initial_call=True,
)

# ---------------------------------------------------------------------------
# Callbacks: region / dataset
# ---------------------------------------------------------------------------

@callback(
    Output("dataset-info", "data"),
    Input("region-dropdown", "value"),
)
def update_dataset_info(region: str):
    if not region or region not in ALL_REGIONS:
        raise PreventUpdate
    paths = ALL_REGIONS[region]
    ds    = get_dataset(paths)
    dc    = build_date_cache(ds)
    years = dc["years"]
    return {"year_min": int(years.min()), "year_max": int(years.max())}


@callback(
    Output("year-range", "min"),
    Output("year-range", "max"),
    Output("year-range", "value"),
    Input("dataset-info", "data"),
    prevent_initial_call=True,
)
def update_year_slider(dataset_info: dict):
    if dataset_info is None:
        raise PreventUpdate
    yr_min = dataset_info["year_min"]
    yr_max = dataset_info["year_max"]
    return yr_min, yr_max, [yr_min, yr_max]




# ---------------------------------------------------------------------------
# Callbacks: basemap / map state
# ---------------------------------------------------------------------------

@callback(
    Output("metric-overlay", "url"),
    Output("metric-overlay", "bounds"),
    Output("metric-overlay", "opacity"),
    Output("tile-layer", "url"),
    Output("tile-layer", "attribution"),
    Output("tile-layer", "maxZoom"),
    Output("colorbar-div", "children"),
    Output("basemap-info", "data"),
    Output("main-map", "viewport"),
    Input("region-dropdown", "value"),
    Input("metric-select", "value"),
    Input("basemap-style", "value"),
    Input("opacity-slider", "value"),
    Input("colorscale-range", "value"),
)
def update_basemap(region, metric, basemap_style, opacity, colorscale_range):
    if not region or region not in ALL_REGIONS:
        raise PreventUpdate

    metric = metric or _DEFAULT_BASEMAP_KEY
    paths  = ALL_REGIONS[region]

    # --- Load basemap array (fast-path: precomputed → disk cache → Dask) ---
    z = lon = lat = None

    if paths.metrics_path is not None and metric in ALL_19_METRICS:
        try:
            z, lon, lat = load_metrics_for_basemap(paths.metrics_path, metric)
        except Exception:
            pass

    if z is None:
        effective_metric = metric if metric in _FAST_METRIC_KEYS else "peak_ndvi_mean"
        data_path        = paths.zarr_path or paths.nc_path
        # Try pre-computed _d2000.npz first (generated offline, high-res),
        # then fall back to the on-the-fly _d500.npz cache.
        hit = load_basemap_cache(
            basemap_cache_path(data_path, effective_metric, BASEMAP_MAX_DIM_PRECOMPUTED)
        )
        if hit is None:
            hit = load_basemap_cache(
                basemap_cache_path(data_path, effective_metric, BASEMAP_MAX_DIM)
            )
        if hit is None:
            ds          = get_dataset(paths)
            z, lon, lat = compute_basemap_metric(ds, effective_metric, vi_var=paths.vi_var)
        else:
            z, lon, lat = hit

    # Zero data-coverage pixels are outside the datacube extent and should be
    # transparent, not rendered as the low end of the colorscale.
    if metric == "data_coverage":
        z[z == 0.0] = np.nan

    # --- Colorscale limits ---
    zmin, zmax = _compute_colorscale_limits(z, colorscale_range or "3sd", metric)

    # --- Overlay PNG ---
    overlay_url, bounds = get_overlay_url_and_bounds(
        z, lat, lon, metric, float(opacity if opacity is not None else 0.75), zmin, zmax
    )

    # --- Tile layer ---
    if basemap_style == "none":
        tile_props = {"url": "", "attribution": "", "maxZoom": 18}
    else:
        tile_props = get_tile_layer_props(basemap_style or "World_Imagery")

    # --- Colorbar ---
    valid  = z[~np.isnan(z)]
    cb_min = float(valid.min()) if zmin is None else zmin
    cb_max = float(valid.max()) if zmax is None else zmax
    colorbar = make_colorbar_component(metric, cb_min, cb_max) if len(valid) > 0 else None

    # --- Store ---
    basemap_info = {
        "zmin":       _safe_float(zmin),
        "zmax":       _safe_float(zmax),
        "metric_key": metric,
        "lat_min":    float(lat.min()),
        "lat_max":    float(lat.max()),
        "lon_min":    float(lon.min()),
        "lon_max":    float(lon.max()),
        "vi_var":     paths.vi_var,
    }

    # Only fit the map to the region when the region changes
    if ctx.triggered_id == "region-dropdown":
        viewport = dict(
            bounds=[
                [float(lat.min()), float(lon.min())],
                [float(lat.max()), float(lon.max())],
            ],
            transition="flyToBounds",
        )
    else:
        viewport = no_update

    return (
        overlay_url,
        bounds,
        float(opacity if opacity is not None else 0.75),
        tile_props["url"],
        tile_props["attribution"],
        tile_props["maxZoom"],
        colorbar,
        basemap_info,
        viewport,
    )


# ---------------------------------------------------------------------------
# Callbacks: pixel selection
# ---------------------------------------------------------------------------

@callback(
    Output("selected-pixel", "data"),
    Output("pixel-marker", "position"),
    Output("pixel-marker", "opacity"),
    Input("main-map", "clickData"),
    Input("region-dropdown", "value"),
    State("basemap-info", "data"),
    prevent_initial_call=True,
)
def update_selected_pixel(clickData, region, basemap_info):
    # Reset marker when region changes
    if ctx.triggered_id == "region-dropdown":
        return None, [0, 0], 0

    if clickData is None or basemap_info is None:
        raise PreventUpdate

    latlng    = clickData.get("latlng", {})
    click_lat = float(latlng.get("lat", 0))
    click_lon = float(latlng.get("lng", 0))

    # Ignore clicks outside the current datacube extent
    if not (
        basemap_info["lon_min"] <= click_lon <= basemap_info["lon_max"]
        and basemap_info["lat_min"] <= click_lat <= basemap_info["lat_max"]
    ):
        raise PreventUpdate

    paths  = ALL_REGIONS[region]
    ds     = get_dataset(paths)
    yi, xi = click_to_array_index(click_lon, click_lat, ds)

    pixel_data = {
        "region": region,
        "yi":     int(yi),
        "xi":     int(xi),
        "lon":    click_lon,
        "lat":    click_lat,
    }
    return pixel_data, [click_lat, click_lon], 1


# ---------------------------------------------------------------------------
# Callbacks: pixel computation (Whittaker solve)
# ---------------------------------------------------------------------------

@callback(
    Output("pixel-result", "data"),
    Input("selected-pixel", "data"),
    Input("year-range", "value"),
    Input("basemap-info", "data"),
    Input("lambda-slider", "value"),
    State("region-dropdown", "value"),
)
def compute_pixel_result(
    selected_pixel, year_range,
    basemap_info, lambda_val, region,
):
    if selected_pixel is None or selected_pixel.get("region") != region:
        return None

    paths = ALL_REGIONS.get(region)
    if paths is None:
        return None

    yi, xi = selected_pixel["yi"], selected_pixel["xi"]

    try:
        ts = extract_pixel_timeseries(
            paths.nc_path, yi, xi,
            vi_var=paths.vi_var,
            zarr_path=paths.zarr_path,
        )
    except Exception:
        return None

    # --- Apply filters (narrowed_timeseries logic from Shiny version) ---
    # Only apply colorscale limits as VI filters when the basemap is showing
    # a VI-range metric (the four quick metrics). Phenology metrics (Peak DOY,
    # Season Length, etc.) have colorscale ranges in completely different units
    # and must not be used to filter raw VI observations.
    _basemap_metric = (basemap_info or {}).get("metric_key", "")
    if _basemap_metric in _FAST_METRIC_KEYS:
        zmin = basemap_info.get("zmin") if basemap_info else None
        zmax = basemap_info.get("zmax") if basemap_info else None
    else:
        zmin = zmax = None
    mask = ts.valid_mask.copy()
    if zmin is not None:
        mask &= ts.raw_vi >= float(zmin)
    if zmax is not None:
        mask &= ts.raw_vi <= float(zmax)

    _yr = year_range or [2000, 2030]
    yr_lo = int(_yr[0])
    yr_hi = int(_yr[1])
    obs_years = ts.dates.astype("datetime64[Y]").astype(int) + 1970
    year_keep = (obs_years >= yr_lo) & (obs_years <= yr_hi)
    mask &= year_keep

    narrowed = PixelTimeSeries(
        dates      = ts.dates[year_keep],
        raw_vi     = ts.raw_vi[year_keep],
        valid_mask = mask[year_keep],
        x_coord    = ts.x_coord,
        y_coord    = ts.y_coord,
        lon        = ts.lon,
        lat        = ts.lat,
    )

    if len(narrowed.dates) < 2:
        return None

    dc = build_date_cache_from_dates(narrowed.dates)

    vi_var = paths.vi_var
    vi_min, vi_max = VI_VALID_RANGE.get(vi_var, VI_VALID_RANGE[DEFAULT_VI_VAR])
    if zmin is not None:
        vi_min = max(vi_min, float(zmin))
    if zmax is not None:
        vi_max = min(vi_max, float(zmax))
    metric_config = {**PIXEL_METRIC_CONFIG, "vi_min": vi_min, "vi_max": vi_max}

    try:
        result = compute_pixel_with_annual(
            narrowed, dc, float(lambda_val or LAMBDA_DEFAULT), metric_config
        )
    except Exception:
        return None

    if result is None:
        return None

    return _serialize_pixel_result(result, narrowed, ts, region, vi_var)


# ---------------------------------------------------------------------------
# Chart callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("timeseries-chart", "figure"),
    Input("pixel-result", "data"),
    State("basemap-info", "data"),
)
def render_timeseries(pixel_result, basemap_info):
    if pixel_result is None:
        return make_empty_timeseries_figure()

    smoothed_daily = np.array(
        [v if v is not None else np.nan for v in pixel_result["smoothed_daily"]]
    )
    daily_dates = np.array(pixel_result["daily_dates"], dtype="datetime64[D]")

    # Reconstruct a PixelTimeSeries-compatible namespace from the store
    ts_proxy = SimpleNamespace(
        dates      = np.array(pixel_result["obs_dates"], dtype="datetime64[D]"),
        raw_vi     = np.array(pixel_result["obs_vi"]),
        valid_mask = np.ones(len(pixel_result["obs_vi"]), dtype=bool),
        lat        = pixel_result["ts_lat"],
        lon        = pixel_result["ts_lon"],
    )

    return make_timeseries_figure(
        ts             = ts_proxy,
        smoothed_daily = smoothed_daily,
        daily_dates    = daily_dates,
        region_id      = pixel_result["region_id"],
        vi_var         = pixel_result["vi_var"],
        zmin           = basemap_info.get("zmin") if basemap_info else None,
        zmax           = basemap_info.get("zmax") if basemap_info else None,
    )


@callback(
    Output("annual-cycle-chart", "figure"),
    Input("pixel-result", "data"),
    State("basemap-info", "data"),
)
def render_annual_cycle(pixel_result, basemap_info):
    if pixel_result is None:
        return make_empty_timeseries_figure()

    smoothed_daily = np.array(
        [v if v is not None else np.nan for v in pixel_result["smoothed_daily"]]
    )
    daily_dates = np.array(pixel_result["daily_dates"], dtype="datetime64[D]")

    return make_annual_cycle_figure(
        smoothed    = smoothed_daily,
        daily_dates = daily_dates,
        region_id   = pixel_result["region_id"],
        vi_var      = pixel_result["vi_var"],
        zmin        = basemap_info.get("zmin") if basemap_info else None,
        zmax        = basemap_info.get("zmax") if basemap_info else None,
    )


@callback(
    Output("metrics-annual-chart", "figure"),
    Output("metrics-annual-chart", "style"),
    Input("pixel-result", "data"),
)
def render_metrics_annual(pixel_result):
    _empty_style = {"height": "100%"}
    if pixel_result is None:
        return make_empty_timeseries_figure(), _empty_style

    valid_years = pixel_result.get("valid_years", [])
    if not valid_years:
        return make_empty_timeseries_figure(), _empty_style

    fig = make_metrics_annual_figure(
        valid_years = valid_years,
        annual_data = _restore_annual_nan(pixel_result["annual_data"]),
        metrics     = _restore_nan(pixel_result["metrics"]),
        region_id   = pixel_result["region_id"],
    )
    return fig, {"height": f"{fig.layout.height}px"}


@callback(
    Output("phenology-scatter-chart", "figure"),
    Input("pixel-result", "data"),
    State("basemap-info", "data"),
    State("lambda-slider", "value"),
)
def render_phenology_scatter(pixel_result, basemap_info, lambda_val):
    if pixel_result is None:
        return make_empty_timeseries_figure()

    ts_proxy = SimpleNamespace(
        dates      = np.array(pixel_result["obs_dates"], dtype="datetime64[D]"),
        raw_vi     = np.array(pixel_result["obs_vi"]),
        valid_mask = np.ones(len(pixel_result["obs_vi"]), dtype=bool),
        lat        = pixel_result["ts_lat"],
        lon        = pixel_result["ts_lon"],
    )

    return make_phenology_scatter_figure(
        ts        = ts_proxy,
        vi_var    = pixel_result["vi_var"],
        lam       = float(lambda_val or LAMBDA_DEFAULT),
        region_id = pixel_result["region_id"],
        zmin      = basemap_info.get("zmin") if basemap_info else None,
        zmax      = basemap_info.get("zmax") if basemap_info else None,
    )


# ---------------------------------------------------------------------------
# Sidebar callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("pixel-info", "children"),
    Input("selected-pixel", "data"),
    Input("pixel-result", "data"),
)
def update_pixel_info(selected_pixel, pixel_result):
    if selected_pixel is None:
        return html.P(
            "No pixel selected. Click on the map.",
            style={"color": "rgba(255,255,255,0.28)", "fontSize": "0.82em", "fontStyle": "italic", "fontFamily": "'Space Mono', monospace", "letterSpacing": "0.06em"},
        )

    lon = selected_pixel["lon"]
    lat = selected_pixel["lat"]
    yi  = selected_pixel["yi"]
    xi  = selected_pixel["xi"]

    _row = lambda lbl, val: html.Div([
        html.Span(lbl + " ", style={"color": "rgba(255,255,255,0.35)"}),
        html.Span(val,        style={"color": "#5be3ff"}),
    ], style={"fontFamily": "'Space Mono', monospace", "fontSize": "10px", "marginBottom": "2px"})

    if pixel_result is None:
        return html.Div([
            html.Div("Selected Pixel", style={"fontFamily": "'Space Mono', monospace", "fontSize": "9px", "letterSpacing": "0.13em", "textTransform": "uppercase", "color": "rgba(255,255,255,0.45)", "marginBottom": "5px"}),
            _row("LAT", f"{lat:.4f}°"),
            _row("LON", f"{lon:.4f}°"),
            html.Div("Computing…",
                     style={"color": "rgba(255,255,255,0.28)", "fontStyle": "italic", "fontSize": "10px", "fontFamily": "'Space Mono', monospace", "marginTop": "4px"}),
        ], style={"background": "#0d1c2e", "border": "1px solid rgba(91,227,255,0.13)", "borderRadius": "2px", "padding": "8px 10px"})

    # pixel_result is available — fall through to render full info below

    n_total    = pixel_result["ts_n_total"]
    n_valid    = pixel_result["ts_n_valid"]
    n_in_range = pixel_result["ts_n_in_range"]
    pct_valid  = 100.0 * n_valid / n_total if n_total > 0 else 0.0
    pct_range  = 100.0 * n_in_range / n_valid if n_valid > 0 else 0.0

    obs_dates = pixel_result.get("obs_dates", [])
    obs_vi    = pixel_result.get("obs_vi", [])
    date_first = obs_dates[0][:10]  if obs_dates else "—"
    date_last  = obs_dates[-1][:10] if obs_dates else "—"
    vi_lo = min(obs_vi) if obs_vi else float("nan")
    vi_hi = max(obs_vi) if obs_vi else float("nan")

    range_line = (
        [html.B("In range: "),
         f"{n_in_range} / {n_valid} valid  ({pct_range:.1f}%)",
         html.Br()]
        if n_in_range != n_valid else []
    )

    _row = lambda lbl, val: html.Div([
        html.Span(lbl + " ", style={"color": "rgba(255,255,255,0.35)"}),
        html.Span(val,        style={"color": "#5be3ff"}),
    ], style={"fontFamily": "'Space Mono', monospace", "fontSize": "10px", "marginBottom": "2px"})

    extra = [_row("IN RANGE", f"{n_in_range} / {n_valid}")] if n_in_range != n_valid else []

    return html.Div([
        html.Div("Selected Pixel", style={"fontFamily": "'Space Mono', monospace", "fontSize": "9px", "letterSpacing": "0.13em", "textTransform": "uppercase", "color": "rgba(255,255,255,0.45)", "marginBottom": "5px"}),
        _row("LAT",      f"{lat:.4f}°"),
        _row("LON",      f"{lon:.4f}°"),
        _row("VALID OBS", f"{n_valid} / {n_total}  ({pct_valid:.1f}%)"),
        *extra,
        _row("DATE RANGE", f"{date_first} → {date_last}"),
        _row(pixel_result['vi_var'] + " RANGE", f"{vi_lo:.3f} – {vi_hi:.3f}"),
    ], style={"background": "#0d1c2e", "border": "1px solid rgba(91,227,255,0.13)", "borderRadius": "2px", "padding": "8px 10px"})


@callback(
    Output("metrics-table", "children"),
    Input("pixel-result", "data"),
    State("basemap-info", "data"),
    State("metric-select", "value"),
)
def update_metrics_table(pixel_result, basemap_info, metric_key):
    if pixel_result is None:
        return html.P(
            "Select a pixel to compute metrics.",
            style={"color": "rgba(255,255,255,0.28)", "fontSize": "0.82em", "fontStyle": "italic", "fontFamily": "'Space Mono', monospace", "letterSpacing": "0.06em"},
        )

    metrics    = _restore_nan(pixel_result["metrics"])
    zmin       = basemap_info.get("zmin") if basemap_info else None
    zmax       = basemap_info.get("zmax") if basemap_info else None
    table_html = make_metrics_table(metrics, selected_metric=metric_key,
                                    zmin=zmin, zmax=zmax)
    return dcc.Markdown(table_html, dangerously_allow_html=True)


# ---------------------------------------------------------------------------
# Callbacks: shapefile click → region selection
# ---------------------------------------------------------------------------

if _SHAPEFILE_LIST:
    @callback(
        Output("region-dropdown", "value"),
        Output("shapefile-region-locked", "data"),
        Output("no-data-toast", "is_open"),
        Output("no-data-toast", "children"),
        [Input(f"shapefile-layer-{i}", "clickData") for i in range(len(_SHAPEFILE_LIST))],
        State("shapefile-region-locked", "data"),
        State("region-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_shapefile_click(*args):
        n               = len(_SHAPEFILE_LIST)
        click_data_list = args[:n]
        current_region  = args[n + 1]

        # Identify the clicked feature
        feature = next((cd for cd in click_data_list if cd is not None), None)
        if feature is None:
            raise PreventUpdate

        box_nr = (feature.get("properties") or {}).get("box_nr")
        if box_nr is None or box_nr == current_region:
            raise PreventUpdate

        if box_nr in ALL_REGIONS:
            return box_nr, no_update, False, ""

        return (
            no_update, no_update, True,
            f"No datacube found for flight box {box_nr}.",
        )


# Shapefile visibility toggle — only registered if layers exist
if _SHAPEFILE_LIST:
    _sf_outputs = [Output(f"shapefile-layer-{i}", "style")
                   for i in range(len(_SHAPEFILE_LIST))]

    @callback(
        _sf_outputs,
        Input("shapefile-visible", "value"),
    )
    def toggle_shapefile_layers(visible_set):
        visible_set = set(visible_set or [])
        return [
            {"color": "#ffffff", "weight": 2, "fillOpacity": 0.0}
            if str(i) in visible_set
            else {"opacity": 0, "fillOpacity": 0, "color": "transparent"}
            for i in range(len(_SHAPEFILE_LIST))
        ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
