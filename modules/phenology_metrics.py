"""
phenology_metrics.py — Single-pixel Whittaker smoothing and 19-metric
computation for the VI Phenology Dashboard.

The three core functions (_build_whittaker_system, _whittaker_smooth_pixel,
_extract_pixel_metrics) are reproduced verbatim from:
    VI_Phenology/src/pixel_phenology_extract.py

They are copied here rather than imported because the source module has
heavy top-level imports (matplotlib, tqdm, io_utils) that are not part of
the dashboard environment.  The logic is identical.  All mathematical
credit belongs to the original authors listed in that file.

Key implementation details
--------------------------
- The Whittaker system is keyed on (n_days, lam), where n_days is the
  calendar span (first obs → last obs), NOT the number of observations.
- smooth_pixel() returns values on the DAILY grid (n_days points).
- compute_pixel_metrics() returns only the 19 aggregated scalars.
- compute_pixel_with_annual() returns both the 19 metrics AND per-year
  lists (valid_years, annual_data) for plotting each metric over time.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve

from modules.datacube_io import PixelTimeSeries

# ---------------------------------------------------------------------------
# Constants (from pixel_phenology_extract.py)
# ---------------------------------------------------------------------------

_MIN_AMPLITUDE = 1e-6

_METRIC_NAMES = [
    "peak_ndvi_mean", "peak_ndvi_std",
    "peak_doy_mean",  "peak_doy_std",
    "integrated_ndvi_mean", "integrated_ndvi_std",
    "greenup_rate_mean", "greenup_rate_std",
    "floor_ndvi_mean", "ceiling_ndvi_mean",
    "season_length_mean", "season_length_std",
    "cv", "interannual_peak_range", "interannual_peak_std",
    "n_peaks_mean",
    "peak_separation_mean",
    "relative_peak_amplitude_mean",
    "valley_depth_mean",
]

# Maps primary *_mean metrics to their key in the annual_data dict.
# Metrics not listed here are scalar / derived and have no per-year series.
METRIC_ANNUAL_KEY: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Whittaker functions (copied verbatim from pixel_phenology_extract.py)
# ---------------------------------------------------------------------------

def _build_whittaker_system(n: int, lam: float):
    """Pre-build the λ D^T D penalty term for a daily grid of length n."""
    e = np.ones(n)
    D = sp_diags(
        [e[:-2], -2 * e[:-1], e],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format='csc',
    )
    return lam * D.T @ D


def _whittaker_smooth_pixel(
    daily_y: np.ndarray,
    daily_w: np.ndarray,
    lam_DTD,
) -> np.ndarray:
    """Solve the Whittaker system for one pixel."""
    W = sp_diags(daily_w, format='csc')
    A = W + lam_DTD
    b = daily_w * daily_y
    try:
        return spsolve(A, b)
    except Exception:
        return daily_y.copy()


def _extract_pixel_metrics(
    pixel_ts: np.ndarray,
    lam_DTD,
    config: dict,
    date_cache: dict,
) -> dict:
    """Compute all 19 phenological metrics for one pixel time series."""
    nan_result = {k: np.nan for k in _METRIC_NAMES}

    vi_min = config["vi_min"]
    vi_max = config["vi_max"]
    valid_mask = (
        ~np.isnan(pixel_ts)
        & (pixel_ts >= vi_min)
        & (pixel_ts <= vi_max)
    )
    if valid_mask.sum() < config["min_valid_obs"]:
        return nan_result

    raw_vals = pixel_ts[valid_mask]
    mean_raw = float(np.mean(raw_vals))
    cv = float(np.std(raw_vals) / mean_raw) if mean_raw > 0 else np.nan

    n_days      = date_cache["n_days"]
    day_offsets = date_cache["day_offsets"]
    valid_days  = day_offsets[valid_mask]
    valid_vals  = pixel_ts[valid_mask]

    daily_y   = np.zeros(n_days, dtype=np.float64)
    daily_w   = np.zeros(n_days, dtype=np.float64)
    day_count = np.zeros(n_days, dtype=np.float64)
    np.add.at(daily_y,   valid_days, valid_vals)
    np.add.at(day_count, valid_days, 1.0)
    hit = day_count > 0
    daily_y[hit] /= day_count[hit]
    daily_w[hit]  = 1.0

    if n_days < 3 or lam_DTD is None:
        smoothed = daily_y.copy()
    else:
        smoothed = _whittaker_smooth_pixel(daily_y, daily_w, lam_DTD)
        smoothed = np.clip(smoothed, vi_min, vi_max)

    doy_arr    = date_cache["doy_arr"]
    years      = date_cache["years"]
    year_masks = date_cache["year_masks"]

    peak_prominence  = config["peak_prominence"]
    peak_min_dist    = config["peak_min_distance_days"]
    season_thr       = config["season_threshold"]
    min_obs_per_year = config["min_valid_obs_per_year"]

    annual = {
        "peak_ndvi":  [], "peak_doy":  [], "integrated": [],
        "greenup":    [], "floor":     [], "ceiling":    [],
        "season_len": [], "n_peaks":   [], "peak_sep":   [],
        "rel_amp":    [], "valley":    [],
    }

    for yr in years:
        mask = year_masks[int(yr)]
        y    = smoothed[mask]
        if len(y) < 30:
            continue
        if int(daily_w[mask].sum()) < min_obs_per_year:
            continue
        y    = y.astype(np.float64)
        doys = doy_arr[mask]

        peak_idx = int(np.argmax(y))
        annual["peak_ndvi"].append(float(y[peak_idx]))
        annual["peak_doy"].append(int(doys[peak_idx]))
        annual["integrated"].append(float(np.trapezoid(y)))
        floor_val = float(np.nanmin(y))
        ceil_val  = float(np.nanmax(y))
        annual["floor"].append(floor_val)
        annual["ceiling"].append(ceil_val)

        floor_idx = int(np.argmin(y))
        if floor_idx < peak_idx:
            delta_ndvi = float(y[peak_idx] - y[floor_idx])
            delta_days = int(doys[peak_idx] - doys[floor_idx])
            rate = delta_ndvi / delta_days if delta_days > 0 else np.nan
            annual["greenup"].append(rate)

        amplitude = ceil_val - floor_val
        if amplitude >= _MIN_AMPLITUDE:
            threshold  = floor_val + season_thr * amplitude
            yr_indices = np.where(mask)[0]
            above      = yr_indices[y >= threshold]
            if len(above) >= 2:
                annual["season_len"].append(float(above[-1] - above[0]))

        peaks, _ = _find_peaks(y, prominence=peak_prominence, distance=peak_min_dist)
        n_p = int(len(peaks))
        annual["n_peaks"].append(n_p)

        if n_p >= 2:
            sorted_peaks = peaks[np.argsort(y[peaks])[::-1]]
            p1, p2 = sorted_peaks[0], sorted_peaks[1]
            annual["peak_sep"].append(float(abs(doys[p1] - doys[p2])))
            h1, h2 = float(y[p1]), float(y[p2])
            if max(h1, h2) > 0:
                annual["rel_amp"].append(float(min(h1, h2) / max(h1, h2)))
            lo, hi = min(p1, p2), max(p1, p2)
            valley   = float(np.nanmin(y[lo: hi + 1]))
            mean_pk  = (h1 + h2) / 2.0
            if mean_pk > 0:
                annual["valley"].append(float((mean_pk - valley) / mean_pk))
        else:
            annual["peak_sep"].append(np.nan)
            annual["rel_amp"].append(np.nan)
            annual["valley"].append(np.nan)

    def _safe_mean(lst):
        a = [v for v in lst if not np.isnan(v)]
        return float(np.mean(a)) if a else np.nan

    def _safe_std(lst):
        a = [v for v in lst if not np.isnan(v)]
        return float(np.std(a)) if a else np.nan

    peak_list = annual["peak_ndvi"]
    interannual_range = (
        float(np.nanmax(peak_list) - np.nanmin(peak_list)) if peak_list else np.nan
    )

    return {
        "peak_ndvi_mean":               _safe_mean(annual["peak_ndvi"]),
        "peak_ndvi_std":                _safe_std(annual["peak_ndvi"]),
        "peak_doy_mean":                _safe_mean(annual["peak_doy"]),
        "peak_doy_std":                 _safe_std(annual["peak_doy"]),
        "integrated_ndvi_mean":         _safe_mean(annual["integrated"]),
        "integrated_ndvi_std":          _safe_std(annual["integrated"]),
        "greenup_rate_mean":            _safe_mean(annual["greenup"]),
        "greenup_rate_std":             _safe_std(annual["greenup"]),
        "floor_ndvi_mean":              _safe_mean(annual["floor"]),
        "ceiling_ndvi_mean":            _safe_mean(annual["ceiling"]),
        "season_length_mean":           _safe_mean(annual["season_len"]),
        "season_length_std":            _safe_std(annual["season_len"]),
        "cv":                           cv,
        "interannual_peak_range":       interannual_range,
        "interannual_peak_std":         _safe_std(annual["peak_ndvi"]),
        "n_peaks_mean":                 _safe_mean(annual["n_peaks"]),
        "peak_separation_mean":         _safe_mean(annual["peak_sep"]),
        "relative_peak_amplitude_mean": _safe_mean(annual["rel_amp"]),
        "valley_depth_mean":            _safe_mean(annual["valley"]),
    }


# ---------------------------------------------------------------------------
# Annual-loop variant — returns per-year data for metric plots
# ---------------------------------------------------------------------------

def _run_annual_loop_tracked(
    smoothed: np.ndarray,
    daily_w: np.ndarray,
    date_cache: dict,
    config: dict,
) -> tuple[list[int], dict[str, list]]:
    """
    Run the per-year metric loop and track which year each value belongs to.
    All per-year lists are padded with NaN for years that don't meet conditions,
    so every list has length == len(valid_years).

    Returns (valid_years, annual_data_dict).
    """
    doy_arr    = date_cache["doy_arr"]
    years      = date_cache["years"]
    year_masks = date_cache["year_masks"]

    peak_prominence  = config["peak_prominence"]
    peak_min_dist    = config["peak_min_distance_days"]
    season_thr       = config["season_threshold"]
    min_obs_per_year = config["min_valid_obs_per_year"]
    vi_min           = config["vi_min"]
    vi_max           = config["vi_max"]

    valid_years: list[int] = []
    annual: dict[str, list] = {
        "peak_ndvi":  [], "peak_doy":   [], "integrated": [],
        "greenup":    [], "floor":      [], "ceiling":    [],
        "season_len": [], "n_peaks":    [], "peak_sep":   [],
        "rel_amp":    [], "valley":     [],
    }

    for yr in years:
        mask = year_masks[int(yr)]
        y    = smoothed[mask]
        if len(y) < 30 or int(daily_w[mask].sum()) < min_obs_per_year:
            continue

        valid_years.append(int(yr))
        y    = np.clip(y.astype(np.float64), vi_min, vi_max)
        doys = doy_arr[mask]

        peak_idx = int(np.argmax(y))
        annual["peak_ndvi"].append(float(y[peak_idx]))
        annual["peak_doy"].append(int(doys[peak_idx]))
        annual["integrated"].append(float(np.trapezoid(y)))

        floor_val = float(np.nanmin(y))
        ceil_val  = float(np.nanmax(y))
        annual["floor"].append(floor_val)
        annual["ceiling"].append(ceil_val)

        floor_idx = int(np.argmin(y))
        if floor_idx < peak_idx:
            delta_days = int(doys[peak_idx] - doys[floor_idx])
            rate = float((y[peak_idx] - y[floor_idx]) / delta_days) if delta_days > 0 else np.nan
            annual["greenup"].append(rate)
        else:
            annual["greenup"].append(np.nan)

        amplitude = ceil_val - floor_val
        if amplitude >= _MIN_AMPLITUDE:
            threshold  = floor_val + season_thr * amplitude
            yr_indices = np.where(mask)[0]
            above      = yr_indices[y >= threshold]
            annual["season_len"].append(float(above[-1] - above[0]) if len(above) >= 2 else np.nan)
        else:
            annual["season_len"].append(np.nan)

        peaks, _ = _find_peaks(y, prominence=peak_prominence, distance=peak_min_dist)
        n_p = int(len(peaks))
        annual["n_peaks"].append(float(n_p))

        if n_p >= 2:
            sorted_peaks = peaks[np.argsort(y[peaks])[::-1]]
            p1, p2 = sorted_peaks[0], sorted_peaks[1]
            annual["peak_sep"].append(float(abs(doys[p1] - doys[p2])))
            h1, h2 = float(y[p1]), float(y[p2])
            annual["rel_amp"].append(float(min(h1, h2) / max(h1, h2)) if max(h1, h2) > 0 else np.nan)
            lo, hi   = min(p1, p2), max(p1, p2)
            valley   = float(np.nanmin(y[lo: hi + 1]))
            mean_pk  = (h1 + h2) / 2.0
            annual["valley"].append(float((mean_pk - valley) / mean_pk) if mean_pk > 0 else np.nan)
        else:
            annual["peak_sep"].append(np.nan)
            annual["rel_amp"].append(np.nan)
            annual["valley"].append(np.nan)

    return valid_years, annual


# ---------------------------------------------------------------------------
# Cached Whittaker penalty matrix
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _cached_whittaker_system(n_days: int, lam: float):
    """
    Build and cache the Whittaker λ D^T D sparse matrix for (n_days, lam).
    n_days is the calendar span, NOT the number of observations.
    """
    if n_days < 3:
        return None
    return _build_whittaker_system(n_days, float(lam))


def _build_daily_grid(ts: PixelTimeSeries, date_cache: dict):
    """Map irregular observations onto the regular daily grid. Returns (daily_y, daily_w)."""
    n_days      = date_cache["n_days"]
    day_offsets = date_cache["day_offsets"]
    valid_days  = day_offsets[ts.valid_mask]
    valid_vals  = ts.raw_vi[ts.valid_mask].astype(np.float64)

    daily_y   = np.zeros(n_days, dtype=np.float64)
    daily_w   = np.zeros(n_days, dtype=np.float64)
    day_count = np.zeros(n_days, dtype=np.float64)
    np.add.at(daily_y,   valid_days, valid_vals)
    np.add.at(day_count, valid_days, 1.0)
    hit = day_count > 0
    daily_y[hit] /= day_count[hit]
    daily_w[hit]  = 1.0
    return daily_y, daily_w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smooth_pixel(
    ts: PixelTimeSeries,
    date_cache: dict,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Whittaker smoothing. Returns (smoothed_daily, daily_dates).
    smoothed_daily has length n_days (regular daily grid).
    """
    n_days      = date_cache["n_days"]
    daily_y, daily_w = _build_daily_grid(ts, date_cache)
    first_date  = ts.dates[0]
    daily_dates = first_date + np.arange(n_days).astype("timedelta64[D]")

    lam_DTD = _cached_whittaker_system(n_days, lam)
    if lam_DTD is None:
        return daily_y, daily_dates

    smoothed = _whittaker_smooth_pixel(daily_y, daily_w, lam_DTD)
    return smoothed, daily_dates


def compute_pixel_metrics(
    ts: PixelTimeSeries,
    date_cache: dict,
    lam: float,
    config: dict,
) -> dict[str, float]:
    """Compute all 19 aggregated phenological metrics for a single pixel."""
    nan_result = {m: np.nan for m in _METRIC_NAMES}
    if ts.valid_mask.sum() < config.get("min_valid_obs", 20):
        return nan_result
    lam_DTD = _cached_whittaker_system(date_cache["n_days"], lam)
    try:
        return _extract_pixel_metrics(
            ts.raw_vi.astype(np.float64), lam_DTD, config, date_cache
        )
    except Exception:
        return nan_result


def compute_pixel_with_annual(
    ts: PixelTimeSeries,
    date_cache: dict,
    lam: float,
    config: dict,
) -> tuple[dict[str, float], list[int], dict[str, list], np.ndarray, np.ndarray]:
    """
    Compute all 19 metrics AND per-year data arrays for plotting.

    Returns
    -------
    metrics       : dict[str, float]   — 19 aggregated metrics
    valid_years   : list[int]          — years that had sufficient data
    annual_data   : dict[str, list]    — per-year arrays aligned with valid_years:
                        peak_ndvi, peak_doy, integrated, greenup, floor,
                        ceiling, season_len, n_peaks, peak_sep, rel_amp, valley
                    All lists have length == len(valid_years), NaN where the
                    per-year condition was not met.
    smoothed_daily : np.ndarray        — Whittaker-smoothed values on the daily grid
    daily_dates    : np.ndarray        — datetime64[D] daily date array
    """
    nan_metrics = {m: np.nan for m in _METRIC_NAMES}
    empty_annual: dict[str, list] = {k: [] for k in [
        "peak_ndvi", "peak_doy", "integrated", "greenup", "floor",
        "ceiling", "season_len", "n_peaks", "peak_sep", "rel_amp", "valley",
    ]}
    _empty_dates = ts.dates[:0].astype("datetime64[D]")  # zero-length sentinel

    if ts.valid_mask.sum() < config.get("min_valid_obs", 20):
        return nan_metrics, [], empty_annual, np.array([], dtype=np.float64), _empty_dates

    n_days = date_cache["n_days"]
    lam_DTD = _cached_whittaker_system(n_days, lam)
    vi_min  = config["vi_min"]
    vi_max  = config["vi_max"]

    daily_y, daily_w = _build_daily_grid(ts, date_cache)

    if n_days < 3 or lam_DTD is None:
        smoothed = daily_y.copy()
    else:
        smoothed = _whittaker_smooth_pixel(daily_y, daily_w, lam_DTD)
        smoothed = np.clip(smoothed, vi_min, vi_max)

    valid_years, annual_data = _run_annual_loop_tracked(
        smoothed, daily_w, date_cache, config
    )

    # Build the daily date array (needed by callers for time-axis labelling)
    first_date  = ts.dates[0]
    daily_dates = first_date + np.arange(n_days).astype("timedelta64[D]")

    # Build aggregate metrics from the per-year lists
    def _safe_mean(lst):
        a = [v for v in lst if not np.isnan(float(v))]
        return float(np.mean(a)) if a else np.nan

    def _safe_std(lst):
        a = [v for v in lst if not np.isnan(float(v))]
        return float(np.std(a)) if a else np.nan

    raw_vals = ts.raw_vi[ts.valid_mask]
    mean_raw = float(np.mean(raw_vals))
    cv = float(np.std(raw_vals) / mean_raw) if mean_raw > 0 else np.nan

    peak_list = annual_data["peak_ndvi"]
    interannual_range = (
        float(np.nanmax(peak_list) - np.nanmin(peak_list)) if peak_list else np.nan
    )

    metrics = {
        "peak_ndvi_mean":               _safe_mean(annual_data["peak_ndvi"]),
        "peak_ndvi_std":                _safe_std(annual_data["peak_ndvi"]),
        "peak_doy_mean":                _safe_mean(annual_data["peak_doy"]),
        "peak_doy_std":                 _safe_std(annual_data["peak_doy"]),
        "integrated_ndvi_mean":         _safe_mean(annual_data["integrated"]),
        "integrated_ndvi_std":          _safe_std(annual_data["integrated"]),
        "greenup_rate_mean":            _safe_mean(annual_data["greenup"]),
        "greenup_rate_std":             _safe_std(annual_data["greenup"]),
        "floor_ndvi_mean":              _safe_mean(annual_data["floor"]),
        "ceiling_ndvi_mean":            _safe_mean(annual_data["ceiling"]),
        "season_length_mean":           _safe_mean(annual_data["season_len"]),
        "season_length_std":            _safe_std(annual_data["season_len"]),
        "cv":                           cv,
        "interannual_peak_range":       interannual_range,
        "interannual_peak_std":         _safe_std(annual_data["peak_ndvi"]),
        "n_peaks_mean":                 _safe_mean(annual_data["n_peaks"]),
        "peak_separation_mean":         _safe_mean(annual_data["peak_sep"]),
        "relative_peak_amplitude_mean": _safe_mean(annual_data["rel_amp"]),
        "valley_depth_mean":            _safe_mean(annual_data["valley"]),
    }

    return metrics, valid_years, annual_data, smoothed, daily_dates


def source_available() -> bool:
    return True


def source_error() -> str | None:
    return None
