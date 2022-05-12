"""Functions operating on dates and times."""

import datetime
import itertools
from numbers import Number

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

from power_perceiver.geospatial import osgb_to_lat_lon


def get_contiguous_time_periods(
    datetimes: pd.DatetimeIndex,
    min_seq_length: int,
    max_gap_duration: datetime.timedelta,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
      datetimes: pd.DatetimeIndex. Must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.  Typically, this
        would be set to the `total_seq_length` of each machine learning example.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences. Typically, `max_gap_duration` would be set to the sample period of
        the timeseries.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert min_seq_length > 1
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique

    # Find indices of gaps larger than max_gap:
    gap_mask = np.diff(datetimes) > max_gap_duration
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into dt_index
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    periods: list[dict[str, pd.Timestamp]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    assert len(periods) > 0

    return pd.DataFrame(periods)


def get_dates(xr_sat_dataset: xr.Dataset) -> np.ndarray:
    return np.sort(np.unique(pd.DatetimeIndex(xr_sat_dataset.time_utc).date))


def num_days(xr_sat_dataset: xr.Dataset) -> int:
    dates = get_dates(xr_sat_dataset)
    return len(dates)


def date_summary_str(xr_sat_dataset: xr.Dataset) -> str:
    """Convert to pd.DatetimeIndex to get prettier date string formatting."""
    time_index = pd.DatetimeIndex(xr_sat_dataset.time_utc)
    return (
        f"there are {num_days(xr_sat_dataset):,d} days of data"
        f" from {time_index[0]} to {time_index[-1]}."
        f" A total of {len(time_index):,d} timesteps."
    )


def _select_timesteps_in_contiguous_periods(
    xr_sat_dataset: xr.Dataset, min_seq_length: int
) -> xr.Dataset:
    dt_index = pd.DatetimeIndex(xr_sat_dataset.time)
    good_time_idx = get_contiguous_time_periods(dt_index, min_seq_length=min_seq_length)
    good_time_idx = np.concatenate(good_time_idx)
    return xr_sat_dataset.isel(time=good_time_idx)


def _select_data_in_daylight(
    xr_sat_dataset: xr.Dataset, solar_elevation_threshold_degrees: Number = 5
) -> xr.Dataset:
    """Only select data where, for at least one of the four corners of the satellite imagery,
    the Sun is at least `solar_elevation_threshold_degrees` above the horizon."""
    y_osgb = xr_sat_dataset.y_osgb
    x_osgb = xr_sat_dataset.x_osgb

    corners_osgb = [
        (x_osgb.isel(x=x, y=y).values, y_osgb.isel(x=x, y=y).values)
        for x, y in itertools.product((0, -1), (0, -1))
    ]

    corners_osgb = pd.DataFrame(corners_osgb, columns=["x", "y"])

    lats, lons = osgb_to_lat_lon(x=corners_osgb.x, y=corners_osgb.y)

    elevation_for_all_corners = []
    for lat, lon in zip(lats, lons):
        solpos = pvlib.solarposition.get_solarposition(
            time=xr_sat_dataset.time,
            latitude=lat,
            longitude=lon,
        )
        elevation = solpos["elevation"]
        elevation_for_all_corners.append(elevation)

    elevation_for_all_corners = pd.concat(elevation_for_all_corners, axis="columns")
    max_elevation = elevation_for_all_corners.max(axis="columns")
    daylight_hours_mask = max_elevation >= solar_elevation_threshold_degrees
    return xr_sat_dataset.isel(time=daylight_hours_mask)
