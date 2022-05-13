"""Functions operating on dates and times."""

import datetime
import itertools
from numbers import Number
from typing import Union

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


def get_dates(xr_data: Union[xr.Dataset, xr.DataArray]) -> np.ndarray:
    return np.sort(np.unique(pd.DatetimeIndex(xr_data.time_utc).date))


def num_days(xr_data: Union[xr.Dataset, xr.DataArray]) -> int:
    dates = get_dates(xr_data)
    return len(dates)


def date_summary_str(xr_data: Union[xr.Dataset, xr.DataArray]) -> str:
    """Convert to pd.DatetimeIndex to get prettier date string formatting."""
    time_index = pd.DatetimeIndex(xr_data.time_utc)
    return (
        f"there are {num_days(xr_data):,d} days of data"
        f" from {time_index[0]} to {time_index[-1]}."
        f" A total of {len(time_index):,d} timesteps."
    )


def select_data_in_daylight(
    xr_data: Union[xr.Dataset, xr.DataArray], solar_elevation_threshold_degrees: Number = 5
) -> Union[xr.Dataset, xr.DataArray]:
    """Only select data where, for at least one of the four corners of the imagery,
    the Sun is at least `solar_elevation_threshold_degrees` above the horizon."""
    y_osgb = xr_data.y_osgb
    x_osgb = xr_data.x_osgb

    corners_osgb = [
        (x_osgb.isel(x=x, y=y).values, y_osgb.isel(x=x, y=y).values)
        for x, y in itertools.product((0, -1), (0, -1))
    ]

    corners_osgb = pd.DataFrame(corners_osgb, columns=["x", "y"])

    lats, lons = osgb_to_lat_lon(x=corners_osgb.x, y=corners_osgb.y)

    elevation_for_all_corners = []
    for lat, lon in zip(lats, lons):
        solpos = pvlib.solarposition.get_solarposition(
            time=xr_data.time,
            latitude=lat,
            longitude=lon,
        )
        elevation = solpos["elevation"]
        elevation_for_all_corners.append(elevation)

    elevation_for_all_corners = pd.concat(elevation_for_all_corners, axis="columns")
    max_elevation = elevation_for_all_corners.max(axis="columns")
    daylight_hours_mask = max_elevation >= solar_elevation_threshold_degrees
    return xr_data.isel(time=daylight_hours_mask)


def intersection_of_multiple_dataframes_of_periods(
    time_periods: list[pd.DataFrame],
) -> pd.DataFrame:
    """Find the intersection of a list of time periods.

    See the docstring of intersection_of_2_dataframes_of_periods() for more details.
    """
    assert len(time_periods) > 0
    if len(time_periods) == 1:
        return time_periods[0]
    intersection = intersection_of_2_dataframes_of_periods(time_periods[0], time_periods[1])
    for time_period in time_periods[2:]:
        intersection = intersection_of_2_dataframes_of_periods(intersection, time_period)
    return intersection


def intersection_of_2_dataframes_of_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of two pd.DataFrames of time periods.

    Each row of each pd.DataFrame represents a single time period.  Each pd.DataFrame has
    two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

    A typical use-case is that each pd.DataFrame represents all the time periods where
    a `DataSource` has contiguous, valid data.

    Here's a graphical example of two pd.DataFrames of time periods and their intersection:

                 ----------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a, b: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        Sorted list of intersecting time periods represented as a pd.DataFrame with two columns:
        start_dt and end_dt.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    all_intersecting_periods = []
    for a_period in a.itertuples():
        # Five ways in which two periods may overlap:
        # a: |----| or |---|   or  |---| or   |--|   or |-|
        # b:  |--|       |---|   |---|      |------|    |-|
        # In all five, `a` must always start before `b` ends,
        # and `a` must always end after `b` starts:
        overlapping_periods = b[(a_period.start_dt < b.end_dt) & (a_period.end_dt > b.start_dt)]

        # There are two ways in which two periods may *not* overlap:
        # a: |---|        or        |---|
        # b:       |---|      |---|
        # `overlapping` will not include periods which do *not* overlap.

        # Now find the intersection of each period in `overlapping_periods` with
        # the period from `a` that starts at `a_start_dt` and ends at `a_end_dt`.
        # We do this by clipping each row of `overlapping_periods`
        # to start no earlier than `a_start_dt`, and end no later than `a_end_dt`.

        # First, make a copy, so we don't clip the underlying data in `b`.
        intersecting_periods = overlapping_periods.copy()
        intersecting_periods.start_dt.clip(lower=a_period.start_dt, inplace=True)
        intersecting_periods.end_dt.clip(upper=a_period.end_dt, inplace=True)

        all_intersecting_periods.append(intersecting_periods)

    all_intersecting_periods = pd.concat(all_intersecting_periods)
    return all_intersecting_periods.sort_values(by="start_dt").reset_index(drop=True)