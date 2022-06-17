from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import xarray as xr

from power_perceiver.time import set_new_sample_period_and_t0_idx_attrs


@dataclass
class PVPowerRollingWindow:
    """Compute rolling mean of PV power.

    The xr.Dataset is modified in-place, and is returned.

    Initialisation arguments: (taken from pandas.DataFrame.rolling docs)
        window: Size of the moving window.
            If an integer, the fixed number of observations used for each window.

            If an offset, the time period of each window. Each window will be a variable sized
            based on the observations included in the time-period. This is only valid for
            datetimelike indexes. To learn more about the offsets & frequency strings, please see:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

            If a BaseIndexer subclass, the window boundaries based on the defined
            `get_window_bounds` method. Additional rolling keyword arguments,
            namely `min_periods` and `center` will be passed to `get_window_bounds`.

        min_periods: Minimum number of observations in window required to have a value;
            otherwise, result is `np.nan`.

            To avoid NaNs at the start and end of the timeseries, this should be <= ceil(window/2).

            For a window that is specified by an offset, `min_periods` will default to 1.

            For a window that is specified by an integer, `min_periods` will default to the size of
            the window.

        center: If False, set the window labels as the right edge of the window index.
            If True, set the window labels as the center of the window index.
    """

    window: Union[int, pd.tseries.offsets.DateOffset, pd.core.indexers.objects.BaseIndexer] = 3
    min_periods: Optional[int] = 2
    center: bool = True
    win_type: Optional[str] = None
    expect_dataset: bool = True

    def __call__(self, xr_data: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        if self.expect_dataset:
            data_to_resample = xr_data["power_w"]
        else:
            data_to_resample = xr_data

        resampled = data_to_resample.rolling(
            dim={"time_utc": self.window},
            min_periods=self.min_periods,
            center=self.center,
        ).mean()

        if self.expect_dataset:
            xr_data["power_w"] = resampled
            resampled = xr_data

        # Resampling removes the attributes, so put them back:
        for attr_name in ("t0_idx", "sample_period_duration"):
            resampled.attrs[attr_name] = xr_data.attrs[attr_name]

        return resampled


@dataclass
class PVDownsample:
    """Downsample PV.

    The xr.Dataset is modified in-place, and is returned.

    Initialisation arguments:
        freq:
        expect_dataset:
    """

    freq: str = "30T"
    expect_dataset: bool = True

    def __call__(self, xr_data: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        if self.expect_dataset:
            data_to_resample = xr_data["power_w"]
        else:
            data_to_resample = xr_data

        resampled = data_to_resample.resample(time_utc=self.freq, label="right").mean()

        if self.expect_dataset:
            xr_data["power_w"] = resampled
            resampled = xr_data

        # Resampling removes the attributes, so put them back:
        for attr_name in ("t0_idx", "sample_period_duration"):
            resampled.attrs[attr_name] = xr_data.attrs[attr_name]

        # Change the pv_t0_idx and the sample_period_duration attributes:
        resampled = set_new_sample_period_and_t0_idx_attrs(resampled, new_sample_period=self.freq)
        return resampled
