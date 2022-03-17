from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import xarray as xr


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

            For a window that is specified by an offset, `min_periods` will default to 1.

            For a window that is specified by an integer, `min_periods` will default to the size of
            the window.

        center: If False, set the window labels as the right edge of the window index.
            If True, set the window labels as the center of the window index.

        win_type: If None, all points are evenly weighted.
            If a string, it must be a valid scipy.signal window function:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows

            Certain Scipy window types require additional parameters to be passed in the
            aggregation function. The additional parameters must match the keywords specified in
            the Scipy window type method signature.
    """

    window: Union[int, pd.offset, pd.BaseIndexer] = 3
    min_periods: Optional[int] = 2
    center: bool = True
    win_type: Optional[str] = None

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset["power_mw"] = (
            dataset["power_mw"]
            .rolling(
                dim={"time_index": self.window},
                min_periods=self.min_periods,
                center=self.center,
                win_type=self.win_type,
            )
            .mean()
        )
        return dataset
