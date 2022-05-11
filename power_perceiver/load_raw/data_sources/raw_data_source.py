import datetime
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, Optional, Union

import pandas as pd
import xarray as xr


@dataclass
class RawDataSource:
    history_duration: datetime.timedelta
    forecast_duration: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    transforms: Optional[Iterable[Callable]] = None

    def get_location(self) -> tuple[Number, Number]:
        """Find a valid geographical location.

        Should be overridden by DataSources which may be used to define the location
        of each example.

        Returns:  x_osgb, y_osgb
        """
        raise NotImplementedError()

    def get_example(
        self,
        location: SpaceTimeLocation,  #: Location object of the most recent observation
    ) -> xr.Dataset:
        """Must be overridden by child classes."""
        raise NotImplementedError()

    def check_input_paths_exist(self) -> None:
        """Check any input paths exist.  Raise FileNotFoundError if not.

        Must be overridden by child classes.
        """
        raise NotImplementedError()

    def _get_start_dt(
        self, t0_datetime_utc: Union[datetime.datetime, pd.DatetimeIndex]
    ) -> Union[datetime.datetime, pd.DatetimeIndex]:

        return t0_datetime_utc - self.history_duration

    def _get_end_dt(
        self, t0_datetime_utc: Union[datetime.datetime, pd.DatetimeIndex]
    ) -> Union[datetime.datetime, pd.DatetimeIndex]:
        return t0_datetime_utc + self.forecast_duration
