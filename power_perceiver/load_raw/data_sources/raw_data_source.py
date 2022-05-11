import datetime
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass
class RawDataSource:
    forecast_duration: datetime.timedelta
    history_duration: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    transforms: Optional[Iterable[Callable]] = None
