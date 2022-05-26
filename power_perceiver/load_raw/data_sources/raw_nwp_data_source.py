from dataclasses import dataclass

from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDataSource,
)


@dataclass(kw_only=True)
class RawNWPDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    ZarrDataSource,
    TimeseriesDataSource,
    SpatialDataSource,
    RawDataSource,
):
    pass
