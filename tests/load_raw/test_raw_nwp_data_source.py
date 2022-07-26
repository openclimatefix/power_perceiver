import pandas as pd
import pytest

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_nwp_data_source import RawNWPDataSource


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_convert_t0_time_periods_to_periods_to_load(nwp_data_source: RawNWPDataSource):
    nwp_data_source.per_worker_init(worker_id=0, seed=0)
    t0_periods = pd.DataFrame(
        [
            {"start_dt": "2020-02-19 12:00", "end_dt": "2020-02-19 15:00"},
            {"start_dt": "2020-02-21 08:20", "end_dt": "2020-02-21 15:00"},
        ]
    )
    for col_name in t0_periods.columns:
        t0_periods[col_name] = pd.to_datetime(t0_periods[col_name])
    nwp_data_source.load_subset_into_ram(t0_periods)
    location = Location(x=379379.90625, y=583073.0)
    # Use the same datetime that caused the crash described in:
    # https://github.com/openclimatefix/power_perceiver/issues/138
    _ = nwp_data_source.get_example(
        t0_datetime_utc=pd.Timestamp("2020-02-21T11:45"),
        center_osgb=location,
    )
