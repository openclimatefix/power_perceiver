import datetime

from power_perceiver.load_raw.data_sources.raw_pv_data_source import (
    RawPVDataSource,
    _load_pv_metadata,
    _load_pv_power_watts_and_capacity_wp,
)

# TODO: Use public data :)
PV_METADATA_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv"
PV_POWER_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf"


def test_load_pv_metadata():  # noqa: D103
    pv_metadata = _load_pv_metadata(PV_METADATA_FILENAME)
    assert len(pv_metadata) > 1000


def test_load_pv_power_watts_and_capacity_wp():  # noqa: D103
    pv_power_watts, pv_capacity_wp = _load_pv_power_watts_and_capacity_wp(
        PV_POWER_FILENAME, start_date="2020-01-01", end_date="2020-01-03"
    )
    assert len(pv_power_watts) == 863
    assert len(pv_power_watts.columns) == 956


def test_init():  # noqa: D103
    pv = RawPVDataSource(
        pv_power_filename=PV_POWER_FILENAME,
        pv_metadata_filename=PV_METADATA_FILENAME,
        start_date="2020-01-01",
        end_date="2020-01-03",
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
        roi_height_meters=48_000,
        roi_width_meters=48_000,
    )
    print(pv._data_in_ram)

    pv_power_normalised = pv._data_in_ram / pv._data_in_ram.capacity_wp
    assert pv_power_normalised.max().max() <= 1
