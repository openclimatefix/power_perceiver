import datetime

import numpy as np
import pytest

from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource


@pytest.fixture(scope="module")
def gsp_data_source() -> RawGSPDataSource:
    gsp = RawGSPDataSource(
        gsp_pv_power_zarr_path="~/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="~/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="~/data/PV/GSP/gsp_shape",
        start_date="2020-01-01",
        end_date="2020-06-01",
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
    )
    gsp.per_worker_init(worker_id=0)
    return gsp


def test_init(gsp_data_source: RawGSPDataSource):
    assert np.isfinite(gsp_data_source._data_in_ram.data).all()
