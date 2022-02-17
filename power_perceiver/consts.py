"""Constants and Enums."""

from enum import Enum

from pathy import Pathy


class DataSourceName(Enum):  # noqa: D101
    gsp = "gsp"
    nwp = "nwp"
    opticalflow = "opticalflow"
    pv = "pv"
    satellite = "satellite"
    sun = "sun"
    topographic = "topographic"


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)