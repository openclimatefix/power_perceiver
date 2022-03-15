"""Constants and Enums."""

from enum import Enum

from pathy import Pathy


class DataSourceName(Enum):  # noqa: D101
    gsp = "gsp"
    nwp = "nwp"
    opticalflow = "opticalflow"
    pv = "pv"
    satellite = "satellite"
    hrvsatellite = "hrvsatellite"
    sun = "sun"
    topographic = "topographic"


class BatchKey(Enum):
    """The names of the different elements of each batch.

    This is also where we document the exact shape of each element.

    This is basically a superset of DataSourceName, because each DataSource
    may be split into several different BatchKey elements. For example, the
    pv DataSource yields `pv` and `pv_system_id` BatchKeys.
    """

    satellite = "satellite"
    hrvsatellite = "hrvsatellite"
    pv = "pv"
    pv_system_id = "pv_system_id"


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)
