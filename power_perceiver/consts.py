"""Constants and Enums."""

from enum import Enum

from pathy import Pathy

PV_TIME_AXIS = 1
PV_SYSTEM_AXIS = 2


class BatchKey(Enum):
    """The names of the different elements of each batch.

    This is also where we document the exact shape of each element.

    This is basically a superset of all the DataLoaders, because each DataLoader
    may be split into several different BatchKey elements. For example, the
    PV DataLoader yields `pv` and `pv_system_row_number` BatchKeys.
    """

    # -------------- SATELLITE --------------------------------------
    # shape: (batch_size, time, channels, y, x)
    satellite = "satellite"

    # -------------- HRVSATELLITE -----------------------------------
    # shape: (batch_size, time, channels, y, x)
    hrvsatellite = "hrvsatellite"

    # HRV satellite coordinates:
    hrvsatellite_x_osgb = "hrvsatellite_x_osgb"  # shape: (batch_size, y, x)
    hrvsatellite_y_osgb = "hrvsatellite_y_osgb"  # shape: (batch_size, y, x)
    #: Time is seconds since UNIX epoch (1970-01-01). Shape: (batch_size, n_timesteps)
    hrvsatellite_time_utc = "hrvsatellite_time_utc"

    # HRV satellite Fourier coordinates:
    hrvsatellite_x_osgb_fourier = "hrvsatellite_x_osgb_fourier"  # shape: (batch_size, y, x, n_fourier_features_per_dim)
    hrvsatellite_y_osgb_fourier = "hrvsatellite_y_osgb_fourier"  # shape: (batch_size, y, x, n_fourier_features_per_dim)
    #: Time is seconds since UNIX epoch (1970-01-01). Shape: (batch_size, n_timesteps, n_fourier_features_per_dim)
    hrvsatellite_time_utc_fourier = "hrvsatellite_time_utc_fourier"

    # -------------- PV ---------------------------------------------
    pv = "pv"  # shape: (batch_size, time, n_pv_systems)
    pv_system_row_number = "pv_system_row_number"  # shape: (batch_size, n_pv_systems)
    pv_id = "pv_id"  # shape: (batch_size, n_pv_systems)
    # PV AC system capacity in watts peak.
    # Warning: In v15, pv_capacity_wp is sometimes 0. This will be fixed in
    # https://github.com/openclimatefix/nowcasting_dataset/issues/622
    pv_capacity_wp = "pv_capacity_wp"  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    pv_mask = "pv_mask"  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    pv_x_osgb = "pv_x_osgb"
    pv_y_osgb = "pv_y_osgb"
    pv_time_utc = "pv_time_utc"  # Seconds since UNIX epoch (1970-01-01).

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    pv_x_osgb_fourier = "pv_x_osgb_fourier"
    pv_y_osgb_fourier = "pv_y_osgb_fourier"
    pv_time_utc_fourier = "pv_time_utc_fourier"


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)
