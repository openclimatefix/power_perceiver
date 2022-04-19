"""Constants and Enums."""

from enum import Enum, auto

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

    # -------------- HRVSATELLITE -----------------------------------
    # shape: (batch_size, time, channels, y, x)
    #
    # Or, if the imagery has been patched,
    # shape: (batch_size, time, channels, y, x, n_pixels_per_patch) where n_pixels_per_patch
    # is the *total* number of pixels,
    # i.e. n_pixels_per_patch_along_height * n_pixels_per_patch_along_width.
    hrvsatellite = auto()

    # HRV satellite coordinates:
    hrvsatellite_y_osgb = auto()  # shape: (batch_size, y, x)
    hrvsatellite_x_osgb = auto()  # shape: (batch_size, y, x)
    hrvsatellite_y_geostationary = auto()  # shape: (batch_size, y)
    hrvsatellite_x_geostationary = auto()  # shape: (batch_size, x)
    #: Time is seconds since UNIX epoch (1970-01-01). Shape: (batch_size, n_timesteps)
    hrvsatellite_time_utc = auto()
    # Added by np_batch_processor.Topography:
    hrvsatellite_surface_height = auto()  # The surface height at each pixel. (batch_size, y, x)

    # HRV satellite Fourier coordinates:
    # Spatial coordinates. Shape: (batch_size, y, x, n_fourier_features_per_dim)
    hrvsatellite_y_osgb_fourier = auto()
    hrvsatellite_x_osgb_fourier = auto()
    #: Time shape: (batch_size, n_timesteps, n_fourier_features_per_dim)
    hrvsatellite_time_utc_fourier = auto()

    # -------------- PV ---------------------------------------------
    pv = auto()  # shape: (batch_size, time, n_pv_systems)
    pv_system_row_number = auto()  # shape: (batch_size, n_pv_systems)
    pv_id = auto()  # shape: (batch_size, n_pv_systems)
    # PV AC system capacity in watts peak.
    # Warning: In v15, pv_capacity_wp is sometimes 0. This will be fixed in
    # https://github.com/openclimatefix/nowcasting_dataset/issues/622
    pv_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    pv_mask = auto()  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    pv_y_osgb = auto()
    pv_x_osgb = auto()
    pv_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01).
    # Added by np_batch_processor.Topography:
    pv_surface_height = auto()  # The surface height at the location of the PV system.

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    pv_y_osgb_fourier = auto()
    pv_x_osgb_fourier = auto()
    pv_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)

    # -------------- SUN --------------------------------------------
    # shape = (batch_size, n_timesteps)
    solar_azimuth = auto()
    solar_elevation = auto()


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data"
)
