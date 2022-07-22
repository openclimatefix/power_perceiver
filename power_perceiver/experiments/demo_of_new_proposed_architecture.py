from dataclasses import dataclass


@dataclass
class DataSource:
    pre_processors: list
    post_processors: list


class ProductionPVDataSource(DataSource):
    pass


class SatelliteDataSource(DataSource):
    pass


class RemoveBadPVSystems:
    pass


class FillNighttimePVWithNaNs:
    pass


@dataclass
class NowcastingProductionDataset:
    data_sources: list


class FourierEncodeRelativeSpaceTime:
    pass


class SunPosition:
    pass


class PVNet:
    pass


@dataclass
class RunModel:
    model: PVNet


class DisableForecastsRunAtNight:
    pass


@dataclass
class ClipPredictedPowerToZeroAtNight:
    elevation_threshold_degrees: int


class SaveForecastsToDB:
    pass


def run_data_pipeline(pipeline: list, space_time_location_selector):
    pass


class Normalize:
    pass


class SaveBatchesToDisk:
    pass


class NormalizePV:
    pass


class EncodePVSystemsWithGSPID:
    pass


class Select15MinSatellite:
    pass


class NormalizeSatellite:
    pass


class PatchSatellite:
    pass


class EncodePixelsWithGSPID:
    pass


def now():
    pass


"""
Design objectives:

- Make the code as modular as possible.
- Make it super-easy to see all the data processing steps in one glance.
- Make it easy to configure different data processing for different models.
- Share code across training and production:
  - Data processors
  - DataSources that load from Zarr.

Definitely haven't fully thought this through yet!
"""

sat_xr_dataset = OpenSatelliteZarr(sat_zarr_path)
# OpenSatelliteZarr returns the entire, lazily-opened satellite dataset,
# and does any renaming and transposing necessary to get the data into the
# "standard" form. I've put some thought into what that "standard" form could be. Please see:
# https://github.com/openclimatefix/power_perceiver/blob/main/power_perceiver/load_raw/data_sources/raw_satellite_data_source.py#L147

pv_xr_dataset = LoadPVFromDB(db_credentials)
# Returns an xr.DataArray of all the PV data in RAM.
# Replace `LoadPVFromDB` with LoadPVFromNetCDF when training. Both output data in exactly the same shape.
pv_xr_dataset = remove_bad_pv_systems(pv_xr_dataset)

############# GET LIST OF ALL AVAILABLE T0 DATETIMES, ACROSS MODALITIES ##################
# This only needs to be done once (not for every example).
# `sat_t0_datetimes` and `pv_t0_datetimes` could just be `pd.DatetimeIndex` objects.
sat_t0_datetimes = get_all_available_t0_datetimes(
    sat_xr_dataset,
    history_duration=config.satellite.history_duration,
    forecast_duration=config.satellite.forecast_duration,
)
pv_t0_datetimes = get_all_available_t0_datetimes(
    pv_xr_dataset,
    history_duration=config.pv.history_duration,
    forecast_duration=config.pv.forecast_duration,
)

# Find all available t0 datetimes across all modalities:
available_t0_datetimes = sat_t0_datetimes.intersection(pv_t0_datetimes)

# The PV and Satellite pipes both need to be told which locations to load during training:
# This sends the following instructions to the data loaders:
# Batch 1: Load super-batch 0 into RAM. When done, yield example at t0 & location x,y.
#          Then, in the background, load super-batch 1.
# Batch 2: Yield example at t0, location x, y
# <last batch of the epoch>: Switch to super-batch 1. (Delete super-batch 0 from RAM).
# In the background, load super-batch 2. Yield example at t0, location x, y.
# Replace T0PickerForSuperBatchLoader when in production, if when training and you don't want to load super batches.
t0 = T0PickerForSuperBatchLoader(
    available_t0_datetimes=available_t0_datetimes,
    t0_freq="15T",
    locations_per_timestep=4,
)
t0_for_pv, t0_for_sat = Forker(t0, num_instances=2, buffer_size=0)

# Select random locations for each example. Center each example on a random PV system:
location = LocationPicker(locations=pv_xr_dataset)

# -------------- PV DataPipe ------------
pv_pipe = SelectTimeSlice(
    pv_xr_dataset, t0_for_pv
)  # Forwards the t0_datetime & location to subsequent steps.
# Maybe it starts to populate an Example object, where Example.pv is the PV xr.DataArray; and Example.pv_t0_time_utc, Example.center_lat, and Example.center_lon specify the location?
# New: Split the history and forecast into two separate objects. e.g. BatchML.inputs.pv and BatchML.targets.pv?
# Or maybe that adds more complexity???
pv_pipe = SelectPVSystemsWithinRegion(
    pv_pipe,
    location=location,
    roi_width_km=config.pv.roi.width_km,
    roi_height_km=config.pv.roi.height_km,
)
pv_pipe = FillNighttimePVWithNaNs(pv_pipe)
pv_pipe = InterpolateMissingPV(pv_pipe)
pv_pipe = NormalizePV(pv_pipe)
pv_pipe = SunPosition(pv_pipe)

# Validation:
# In this example, don't check for NaNs in PV, because the model expects missing PV to be represented as NaNs.
pv_pipe = CheckStatisticalProperties(pv_pipe)
pv_pipe = CheckPVShape(pv_pipe)

# -------------- Satellite DataPipe ---------
sat_pipe = LoadSuperBatchIntoRAM(sat_xr_dataset, t0_for_sat)  # Optional. Useful during training.
# LoadSuperBatchIntoRAM speeds up training by caching a subset of the on-disk dataset into RAM,
# specified by SpateTimeLocationPicker.
sat_pipe = SelectTimeSlice(sat_pipe)
sat_pipe = SelectSpatialSlice(sat_pipe)
sat_pipe = NormalizeSatellite(sat_pipe)
sat_pipe = PatchSatellite(sat_pipe)
sat_pipe = SunPosition(sat_pipe)

# Validation:
sat_pipe = CheckForNaNs(sat_pipe)
sat_pipe = CheckStatisticalProperties(sat_pipe)
sat_pipe = CheckSatShape(sat_pipe)

# -------------- Merge & process -----------------
main_pipe = MergeBatchML(pv_pipe, sat_pipe)

main_pipe = FourierEncodeRelativeSpaceTime(main_pipe)

# Saving batches to disk is entirely optional during production,
# where it may be useful for debugging:
# During training, SaveBatchesToDisk can be used to save pre-prepared batches.
# And then `LoadBatchesFromDisk()` can be used to load batches into this pipeline.
main_pipe = SaveBatchesToDisk(main_pipe)

# Run ML model:
main_pipe = RunModel(main_pipe, model=PVNet())

# Post-process output from model:
main_pipe = ClipPredictedPowerToZeroAtNight(main_pipe, elevation_threshold_degrees=5)
main_pipe = SaveForecastsToDB(main_pipe)
