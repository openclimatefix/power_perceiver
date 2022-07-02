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

sat_loader = OpenSatelliteZarr(sat_config)
# OpenSatelliteZarr yields the entire, lazily-opened satellite dataset
pv_loader = LoadPVFromDB(
    pipeline_to_run_when_loading_into_ram=[RemoveBadPVSystems()]
    )
# LoadPV yields an xr.DataArray of all the PV data in RAM.

# The PV and Satellite pipes both need to be told which locations to load during training:
# This sends the following instructoins to the data loaders:
# Batch 1: Load super-batch 0 into RAM. When done, yield example at t0 & location x,y.
#          The, in the background, load super-batch 1.
# Batch 2: Yield exmaple at t0, location x, y
# <last batch of the epoch>: Switch to super-batch 1. (Delete super-batch 0 from RAM).
# In the background, load super-batch 2. Yield example at t0, location x, y.
space_time_locations = SpaceTimeLocationPicker(
    data_sources=(Ssat_loader, pv_loader),
    t0_freq="15T",
)
space_time_locations = DisableForecastsRunAtNight(space_time_locations)

space_time_loc_1, space_time_loc_2 = Forker(space_time_locations, num_instances=2, buffer_size=0)

# -------------- PV DataPipe ------------
# Replace `LoadPVFromDB` with LoadPVFromNetCDF when training. Both output data in exactly the same shape.
pv_pipe = SelectTimeSlice(pv_loader, space_time_locations)  # Forwards the t0_datetime & location.
pv_pipe = SelectPVSystemsWithinRegion(
    pv_pipe,
    roi_width_km=config.pv.roi.width_km,
    roi_height_km=config.pv.roi.height_km,
    )
pv_pipe = FillNighttimePVWithNaNs(pv_pipe)
pv_pipe = InterpolateMissingPV(pv_pipe)
pv_pipe = NormalizePV(pv_pipe)
pv_pipe = SunPosition(pv_pipe)


# -------------- Satellite DataPipe ---------
sat_pipe = LoadSuperBatchIntoRAM(sat_loader, space_time_locations)  # Optional. Useful during training.
# LoadSuperBatchIntoRAM speeds up training by caching a subset of the on-disk dataset into RAM,
# specified by SpateTimeLocationPicker.
sat_pipe = SelectTimeSlice(sat_pipe)
sat_pipe = SelectSpatialSlice(sat_pipe)
sat_pipe = NormalizeSatellite(sat_pipe)
sat_pipe = PatchSatellite(sat_pipe)
sat_pipe = SunPosition(sat_pipe)

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
