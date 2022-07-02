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

# The PV and Satellite pipes both need to be told which locations to load during training:
# This sends the following instructoins to the data loaders:
# Batch 1: Load super-batch 0 into RAM. When done, yield example at t0 & location x,y.
#          The, in the background, load super-batch 1.
# Batch 2: Yield exmaple at t0, location x, y
# <last batch of the epoch>: Switch to super-batch 1. (Delete super-batch 0 from RAM).
# In the background, load super-batch 2. Yield example at t0, location x, y.
space_time_locations = SpaceTimeLocationPicker(
    data_sources=(SatelliteMetaData(), PVMetaData()),
    t0_freq="15T",
)
space_time_locations = DisableForecastsRunAtNight(space_time_locations)

space_time_loc_1, space_time_loc_2 = Forker(space_time_locations, num_instances=2, buffer_size=0)

# -------------- PV DataPipe ------------
pv_pipe = LoadPVFromDB(space_time_locations, pv_config)
# Replace `LoadPVFromDB` with LoadPVFromNetCDF when training. Both output data in exactly the same shape.
pv_pipe = FillNighttimePVWithNaNs(pv_pipe)
pv_pipe = InterpolateMissingPV(pv_pipe)
pv_pipe = NormalizePV(pv_pipe)
pv_pipe = SunPosition(pv_pipe)


# -------------- Satellite DataPipe ---------
sat_pipe = OpenSatelliteZarr(space_time_locations, sat_config)
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
