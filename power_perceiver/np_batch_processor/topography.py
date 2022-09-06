"""At the moment, it's actually quite tricky to align the topo data with the satellite data
(because xarray.align doesn't like the fact that the satellite data's OSGB coords are each 2D.
So we have to reproject the topo data first, and then align it).

This will no longer be necessary if / when this is implemented:
https://github.com/openclimatefix/nowcasting_dataset/issues/642

The strategy is:

1. Load all the topo data into memory.
2. Reproject it from OSGB to geostationary CRS. We have to do this because the satellite
   data is natively in geostationary CRS. Yes, the satellite data comes with OSGB coords,
   but those are 2D, and xr.combine_by_coords doesn't work with 2D coords. So we have to
   reproject the topo data to geostationary so we end up with both the topo data and the
   satellite data having "normal" 1D coords.
3. Then, for each satellite example, use xarray.combine_by_coords twice:
   1. An "outer join" to get a DataArray with the coords from both the satellite
      data and the topo data. This results in topo data with lots of NaNs! So we
      forward-fill and backwards-fill the NaNs.
    2. A "right join" to select just the coords that exactly match the satellite data's
       geostationary coords.
"""

from dataclasses import dataclass

import cartopy.crs as ccrs
import numpy as np
import pyproj
import pyresample
import rioxarray
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch


@dataclass
class Topography:
    topo_filename: str

    def __post_init__(self):
        self.topo = load_topo_data(self.topo_filename)
        self.topo = reproject_topo_data_from_osgb_to_geostationary(self.topo)
        self.topo = self.topo.fillna(0)
        self.topo_mean = self.topo.mean().item()
        self.topo_std = self.topo.std().item()

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        if BatchKey.hrvsatellite_x_geostationary in np_batch:
            # Recreate an xr.DataArray of the satellite data. This is required so we can
            # use xr.combine_by_coords to align the topo data with the satellite data.
            hrvsatellite_data_array = xr.DataArray(
                # We're not actually interested in the image. But xarray won't make an
                # empty DataArray without data in the right shape.
                # There's nothing special about the x_osgb data. It's just convenient because
                # it's the right shape!
                np_batch[BatchKey.hrvsatellite_x_osgb],
                dims=("example", "y", "x"),
                coords={
                    "y_geostationary": (
                        ("example", "y"),
                        np_batch[BatchKey.hrvsatellite_y_geostationary],
                    ),
                    "x_geostationary": (
                        ("example", "x"),
                        np_batch[BatchKey.hrvsatellite_x_geostationary],
                    ),
                },
            )
            hrvsatellite_surface_height = _get_surface_height_for_satellite(
                surface_height=self.topo, satellite=hrvsatellite_data_array
            )
            hrvsatellite_surface_height = self._normalise(hrvsatellite_surface_height)
            np_batch[BatchKey.hrvsatellite_surface_height] = hrvsatellite_surface_height

        return np_batch

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        data = np.nan_to_num(data, nan=0.0)
        data -= self.topo_mean
        data /= self.topo_std
        return data


def load_topo_data(filename: str) -> xr.DataArray:
    topo = rioxarray.open_rasterio(filename=filename, parse_coordinates=True, masked=True)

    # `band` and `spatial_ref` don't appear to hold any useful info. So get rid of them:
    topo = topo.isel(band=0)
    topo = topo.drop_vars(["spatial_ref", "band"])

    # Use our standard naming:
    topo = topo.rename({"x": "x_osgb", "y": "y_osgb"})

    # Select Western Europe:
    topo = topo.sel(x_osgb=slice(-300_000, 1_500_000), y_osgb=slice(1_300_000, -800_000))

    return topo


def reproject_topo_data_from_osgb_to_geostationary(topo: xr.DataArray) -> xr.DataArray:
    topo_osgb_area_def = _get_topo_osgb_area_def(topo)
    topo_geostationary_area_def = _get_topo_geostationary_area_def(topo)
    topo_image = pyresample.image.ImageContainerQuick(topo.values, topo_osgb_area_def)
    topo_image_resampled = topo_image.resample(topo_geostationary_area_def)
    return _get_data_array_of_resampled_topo_image(topo_image_resampled)


def _get_topo_osgb_area_def(topo: xr.DataArray) -> pyresample.geometry.AreaDefinition:
    # Get AreaDefinition of the OSGB topographical data:
    osgb = ccrs.OSGB(approx=False)
    return pyresample.create_area_def(
        area_id="OSGB",
        projection=osgb.proj4_params,
        shape=topo.shape,  # y, x
        area_extent=(
            topo.x_osgb[0].item(),  # lower_left_x
            topo.y_osgb[-1].item(),  # lower_left_y
            topo.x_osgb[-1].item(),  # upper_right_x
            topo.y_osgb[0].item(),  # upper_right_y
        ),
    )


def _get_topo_geostationary_area_def(topo: xr.DataArray) -> pyresample.geometry.AreaDefinition:
    # Get the geostationary boundaries of the topo data:
    OSGB_EPSG_CODE = 27700
    GEOSTATIONARY_PROJ = {
        "proj": "geos",
        "lon_0": 9.5,
        "h": 35785831,
        "x_0": 0,
        "y_0": 0,
        "a": 6378169,
        "rf": 295.488065897014,
        "no_defs": None,
        "type": "crs",
    }
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=OSGB_EPSG_CODE, crs_to=GEOSTATIONARY_PROJ
    ).transform
    lower_left_geos = osgb_to_geostationary(xx=topo.x_osgb[0], yy=topo.y_osgb[-1])
    upper_right_geos = osgb_to_geostationary(xx=topo.x_osgb[-1], yy=topo.y_osgb[0])
    shape = (topo.shape[0] * 2, topo.shape[1] * 2)  # Oversample to make sure we don't loose info.
    return pyresample.create_area_def(
        area_id="msg_seviri_rss_1km",
        projection=GEOSTATIONARY_PROJ,
        shape=shape,
        # lower_left_x, lower_left_y, upper_right_x, upper_right_y:
        area_extent=lower_left_geos + upper_right_geos,
    )


def _get_data_array_of_resampled_topo_image(
    topo_image_resampled: pyresample.image.ImageContainer,
) -> xr.DataArray:
    (
        lower_left_x,
        lower_left_y,
        upper_right_x,
        upper_right_y,
    ) = topo_image_resampled.geo_def.area_extent
    return xr.DataArray(
        topo_image_resampled.image_data,
        coords=(
            (
                "y",
                np.linspace(
                    start=upper_right_y, stop=lower_left_y, num=topo_image_resampled.shape[0]
                ),
            ),
            (
                "x",
                np.linspace(
                    start=lower_left_x, stop=upper_right_x, num=topo_image_resampled.shape[1]
                ),
            ),
        ),
    )


def _get_surface_height_for_satellite(
    surface_height: xr.DataArray, satellite: xr.DataArray
) -> np.ndarray:
    num_examples = satellite.shape[0]
    surface_height = surface_height.rename("surface_height")
    surface_height_for_batch = np.full_like(satellite.values, fill_value=np.NaN)
    for example_idx in range(num_examples):
        satellite_example = satellite.isel(example=example_idx)
        msg = "Satellite imagery must start in the top-left!"
        assert satellite_example.y_geostationary[0] > satellite_example.y_geostationary[-1], msg
        assert satellite_example.x_geostationary[0] < satellite_example.x_geostationary[-1], msg

        satellite_example = satellite_example.rename(
            {"y_geostationary": "y", "x_geostationary": "x"}
        ).rename("sat")
        surface_height_for_example = surface_height.sel(
            y=slice(
                satellite_example.y[0],
                satellite_example.y[-1],
            ),
            x=slice(
                satellite_example.x[0],
                satellite_example.x[-1],
            ),
        )

        # Align by coordinates. This will result in lots of NaNs in the surface height data:
        aligned = xr.combine_by_coords(
            (surface_height_for_example, satellite_example), join="outer"
        )

        # Fill in the NaNs:
        surface_height_for_example = (
            aligned["surface_height"].ffill("x").ffill("y").bfill("x").bfill("y")
        )

        # Now select exactly the same coordinates from the surface height data as the satellite data
        aligned = xr.combine_by_coords((surface_height_for_example, satellite_example), join="left")

        surface_height_for_batch[example_idx] = aligned["surface_height"].values

    # If we slightly ran off the edge of the topo data then we'll get NaNs.
    # TODO: Enlarge topo data so we never get NaNs!
    surface_height_for_batch = np.nan_to_num(surface_height_for_batch, nan=0)

    return surface_height_for_batch
