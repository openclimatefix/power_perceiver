""" The idea is to have a pydantic model that will be used to be passed into the models """

from typing import Union

import numpy as np
import torch
from pydantic import BaseModel, Field, root_validator, validator

Array = Union[np.ndarray, torch.Tensor]


def validate_shape(shape, correct_shape, variable_name):
    assert shape == correct_shape, Exception(
        f"{variable_name} should be of dimension {correct_shape}, it actually is of dim {shape}"
    )


class BaseModelExtension(BaseModel):
    class Config:
        """Allowed classes e.g. tensor.Tensor"""

        # TODO maybe there is a better way to do this
        arbitrary_types_allowed = True


class HRVSatellite(BaseModelExtension):
    """HRV Satellite data"""

    # Hrv satellite

    hrvsatellite_actual: Array = Field(
        ...,
        description="HRV satellite data. Shape is "
        "[batch_size, timesteps, channels, height, image width]. "
        "For example this could be [32,12,1,128,256]",
    )
    hrvsatellite_t0_idx: int = Field(...)
    hrvsatellite_y_osgb: Array = Field(
        ...,
        description="The Y coordinates [OSGB] of the HRV Satellite image. "
        "Shape is # shape: [batch_size, y, x]",
    )
    hrvsatellite_x_osgb: Array = Field(
        ...,
        description="The C coordinates [OSGB] of the HRV Satellite image. "
        "Shape is # shape: [batch_size, y, x]",
    )
    hrvsatellite_y_geostationary: Array = Field(
        ...,
        description="The Y coordinates (in geo stationary coordinates)"
        " of the HRV Satellite image. "
        "Shape is # shape: [batch_size, y, x]",
    )
    hrvsatellite_x_geostationary: Array = Field(
        ...,
        description="The X coordinates (in geo stationary coordinates)"
        " of the HRV Satellite image. "
        "Shape is # shape: [batch_size, y, x]",
    )
    hrvsatellite_time_utc: Array = Field(
        ...,
        description="Time is seconds since UNIX epoch (1970-01-01). Shape: [batch_size, n_timesteps]",
    )

    @classmethod
    def v_shape(cls, v):
        validate_shape(len(v.hrvsatellite_actual), 5, "hrvsatellite_actual")
        validate_shape(len(v.hrvsatellite_y_osgb), 3, "hrvsatellite_y_osgb")
        validate_shape(len(v.hrvsatellite_x_osgb), 3, "hrvsatellite_x_osgb")
        validate_shape(len(v.hrvsatellite_y_geostationary), 3, "hrvsatellite_y_osgb")
        validate_shape(len(v.hrvsatellite_x_geostationary), 3, "hrvsatellite_x_osgb")

    # TODO
    # - validate all variables have same batch_size
    # - validate date has same y dims
    # - validate date has same x dims
    # - validate date has same time dims

    # validate x and y coordinates in correct range
    # validate hrvsatellite_actual values in correct range


class NWP(BaseModelExtension):
    """NWP data"""

    nwp: Array = Field(
        ...,
        description="The NWP data. The shape is  [batch_size, target_time_utc, channel, y_osgb, x_osgb]",
    )
    nwp_t0_idx: int = Field(
        ..., description="The t0 of the data. The time when the data is available"
    )
    nwp_target_time_utc: Array = Field(
        ...,
        description="The target time of the nwp data. The shape is batch_size, target_time_utc]",
    )
    nwp_init_time_utc: Array = Field(
        ..., description="The init time of the nwp data. The shape is batch_size, target_time_utc]"
    )
    nwp_y_osgb: Array = Field(
        ..., description="The Y coordinates [OSGB] of the data. The shape is [batch_size, y_osgb]. "
    )
    nwp_x_osgb: Array = Field(
        ..., description="The X coordinates [OSGB] of the data. The shape is [batch_size, x_osgb]. "
    )

    @classmethod
    def v_shape(cls, v):
        validate_shape(len(v.nwp), 5, "nwp")
        validate_shape(len(v.nwp_target_time_utc), 2, "nwp_target_time_utc")
        validate_shape(len(v.nwp_init_time_utc), 2, "nwp_init_time_utc")
        validate_shape(len(v.nwp_y_osgb), 2, "nwp_y_osgb")
        validate_shape(len(v.nwp_x_osgb), 2, "nwp_x_osgb")

    # TODO
    # - validate all variables have same batch_size
    # - validate date has same y dims
    # - validate date has same x dims
    # - validate date has same time dims

    # validate x and y coordinates in correct range
    # validate nwp values in correct range


class PV(BaseModelExtension):
    """PV data"""

    pv: Array = Field(
        ...,
        description="The PV data from that region. The shape is [batch_size, time, n_pv_systems]",
    )
    pv_t0_idx: int = Field(..., description="The t0 time of the PV data")
    pv_system_row_number: Array = Field(
        ..., description="The row number of the pv system. The shape is [batch_size, n_pv_systems]"
    )
    pv_y_osgb: Array = Field(
        ...,
        description="The Y coordinates [OSGB] of the data. The shape is [batch_size, n_pv_systems ].",
    )
    pv_x_osgb: Array = Field(
        ...,
        description="The Y coordinates [OSGB] of the data. The shape is [batch_size, n_pv_systems ].",
    )
    pv_time_utc: Array = Field(
        ...,
        description="The time of the PV data. Seconds since UNIX epoch (1970-01-01). "
        "The shape is [batch_size, time]",
    )

    # TODO
    # - validate all variables have same batch_size
    # - validate date has same n_pv_systems dims
    # - validate date has same time dims

    # validate x and y coordinates in correct range
    # validate pv values in correct range

    @classmethod
    def v_shape(cls, v):
        validate_shape(len(v.pv), 3, "pv")
        validate_shape(len(v.pv_system_row_number), 2, "pv_system_row_number")
        validate_shape(len(v.pv_y_osgb), 2, "pv_y_osgb")
        validate_shape(len(v.pv_x_osgb), 2, "pv_x_osgb")
        validate_shape(len(v.pv_time_utc), 2, "pv_time_utc")


class GSP(BaseModelExtension):
    """GSP data"""

    gsp: Array = Field(..., description='The GSP data.  The shape is [batch_size, time, 1]"')
    gsp_t0_idx: int = Field(..., description="The t0 time of the GSP data")
    gsp_id: Array = Field(
        ..., description="The gsp id of the gsp system. The shape is [batch_size]"
    )
    gsp_y_osgb: Array = Field(
        ..., description="The x coordinates of the gsp system. The shape is [batch_size]"
    )
    gsp_x_osgb: Array = Field(
        ..., description="The y coordinates of the gsp system. The shape is [batch_size]"
    )
    gsp_time_utc: Array = Field(
        ...,
        description="The time of the GSP data. Seconds since UNIX epoch (1970-01-01). "
        "The shape is [batch_size, time]",
    )

    @classmethod
    def v_shape(cls, v):
        validate_shape(len(v.gsp), 3, "gsp")  # This might be two, need to check what works
        validate_shape(len(v.gsp_id), 1, "gsp_id")
        validate_shape(len(v.gsp_y_osgb), 1, "gsp_y_osgb")
        validate_shape(len(v.gsp_x_osgb), 1, "gsp_x_osgb")
        validate_shape(len(v.gsp_time_utc), 1, "gsp_time_utc")

    # TODO
    # - validate all variables have same batch_size
    # - validate date has same n_pv_systems dims
    # - validate date has same time dims

    # validate x and y coordinates in correct range
    # validate gsp values in correct range


def get_batch(hrvsatelle: HRVSatellite, nwp: NWP, pv: PV, gsp: GSP) -> dict:
    """
    Get flat dictionary from various data sources

    Merge together the following data sources into a flat dict
    - hrvsatellite
    - nwp
    - pv
    - gsp

    """
    return {**hrvsatelle.__dict__, **nwp.__dict__, **pv.__dict__, **gsp.__dict__}
