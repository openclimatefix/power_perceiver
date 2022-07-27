""" The idea is to have a pydantic model that will be used to be passed into the models """

from typing import Union

import numpy as np
import torch
from pydantic import BaseModel, Field, root_validator

Array = Union[np.ndarray, torch.Tensor]


def validate_shape(shape, correct_shape, variable_name):
    """Validate shape is what is should be"""
    assert shape == correct_shape, Exception(
        f"{variable_name} should be of dimension {correct_shape}, it actually is of dim {shape}"
    )


def assert_values_ge_value(values: np.ndarray, min_value, variable_name):
    """Validate values are great than a certain value"""
    if (values < min_value).any():
        message = f"Some variable_name data values are less than {min_value}. "
        message += f"The minimum value is {values.min()}. "
        if variable_name is not None:
            message += f" ({variable_name})"
        raise Exception(message)


class BaseModelExtension(BaseModel):
    class Config:
        """Allowed classes e.g. tensor.Tensor"""

        # TODO maybe there is a better way to do this
        arbitrary_types_allowed = True
        validate_assignment = True


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
        description="Time is seconds since UNIX epoch (1970-01-01). "
        "Shape: [batch_size, n_timesteps]",
    )

    @root_validator
    def v_shape(cls, v):
        expected_dims = {
            "hrvsatellite_actual": 5,
            "hrvsatellite_y_osgb": 3,
            "hrvsatellite_x_osgb": 3,
            "hrvsatellite_y_geostationary": 3,
            "hrvsatellite_x_geostationary": 3,
        }

        for key, item in expected_dims.items():
            validate_shape(len(v[key].shape), item, key)

        return v

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
        description="The NWP data. The shape is  "
        "[batch_size, target_time_utc, channel, y_osgb, x_osgb]",
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

    @root_validator
    def v_shape(cls, v):

        expected_dims = {
            "nwp": 5,
            "nwp_target_time_utc": 2,
            "nwp_init_time_utc": 2,
            "nwp_y_osgb": 2,
            "nwp_x_osgb": 2,
        }

        for key, item in expected_dims.items():
            validate_shape(len(v[key].shape), item, key)

        return v

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
        description="The PV data from that region."
        " The shape is [batch_size, time, n_pv_systems]",
    )
    pv_t0_idx: int = Field(..., description="The t0 time of the PV data")
    pv_system_row_number: Array = Field(
        ...,
        description="The row number of the pv system. " "The shape is [batch_size, n_pv_systems]",
    )
    pv_y_osgb: Array = Field(
        ...,
        description="The Y coordinates [OSGB] of the data. "
        "The shape is [batch_size, n_pv_systems ].",
    )
    pv_x_osgb: Array = Field(
        ...,
        description="The Y coordinates [OSGB] of the data. "
        "The shape is [batch_size, n_pv_systems ].",
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

    @root_validator
    def v_shape(cls, v):

        expected_dims = {
            "pv": 3,
            "pv_system_row_number": 2,
            "pv_y_osgb": 2,
            "pv_x_osgb": 2,
            "pv_time_utc": 2,
        }

        for key, item in expected_dims.items():
            validate_shape(len(v[key].shape), item, key)

        return v

    @root_validator
    def v_values_constraints(cls, v):
        """Check fields are greater than certain values"""

        assert_values_ge_value(values=v["pv_x_osgb"], min_value=0, variable_name="pv_x_osgb")
        assert_values_ge_value(values=v["pv_y_osgb"], min_value=0, variable_name="pv_y_osgb")

        return v


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

    @root_validator
    def v_shape(cls, v):

        expected_dims = {
            "gsp": 3,
            "gsp_id": 1,
            "gsp_y_osgb": 1,
            "gsp_x_osgb": 1,
            "gsp_time_utc": 2,
        }

        for key, item in expected_dims.items():
            validate_shape(len(v[key].shape), item, key)

        return v

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
