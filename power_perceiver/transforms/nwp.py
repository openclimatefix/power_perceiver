from dataclasses import dataclass
from typing import Union

import xarray as xr

from power_perceiver.time import set_new_sample_period_and_t0_idx_attrs


@dataclass
class NWPInterpolate:
    """Interpolate NWPs.

    The xr.Dataset is modified in-place, and is returned.

    Initialisation arguments:
        freq:
    """

    freq: str = "30T"
    kind: str = "cubic"

    def __call__(self, xr_data: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        xr_data = xr_data.resample(target_time_utc=self.freq).interpolate(kind=self.kind)
        xr_data = set_new_sample_period_and_t0_idx_attrs(xr_data, new_sample_period=self.freq)
        return xr_data
