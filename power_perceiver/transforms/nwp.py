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
        resampled = xr_data.resample(target_time_utc=self.freq).interpolate(kind=self.kind)
        # Resampling removes the attributes, so put them back:
        for attr_name in ("t0_idx", "sample_period_duration"):
            resampled.attrs[attr_name] = xr_data.attrs[attr_name]
        resampled = set_new_sample_period_and_t0_idx_attrs(resampled, new_sample_period=self.freq)
        return resampled
