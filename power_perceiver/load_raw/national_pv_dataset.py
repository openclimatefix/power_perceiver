import datetime
import logging

from power_perceiver.consts import Location
from power_perceiver.load_raw.raw_dataset import RawDataset

_log = logging.getLogger(__name__)


class NationalPVDataset(RawDataset):
    """Only pass in a single `data_source_combo` (with GSP as the first entry.)"""

    def _epoch_start(self) -> None:
        super()._epoch_start()
        self._reset()
        self._gsp_data_source = self.data_source_combos[self._gsp_combo_name][0]
        self._num_gsps_total = self._gsp_data_source.num_gsps

    def _choose_osgb_location(self, chosen_combo_name: str) -> Location:
        loc = self._gsp_data_source.get_osgb_location_for_gsp_idx(self._gsp_idx)
        if self._gsp_idx < self._num_gsps_total - 1:
            self._gsp_idx += 1
        else:
            self._reset()
        return loc

    def _reset(self) -> None:
        self._gsp_idx = 0
        self._t0_datetime = super()._choose_t0_datetime(self._gsp_combo_name)
        _log.info("Resetting NationalPVDataset!")

    def _choose_t0_datetime(self, chosen_combo_name: str) -> datetime.datetime:
        return self._t0_datetime

    @property
    def _gsp_combo_name(self) -> str:
        return list(self.data_source_combos.keys())[0]
