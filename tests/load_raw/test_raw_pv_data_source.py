from power_perceiver.load_raw.data_sources.raw_pv_data_source import _load_pv_metadata

# TODO: Use public data :)
PV_METADATA_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv"


def test_load_pv_metadata():
    pv_metadata = _load_pv_metadata(PV_METADATA_FILENAME)
    print(pv_metadata.head())
    print(pv_metadata.columns)
    print(pv_metadata.dtypes)
