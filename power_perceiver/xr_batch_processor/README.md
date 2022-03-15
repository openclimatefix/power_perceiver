Functions which have access to all modalities (each in a separate `xarray.Dataset`)
and can do processing that must be done across all modalities.
Note that processing _within_ a modality is done in `<ModalityName>Loader.to_numpy`
