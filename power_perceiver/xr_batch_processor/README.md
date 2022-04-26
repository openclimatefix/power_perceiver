## Xarray batch processors

Callable objects which have access to all modalities (each in a separate `xarray.Dataset`)
and can do processing that must be done across all modalities.
Note that processing _within_ a modality is done in `<ModalityName>.to_numpy`.

An xarray batch processor _could_ be just a function. But it's usually more convenient to
implement it as a callable object (just like PyTorch transforms) so you can configure the
batch processor.

The `__call__` method must accept an `XarrayBatch` and return an `XarrayBatch`.
`XarrayBatch` is defined in `data_loader.py` as simply
 `XarrayBatch = dict[DataLoader.__class__, xr.Dataset]`.
That is, an `XarrayBatch` is a dictionary containing the unprocessed xarray datasets which hold
data for every data loader for a given batch index.

A list of xarray batch processors is passed into `NowcastingDataset`.
