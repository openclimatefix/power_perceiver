# power_perceiver
Machine learning experiments using the Perceiver IO model to forecast the electricity system (starting with solar)


# Installation

We recommend installing [mamba](https://github.com/mamba-org/mamba) and using `mamba env create -f environment.yml` instead of `conda env create -f environment.yml`.

```shell
conda env create -f environment.yml
conda activate power_perceiver
pip install -e .
pre-commit install
```

If using `Ranger21` optimizer then please install [Ranger21 with my tiny little patch](https://github.com/JackKelly/Ranger21/tree/patch-1).

# Data pipeline

The data flows through several steps, in order:

1. Every `DataLoader` subclass loads a batch off disk and processes the `xr.Dataset` using the sequence of `transforms` passed into the `DataLoader`'s constructor. The processed data for every `DataLoader` goes into an `XarrayBatch`. The transforms live in `power_perceiver.transforms.<data loader name>.py`
2. `NowcastingDataset` then processes this `XarrayBatch` with its list of `xr_batch_processors`. The `xr_batch_processors` are processors which need to see across or touch multiple modalities at once while the data is still in an xarray Dataset.
3. Each `XarrayBatch` is then converted to a `NumpyBatch` by that `DataLoader`'s `to_numpy` method. The `to_numpy` method also normalises, converts units, etc.
4. Finally, `NowcastingDataset` passes the entire `NumpyBatch` through the sequence of `np_batch_processors`.
