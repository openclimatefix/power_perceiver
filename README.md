# power_perceiver
Machine learning experiments using the Perceiver IO model to forecast the electricity system (starting with solar)


# Installation

```shell
conda env create -f environment.yml
conda activate power_perceiver
pip install -e .
pre-commit install
```

# Data pipeline

The data flows through several steps, in order:

1. Every `DataLoader` subclass loads a batch off disk and processes the `xr.Dataset` using the sequence of `transforms` passed into the `DataLoader`'s constructor. The processed data for every `DataLoader` goes into an `XarrayBatch`. The transforms live in `power_perceiver.transforms.<data loader name>.py`
2. `NowcastingDataset` then processes this `XarrayBatch` with its list of `xr_batch_processors`. The `xr_batch_processors` are processors which need to see across or touch multiple modalities at once.
3. Each `XarrayBatch` is then converted to a `NumpyBatch` by that `DataLoader`'s `to_numpy` method. The `to_numpy` method also normalises, converts units, etc.
4. Finally, `NowcastingDataset` passes the entire `NumpyBatch` through the sequence of `np_batch_processors`.
