# power_perceiver
Machine learning experiments for forecasting the electricity system (starting with solar)


# Installation

## Installation with conda
We recommend installing [mamba](https://github.com/mamba-org/mamba) and using `mamba env create -f base_environment.yml` instead of `conda env create -f base_environment.yml`.

If installing on a platform without a GPU, then uncomment `- cpuonly` in `base_environment.yml`.

```shell
conda env create -f base_environment.yml
conda activate power_perceiver

# If training, then also install the dependencies listed in train_environment.yml:
# See https://stackoverflow.com/a/43873901/732596
conda env update --file train_environment.yml --prune

pip install -e .
pre-commit install
```

If using `Ranger21` optimizer then please install [Ranger21 with my tiny little patch](https://github.com/JackKelly/Ranger21/tree/patch-1).

To prevent `mamba update --all` from trying to replace the GPU version of PyTorch with the CPU version,
add this to `~/miniconda3/envs/power_perceiver/conda-meta/pinned`:

```
# Prevent mamba update --all from trying to install CPU version of torch.
# See: https://stackoverflow.com/a/70536292/732596
cudatoolkit<11.6
```

## Installation with pip only
To install the base config, use: `pip install -e .`

To install the code necessary to train, use: `pip install -e .[develop,train]`

# Data pipelines

There are two different data pipelines:

- `power_perceiver.load_prepared_batches`: Loads batches pre-prepared by `nowcasting_dataset`
- `power_perceiver.load_raw`: Loads raw (well, intermediate) data

## Data pipeline for data prepared by `nowcasting_dataset`

The data flows through several steps, in order:

1. Every `PreparedDataSource` subclass loads a batch off disk and processes the `xr.Dataset` using the sequence of `transforms` passed into the `PreparedDataSource`'s constructor. The processed data for every `PreparedDataSource` goes into an `XarrayBatch`. The transforms live in `power_perceiver.transforms.<data source name>.py`
2. `PreparedDataset` then processes this `XarrayBatch` with its list of `xr_batch_processors`. The `xr_batch_processors` are processors which need to see across or touch multiple modalities at once while the data is still in an xarray Dataset.
3. Each `XarrayBatch` is then converted to a `NumpyBatch` by that `PreparedDataSource`'s `to_numpy` method. The `to_numpy` method also normalises, converts units, etc.
4. Finally, `PreparedDataset` passes the entire `NumpyBatch` through the sequence of `np_batch_processors`.

# About the name "power perceiver"

Originally, when I started work on "Power Perceiver" 5 months ago, my intention was to use [DeepMind's Perceiver IO](https://www.deepmind.com/open-source/perceiver-io) at the core of the model. Right now, the model actually just uses a standard transformer encoder, not a Perceiver. But I plan to start using a Perceiver IO again within a month or two, when we start using more input elements than a standard transformer encoder can cope with!
