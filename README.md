# Finetuning Hiera on LUNA25 3D Patches

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![Code License](https://img.shields.io/badge/code_license-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

> Derived from [Zhengro/hiera-luna25-finetuning](https://github.com/Zhengro/hiera-luna25-finetuning) (Apache-2.0). Original copyright and license are retained.

This work, participating in the [LUNA25 Challenge](https://luna25.grand-challenge.org/), finetuned Hiera for lung nodule malignancy risk estimation.

A summary of top 3 solutions in the testing phase can be found in [methods](methods.md).

## Usage
### Set Up
```
git clone https://github.com/marcizzzsr/LUNA25-classification.git

conda create -y -n venv2025 python==3.9
conda activate venv2025
python -m pip install -r LUNA25-classification/requirements.txt
```

### Get Data
```
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.001
wget https://zenodo.org/records/14223624/files/luna25_nodule_blocks.zip.002
wget https://zenodo.org/records/14673658/files/LUNA25_Public_Training_Development_Data.csv

cat luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002 > luna25_nodule_blocks.zip
apt update && apt -y install p7zip-full
7z x luna25_nodule_blocks.zip
rm luna25_nodule_blocks.zip.001 luna25_nodule_blocks.zip.002
```

### Paths
Paths default to locations relative to the repository and can be overridden with environment variables:

| Variable | Default |
|----------|---------|
| `LUNA25_DATA_DIR` | `<repo>/../luna25_nodule_blocks` |
| `LUNA25_RESOURCES_DIR` | `<repo>/resources` (pretrained weights) |
| `LUNA25_RESULTS_DIR` | `<repo>/results` |
| `LUNA25_SPLITS_DIR` | `<repo>/data_splits` |
| `LUNA25_ANNOTATION_DIR` | `<repo>/..` (all-data training) |

### Perform Training
Before starting the training, ensure that the correct `experiment_config.py` is in the project root directory.

By default, `experiment_config.py` is a copy of `configs/experiment_config_3d_hiera.py`, which represents the submitted version for the LUNA25 Challenge. To train 2D models, copy one of the available config files:
```
cp LUNA25-classification/configs/experiment_config_XXX.py LUNA25-classification/experiment_config.py
```
When training using k-fold cross validation, set `kind="k-fold"` in `experiment_config.py` then run
```
python LUNA25-classification/split_data.py
python LUNA25-classification/finetune.py --k_fold
```
When training using all data, set `kind="all-data"` in `experiment_config.py` then run
```
python LUNA25-classification/finetune.py --all_data
```

## License
This work's code is licensed under the [Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).

## References

This work has been adapted from and is inspired by the following repositories:

- [hiera-luna25-finetuning](https://github.com/Zhengro/hiera-luna25-finetuning) (original repository this work is derived from)
- [luna25-baseline-public](https://github.com/DIAGNijmegen/luna25-baseline-public)
- [hiera](https://github.com/facebookresearch/hiera)
- [SlowFast](https://github.com/facebookresearch/SlowFast)
- [torchvision](https://github.com/pytorch/vision/tree/main/torchvision)