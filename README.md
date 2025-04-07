## Enviroment (Conda based)

+ Python 3.12.9 (dev) 3.11.11 (op)
+ Target OS: Linux (No support for windows)
+ Operate: Ubuntu 22.04.4 LTS x86_64 (Google colab Q1 2025)
+ Develop: Debian GNU/Linux 12 x86_64
+ CUDA Version: 12.*

## Dependencies

See `environment.yml`

## Setup

1. install conda/ miniconda into your distribution
2. create conda environment from `environemnt.yml`
```
conda env update --file ./environment.yml --prune
```
3. activate conda environment
```
conda activate sra-env
```
4. Run setup
```
python setup.py
```

## Process & implement steps for new dataset

### Flow map

+ For each dataset:
    - Download dataset
        - From kagglehub
        - Or download zipfile
    - Filter desired (in system used) sound class, omit unused
    - Move used sound files (datapoints) into main dataset path
    - Normalize: Rename sound files to new system rule f"{class_name}_{original_dataset}_{original_idx}"
    - Save filtered dataset info into a .csv file
    - Append to the main dataset info (.csv) file

+ For the whole (filtered, merged) dataset
    - Normalize: Convert soundfiles to PCM wav format (final merged processed)
    - Save final dataset state as .csv file
    - Augment data
    - Asign fold label for each data points
    - Save augmented, folded dataset state as .csv file



### New dataset implementation steps

0. Register the new dataset entry to `datasets.json`
1. Create a new `.py` file which filename is the dataset `key` name registered to `datasets.json`
2. Create the dataset class, extend `DataSet` from `ds.dataset.py`
3. Implement methods except `hellyeah()`
4. Add the `new class instance` to `datasets_registry` in `workflow.py`
5. Run `workflow.py` script
6. Manually map class of original dataset to system default class name in `config.json.classmapping.default`
7. Re-run `workflow.py` script :D