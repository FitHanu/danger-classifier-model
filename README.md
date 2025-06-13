# Hello world

+ This project provide scripts to handle audio data processing & training tensorflow `Danger Classification Model` using `YAMnet` audio embedding extraction layer (transfer learning)


## Enviroment (Conda based)

+ Python Version: 
    + `dev`: 3.12.9 (github_codespace)
    + `training`: 3.11.11 (google_colab)
+ Target OS: Linux (No support for windows)
+ Operate: Ubuntu 22.04.4 LTS x86_64 (Google colab Q1 2025)
+ Develop: Debian GNU/Linux 12 x86_64 (See `/.devcontainer/devcontainer.json`)
+ CUDA Version: 12.*

## Dependencies
+ See `/environment.yml`  for conda
+ See `/requirements.txt` for pip

## Setup

### Using conda

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
5. Training
```
python workflow.py
```

### Using pip

1. run setup script
```
python setup.py
```

2. run setup script
```
python workflow.py
```

# Insights 

### `/workflow.py` procedure 

+ For each dataset:
    - Download dataset
    - Filter sound file based on defined sound classes at `/classes.csv`
    - Move filtered sound files into main dataset path `/dataset/`
    - Normalize:
        - Rename filtered sound files in `/dataset/` to system rule `f"{class_name}_{original_dataset}_{original_idx}"`
    - Save filtered dataset meta into a .csv file in `/ds/meta`
    - Append filtered dataset meta to the main dataset meta in `/ds/meta`

+ After pre-processing each dataset
    - Augmentation
        - Calculate `mean` datapoint `count` for each label in `classes.csv`
        - For each `label` in `classes.csv`
            - if `count` < `mean`
                - duplicate random data in `label` to match with `mean`
            - if `count` > `mean`
                - remove random data in `label` to match with `mean`
    - Normalize:
        - For each `.wav` file in `/dataset/`
            - Convert to PCM 16 wav format
            - Convert bit depth from any to `16`
    - Asign fold label for each data points
    - Save augmented, folded dataset state as .csv file

+ Training: See details at `workflow.py`:`train()`

### How to add & implement new dataset for training?

0. Register the new dataset info object to `datasets.json` (based on previous format)
1. Human class mapping for added dataset to system defined labels id in `config.json`:`classmapping.default`
2. Create a new dataset `.py` file inside `/ds/` which filename is identical to the dataset `key` field value at registered object in step `0`
3. Create the dataset class, extending `DataSet` from `ds.dataset.py`
4. Implement necessary methods
5. Add the dataset class contructor to `datasets_registry` in `workflow.py`
6. Run `workflow.py --process_data_only` script
7. Run `workflow.py --use_processed` script :D