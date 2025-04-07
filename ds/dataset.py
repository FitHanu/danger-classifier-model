import abc
import os
import pandas as pd
import constants as C
from pathlib import Path
from typing import final, Optional, List
from dataclasses import dataclass
from utils.json_utils import get_config_json
import shutil
import kagglehub
from logging_cfg import get_logger
l = get_logger(__name__)

"""
Default Schema for each dataset meta file/ pandas DataFrame
"""
PD_SCHEMA = {
    # Unique ID for each audio file
    # C.DF_ID_COL: "int64",
    # File name
    C.DF_NAME_COL: "string",
    # Audio file path
    C.DF_PATH_COL: "string",
    # Audio file length (must be in milliseconds)
    # C.DF_LENGTH_COL: "int64",
    # Class Name ID
    C.DF_CLASS_ID_COL: "int16",
    # Class Name (string)
    C.DF_CLASS_NAME_COL: "string",
    # Original dataset name
    C.DF_SUB_DS_NAME_COL: "string",
    # ID in the original dataset
    C.DF_SUB_DS_ID_COL: "string", 
}

@dataclass
class DsMeta:
    key: str
    name: str
    format: str
    kaggle_path: Optional[str] = None
    url: Optional[str] = None
    csv_meta_path: Optional[List[str]] = None
    data_path: Optional[List[str]] = None
    
    @staticmethod
    def load_from_json(dataset_key: str) -> Optional["DsMeta"]:
        
        """
        Loads a dataset metadata from `datasets.json` on ds key.
        """
        
        from utils.json_utils import get_dataset_info
        raw_meta = get_dataset_info(dataset_key)
        return DsMeta(**raw_meta)  # Unpack dictionary into DsMeta fields
    
    def has_csv_meta(self) -> bool:
        return self.csv_meta_path is not None and self.csv_meta_path != []


# dataset = DsMeta(
#     name="ESC-50",
#     format="kaggle",
#     kaggle_path="mmoreaux/environmental-sound-classification-50",
#     url="https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50",
#     csv_meta_path=["esc50.csv"],
#     data_path=["audio", "audio"]
# )

class DsPaths:
    """
    For storing dataset paths\n
    Return type for `DataSet.instance.download()` function\n
    
    `dir`: dataset absolute path\n
    `meta_path`: ds absolute meta path\n
    `data_path`: ds absolute data path
    """

    def __init__(self, abs:str, meta:str, data:str):
        self.dir = abs
        self.meta_path = meta
        self.data_path = data
    
    def get_dir(self):
        return self.dir
    
    def get_meta_path(self):
        return self.meta_path
    
    def get_data_path(self):
        return self.data_path
    
    def set_meta_path(self, meta_path):
        self.meta_path = meta_path

class DataSet(abc.ABC):
    """
    Abstract class representing a dataset
    ** ABS: should not be instantiated **
    """
    
    # Default class names
    THIS_SCRIPT_PATH = Path(__file__).resolve().parent.parent
    DEFAULT_CLASSNAME_PATH = os.path.join(THIS_SCRIPT_PATH, "classes.csv")
    DEFAULT_CLASSNAMES_DF = pd.read_csv(DEFAULT_CLASSNAME_PATH)
    NEW_PATH_COL = "new_path"
    
    """
    "FILTERED_DATASET_PATH": Our full dataset (filtered) path
    
    After filter, each sub dataset must have:
    + A meta file in with "META_PATH"
    + Data inside "FILTERED_DATASET_PATH"
    """
    PROJECT_PATH = THIS_SCRIPT_PATH.parent
    FILTERED_DATASET_PATH = os.path.join(PROJECT_PATH, "dataset", "audio")
    META_PATH = os.path.join(PROJECT_PATH, "dataset", "meta")
    
    
    def __init__(self, key: str):

        from utils.json_utils import get_dataset_info

        info = get_dataset_info(key)
        self.json_meta = info
        self.key = info["key"]
        self.name = info["name"]
        self.format = info["format"]
        self.kaggle_path = info["kaggle_path"]
        self.url = info["url"]
        # self.df need to be reinitialized/ poured with data in the child class filter_by_class function
        self.df = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)
        
    def get_key(self):
        return self.key

    def get_filtered_meta_path(self):
        try:
            return self.filtered_meta_path
        except AttributeError:
            raise AttributeError("Call `hell_yeah()` first to get the filtered meta path")

    def __str__(self):
        return (f"DataSet(\n"
                f"  key={self.key},\n"
                f"  name={self.name},\n"
                f"  format={self.format},\n"
                f")")
        
    @abc.abstractmethod
    def init_class_names(self):
        """
        - Initialize the class names for the dataset
        Type of self.class_names should be List[str]
        """
        self.class_names = []
        pass

    def download(self) -> DsPaths:
        """
        Download the dataset if `self.format` is "kaggle"\n
        - Else format must be overriden, or it would raise this shit: `NotImplementedError`\n
        Returns a `dataset.DsPaths` object.
        """
        if self.format == "kaggle":
            
            ds_abs_path = kagglehub.dataset_download(self.kaggle_path)
            
            meta_rel_path = self.json_meta["csv_meta_path"]
            meta_rel_path = os.path.join(ds_abs_path, *meta_rel_path)
            
            data_rel_path = self.json_meta["data_path"]
            data_rel_path = os.path.join(ds_abs_path, *data_rel_path)
            
            ds_paths = DsPaths(ds_abs_path, meta_rel_path, data_rel_path)

            return ds_paths
        else:
            raise NotImplementedError("Format other than Kaggle needs to be implemented manually") 

    @abc.abstractmethod
    def filter_by_class(self) -> pd.DataFrame:
        """
        - Overide this method to perform following operations:
        + Filter the dataset by mapped class in the config.json
        + Should return a pd DataFrame contain audio file paths with their 
        corresponding system default class name
        """
        df = self.df.copy()
        return df
        
    
    # @abc.abstractmethod
    def normalize(self) -> pd.DataFrame:

        """
        #### Normalize the dataset file names\n
        - Call after `filter_by_class()`
        - Verify original file existance, throws `FileNotFoundError`\n
        - rename `C.DF_NAME_COL` with format: \n
        `f"{row[C.DF_CLASS_NAME_COL]}_{self.key}_{row[C.DF_SUB_DS_ID_COL]}.wav"`\n
        """

        df = self.df.copy()
        def create_new_name(row):
            """
            Rename to normalized format
            """
            if os.path.isfile(row[C.DF_PATH_COL]):
                new_name = f"{row[C.DF_CLASS_NAME_COL]}_{self.key}_{row[C.DF_SUB_DS_ID_COL]}.wav"
                row[C.DF_NAME_COL] = new_name
                return row
            else:
                raise FileNotFoundError(f"File not found: {row[C.DF_PATH_COL]}")
        
        df = df.apply(create_new_name, axis=1)
        return df
    
    # @abc.abstractmethod
    def create_meta(self) -> str:
        from utils.csv_utils import write_csv_meta
        df = pd.read_csv(self.get_paths().get_meta_path())
        path = write_csv_meta(self.df, self.key + ".filtered")
        _ = write_csv_meta(df, self.key + ".original")
        return path
    
    # @abc.abstractmethod
    def copy_files(self) -> pd.DataFrame:
        
        """
        #### Copy files of original dataset from `user_cache_dir` to `C.FINAL_DATASET_PATH`
        - Call after `normalize()`\n
        """
        
        target_path = C.FILTERED_DATASET_PATH
        os.makedirs(target_path, exist_ok=True)
        df = self.df.copy()
        
        # Create temp column to store new path
        df["new_path"] = df.apply(lambda row: os.path.join(target_path, row[C.DF_NAME_COL]), axis=1)
        df.apply(lambda row: shutil.copy(row[C.DF_PATH_COL], row["new_path"]), axis=1)
        df[C.DF_PATH_COL] = df["new_path"]
        df.drop(columns=["new_path"], inplace=True)

        return df
    
    @final
    def ensure_files(self) -> bool:
        """
        #### Ensure availability of all files of the dataset inside final path.
        - Call after `copy_files()`
        - Final method, should not be overriden to ensure files in dataset are available
        """
        exist_files = self.df[self.df[C.DF_PATH_COL].apply(lambda x: os.path.isfile(x))]
        missing_no = len(self.df) - len(exist_files)
        if(missing_no > 0):
            l.error(f"Missing {missing_no} files for ds: {self.key}")
            return False
        else:
            return True

    @final
    def get_paths(self) -> DsPaths:
        """
        #### Get dataset paths, type: `dataset.DsPaths`
        - Callable after `download()`
        """
        try:
            return self.ds_paths
        except AttributeError:
            raise AttributeError("Call `hell_yeah()` first to get the dataset paths")
    
    @final
    def hell_yeah(self, ditch_cache: bool = False):
        """
        Processing flows of datasets, `@final` should not change/ override this\n
        """
        # Download and init paths
        ds_paths = self.download()
        self.ds_paths = ds_paths
        
        # Init class names (Might not neccessary)
        self.init_class_names()
        
        # Filter by class map from config.json
        self.df = self.filter_by_class()
        
        # Normalize the dataset (Optional)
        self.df = self.normalize()
        
        # Move files to the final dataset path
        m_df = self.copy_files()
        self.df = m_df
        
        # Ensure all files are available
        if not self.ensure_files():
            raise FileNotFoundError(f"Missing data files for dataset: {self.key} after processing")
        
        # Create meta file
        filtered_meta_path = self.create_meta()
        self.filtered_meta_path = filtered_meta_path
        
        if ditch_cache:
            l.info(f"ditch_cache = True, removing {self.ds_paths.get_dir()}")
            shutil.rmtree(self.ds_paths.get_dir())



def main():
    ds_meta = DsMeta.load_from_json("us8k")
    print(ds_meta)
    print(ds_meta.has_csv_meta())

if __name__ == "__main__":
    main()