import json
import pandas as pd
import constants as C
from ds.dataset import DataSet
from platformdirs import user_cache_dir
from utils.csv_utils import write_csv_meta
from utils.json_utils import append_empty_mapping_to_config
from file_utils import extract
from utils.wav_utils import get_wav_data_length
import os
import gdown
import zipfile
import shutil

class ThaoLp(DataSet):
    """
    Self-collected dataset 'thaoLp'.
    """

    def __init__(self):
        cache_dir = user_cache_dir()
        self.tmp_ds_abs_path = os.path.join(cache_dir, "dataset", "thaoLp")
        os.makedirs(self.ds_abs_path, exist_ok=True)
        super().__init__('thaoLp')

        # 🔹 Locate the config.json file
        datasets_json_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        if not os.path.exists(datasets_json_path):
            raise FileNotFoundError(f"Could not find config.json at {datasets_json_path}!")

        # 🔹 Read the content of config.json
        with open(datasets_json_path, "r") as f:
            datasets_config = json.load(f)

        # 🔹 Retrieve `class_mapping` from the entire config.json, not just `self.ds_config`
        global_class_mapping = datasets_config.get("class_mapping", {})
        self.class_mapping_raw = global_class_mapping.get("thaoLp", {})

        # 🔹 Debug: Check if class_mapping exists
        if not self.class_mapping_raw:
            print("No class mapping found for thaoLp in config.json!")

        # 🔹 Append empty mapping to config.json
        self.hell_yeah()

    def download(self):
        """Downloads the dataset from Google Drive and extracts it."""
        zip_path = os.path.join(self.ds_abs_path, "thaoLp.zip")

        if not os.path.exists(zip_path):
            gdown.download(self.ds_config.get("url", ""), zip_path, quiet=False)

        if not zipfile.is_zipfile(zip_path):
            print("ERROR: Downloaded file is not a valid ZIP file!")
            return  

        extract(zip_path, self.ds_abs_path)

    def find_sound_directory(self):
        """Finds the 'Sound/' directory in the dataset regardless of its location."""
        for root, dirs, _ in os.walk(self.ds_abs_path):
            if "Sound" in dirs:
                return os.path.join(root, "Sound")
        print(f"'Sound/' directory not found in {self.ds_abs_path}!")
        return None

    def normalize_class_name(self, name):
        """Standardizes class names to correctly map with config.json."""
        return name.strip().lower().replace("_", " ")  # Replace "_" with " " to match config.json

    def init_class_names(self):
        """Initializes class names based on folder structure and creates a metadata file."""
        sound_dir = self.find_sound_directory()
        if not sound_dir:
            return

        folders = [f.name for f in os.scandir(sound_dir) if f.is_dir()]
        self.class_names = folders if folders else []

        # 🔹 Normalize class_mapping
        class_mapping = {self.normalize_class_name(k): v for k, v in self.class_mapping_raw.items()}

        if not class_mapping:
            print("No class mapping found for thaoLp in config.json! Using default IDs.")
            class_mapping = {self.normalize_class_name(cls): idx for idx, cls in enumerate(self.class_names)}

        # 🔹 Create metadata CSV
        columns = [
            C.DF_NAME_COL, C.DF_PATH_COL, C.DF_LENGTH_COL, C.DF_CLASS_ID_COL, C.DF_CLASS_NAME_COL,
            C.DF_SUB_DS_NAME_COL, C.DF_SUB_DS_ID_COL
        ]
        data = []

        sub_ds_index = 0  # Initialize sub_ds_index

        for category in folders:
            norm_category = self.normalize_class_name(category)
            category_path = os.path.join(sound_dir, category)
            class_id = class_mapping.get(norm_category, -1)

            if class_id == -1:
                print(f"No class_id found for '{category}', defaulting to -1.")

            for file in os.listdir(category_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(category_path, file)
                    try:
                        length = get_wav_data_length(file_path)
                    except Exception:
                        length = -1
                    data.append([file, file_path, length, class_id, category, "ThaoLp", sub_ds_index])
                    sub_ds_index += 1  # Increment index for each file

        if data:
            self.df = pd.DataFrame(data, columns=columns)
        else:
            self.df = pd.DataFrame(columns=columns)
            print("No valid audio files found, df is empty.")

        self.filtered_meta_path = write_csv_meta(self.df, self.key + ".filtered")

    def filter_by_class(self):
        """This dataset does not require class filtering, just returns the original dataframe."""
        return self.df

    def get_filtered_meta_path(self):
        return self.filtered_meta_path

    def move_files(self):
        pass

    def hell_yeah(self, ditch_cache: bool = False):
        self.download()
        self.init_class_names()

        if self.df.empty:
            print("⚠️ WARNING: Dataset dataframe (df) is empty. Skipping processing steps.")
            return  

        if ditch_cache:
            shutil.rmtree(self.ds_paths.get_dir())