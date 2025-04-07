from ds.dataset import DataSet
from utils.csv_utils import write_csv_meta
from utils.json_utils import get_post_class_mapping
from utils.wav_utils import get_wave_data_length_2
from platformdirs import user_cache_dir
import constants as C
import pandas as pd
import os
import zipfile
import requests
import re

class BDLib2(DataSet):
    """BDLib2 Dataset"""

    def __init__(self):
        # Determine the dataset path
        cache_dir = user_cache_dir()
        self.tmp_ds_abs_path = os.path.join(cache_dir, "datasets", "bdlib2")
        os.makedirs(self.tmp_ds_abs_path, exist_ok=True)
        super().__init__("bdlib2")

    def download(self):
        zip_url = "https://research.playcompass.com/files/BDLib-2.zip"
        zip_path = os.path.join(self.tmp_ds_abs_path, "BDLib2.zip")

        # Ensure the directory exists before downloading the file
        os.makedirs(self.tmp_ds_abs_path, exist_ok=True)

        # Download the ZIP file if it does not exist
        if not os.path.exists(zip_path):
            response = requests.get(zip_url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.tmp_ds_abs_path)

        from dataset import DsPaths
        extracted_path = self.tmp_ds_abs_path
        self.tmp_ds_abs_path = None
        return DsPaths(extracted_path, "", "")

    def init_class_names(self):
        """Create a class list from the directory"""
        columns = ["idx", "filename", "filepath", "category"]
        original_ds_df = pd.DataFrame(columns=columns)
        index = 0

        valid_classes = set(get_post_class_mapping(self.key).keys())
        for fold in ["fold-1", "fold-2", "fold-3"]:
            sub_f_path = os.path.join(self.get_paths().get_dir(), fold)
            if not os.path.exists(sub_f_path):
                continue 

            for f in os.listdir(sub_f_path):
                if f.endswith(".wav"):
                    file_name = f
                    file_path = os.path.join(fold, f)

                    # Find the class name in the filename
                    match = re.match(r"([a-zA-Z_]+)\d+\.wav", file_name)
                    if match:
                        category = match.group(1)  # Extract class name from the filename
                    else:
                        category = "unknown"

                    # Check if the category is valid
                    if category not in valid_classes:
                        continue  # Skip if it's not in config.json

                    original_ds_df.loc[len(original_ds_df)] = [index, file_name, file_path, category]
                    index += 1

        path = write_csv_meta(original_ds_df, self.key + ".original")
        self.ds_paths.set_meta_path(path)
        self.original_ds_df = original_ds_df

    def filter_by_class(self):
        """Filter data by class mapping"""
        df = self.original_ds_df
        class_map = get_post_class_mapping(self.key)
        df = df[df["category"].isin(class_map)]
        from ds.dataset import PD_SCHEMA
        df_filtered = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)

        df_filtered[C.DF_NAME_COL] = df["filename"]
        df_filtered[C.DF_PATH_COL] = df["filepath"].apply(
            lambda fp: os.path.join(self.ds_paths.get_dir(), fp))
        df_filtered[C.DF_CLASS_ID_COL] = df["category"].apply(
            lambda original_classname: class_map[original_classname][C.CLASS_ID])
        df_filtered[C.DF_CLASS_NAME_COL] = df["category"].apply(
            lambda original_classname: class_map[original_classname][C.CLASS_NAME])
        df_filtered[C.DF_SUB_DS_NAME_COL] = self.name
        df_filtered[C.DF_SUB_DS_ID_COL] = df.index
        
        # init length
        # df_filtered = df_filtered.apply(
        #     lambda x: get_wave_data_length_2(x),
        #     axis=1
        # )
        return df_filtered
    
def main():
    ds = BDLib2()
    ds.hell_yeah()
    print(ds.df.head(10))


if __name__ == "__main__":
    main()
