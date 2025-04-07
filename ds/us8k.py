import kagglehub
import os
import pandas as pd
import constants as C
from dataset import DataSet
from utils.json_utils import get_post_class_mapping
from utils.wav_utils import get_wave_data_length_2
from utils.csv_utils import write_csv_meta
from logging_cfg import get_logger
l = get_logger(__name__)

class UrbanSound8K(DataSet):
    def __init__(self):
        super().__init__("us8k")

    def init_class_names(self):
        """Initialize class names from metadata file."""
        df = pd.read_csv(self.get_paths().get_meta_path())
        self.class_names = df["class"].unique()

    def filter_by_class(self):
        """Filter dataset based on predefined class mappings."""
        df = pd.read_csv(self.get_paths().get_meta_path())
        class_map = get_post_class_mapping(self.key)
        df = df[df["class"].isin(class_map.keys())]

        from dataset import PD_SCHEMA
        filtered_df = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)
        filtered_df[C.DF_NAME_COL] = df["slice_file_name"]
        filtered_df[C.DF_PATH_COL] = df.apply(
            lambda row: os.path.abspath(
                os.path.join(self.get_paths().get_dir(), f"fold{row['fold']}", row["slice_file_name"])
            ),
            axis=1
        )
        filtered_df[C.DF_CLASS_ID_COL] = df["class"].apply(lambda original_classname: class_map[original_classname][C.CLASS_ID])
        filtered_df[C.DF_CLASS_NAME_COL] = df["class"].apply(lambda original_classname: class_map[original_classname][C.CLASS_NAME])
        filtered_df[C.DF_SUB_DS_NAME_COL] = self.key
        filtered_df[C.DF_SUB_DS_ID_COL] = df.index
        # filtered_df = filtered_df.apply(
        #     lambda x: get_wave_data_length_2(x),
        #     axis=1
        # )
        return filtered_df

    def get_filtered_meta_path(self):
        return self.filtered_meta_path


# def main():
#     ds = UrbanSound8K()
#     ds.hell_yeah()
#     print(ds.df.head(10))
#     no_length_row = ds.df[ds.df[C.DF_LENGTH_COL] == -1]
#     print(f"Could not get .wav length for {len(no_length_row)} row(s)")
#     print(no_length_row)



# if __name__ == "__main__":
#     main()
