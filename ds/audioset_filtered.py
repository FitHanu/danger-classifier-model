"""
EXPERIMENTAL


Not used in dataprocessing and training
"""

import os
import gdown
import pandas as pd
from platformdirs import user_cache_dir
import constants as C
from dataset import DataSet, DsPaths
from utils.file_utils import extract
from utils.json_utils import get_post_class_mapping
from utils.wav_utils import get_wave_data_length_2
from utils.csv_utils import write_csv_meta



class AuSetFld(DataSet):
    """AudioSet Filtered Dataset"""
    def __init__(self):
        self.tmp_ds_abs_path = os.path.join(user_cache_dir(), "datasets", "asf")
        os.makedirs(self.tmp_ds_abs_path, exist_ok=True)
        super().__init__("asf")
    
    def download(self):
        
        zip_path = os.path.join(self.tmp_ds_abs_path, "asf.zip")
        gdown.download(self.url, zip_path, quiet=False)
        extracted_path = os.path.join(self.tmp_ds_abs_path, "extracted")
        extract(zip_path, extracted_path)
        meta_path = os.path.join(extracted_path, "metadata.csv")
        audio_path = os.path.join(extracted_path, "audio")
        
        return DsPaths(extracted_path, meta_path, audio_path)

    def init_class_names(self):
        df = pd.read_csv(self.get_paths().get_meta_path())
        self.class_names = df["category"].unique()

    def filter_by_class(self) -> pd.DataFrame:
        meta_path = self.get_paths().get_meta_path()
        audio_path = self.get_paths().get_data_path()
        # Load and filter the original meta file
        df = pd.read_csv(meta_path)
        class_map = get_post_class_mapping(self.key)
        df_filtered = df[df["category"].isin(class_map.keys())]
        df = df_filtered
        
        from ds.dataset import PD_SCHEMA
        final_df = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)
        
        # Map filtered dataframe to self.df with the right schema
        # self.df[C.DF_ID_COL] = range(len(df))
        final_df[C.DF_NAME_COL] = df["filename"]
        final_df[C.DF_PATH_COL] = df["filename"].apply(lambda filename: os.path.join(audio_path, filename))
        final_df[C.DF_CLASS_ID_COL] = df["category"].apply(lambda original_classname: class_map[original_classname][C.CLASS_ID])
        final_df[C.DF_CLASS_NAME_COL] = df["category"].apply(lambda original_classname: class_map[original_classname][C.CLASS_NAME])
        final_df[C.DF_SUB_DS_NAME_COL] = self.name
        final_df[C.DF_SUB_DS_ID_COL] = df.index
        
        
        # Get length
        # df_filtered = df_filtered.apply(get_wave_data_length_2, axis=1)
        
        return final_df

    def create_meta(self):
        df = pd.read_csv(self.get_paths().get_meta_path())
        path = write_csv_meta(self.df, self.key + ".filtered")
        _ = write_csv_meta(df, self.key + ".original")
        return path


def main():
    ds = AuSetFld()
    ds.download()



if __name__ == "__main__":
    main()