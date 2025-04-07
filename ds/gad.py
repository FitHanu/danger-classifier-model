from ds.dataset import DataSet
from utils.csv_utils import write_csv_meta
from utils.json_utils import append_empty_mapping_to_config, get_post_class_mapping
from utils.wav_utils import get_wave_data_length_2
import constants as C
import kagglehub
import pandas as pd
import os

class GAD(DataSet):

    """
    Gunshot Audio Dataset
    """

    def __init__(self):
        super().__init__(key="gad")

    def download(self):
        from dataset import DsPaths
        ds_abs_path = kagglehub.dataset_download(self.kaggle_path)
        # No meta file
        meta_rel_path = ""
        data_rel_path = ""
        ds_paths = DsPaths(ds_abs_path, meta_rel_path, data_rel_path)
        return ds_paths
    
    def init_class_names(self):
        """_summary_
        This one doesn;t have meta file so we:
        1. load folder names as class names
        2. Make a meta file in `.csv`
        3. Assign created `.csv` filepath to `self.ds_paths.meta_path`
        """
        
        # Load folder name as class names
        folders = [f.name for f in os.scandir(self.get_paths().get_dir()) if f.is_dir()]
        df = pd.DataFrame(folders, columns=["category"])
        self.class_names = df["category"].unique()

        columns = ["idx", "filename", "filepath", "category"]        
        original_ds_df = pd.DataFrame(columns=columns)
        index = 0;
        for category in folders:
            sub_f_path = os.path.join(self.get_paths().get_dir(), category)
            sub_f_files = [f.name for f in os.scandir(sub_f_path) if f.is_file() and f.name.endswith('.wav')]
            for f in sub_f_files:
                file_name = f
                file_path = os.path.join(category, f)
                original_ds_df.loc[len(original_ds_df)] = [index, file_name, file_path, category]
                index += 1
                
        # Init original dataset .csv meta file
        path = write_csv_meta(original_ds_df, self.key + ".original")
        self.ds_paths.set_meta_path(path)
        

    def filter_by_class(self):
        """_summary_
        """
        # # Load and filter the original meta file
        df = pd.read_csv(self.get_paths().get_meta_path())
        # Get allowed classes and their mapping
        class_map = get_post_class_mapping(self.key)
        # Eliminate classes not in class_map
        df = df[df["category"].isin(class_map.keys())]
        
        from dataset import PD_SCHEMA
        df_filtered = pd.DataFrame(columns=PD_SCHEMA.keys()).astype(PD_SCHEMA)

        # Map filtered dataframe to self.df with the right schema
        df_filtered[C.DF_NAME_COL] = df["filename"]
        df_filtered[C.DF_PATH_COL] = df["filepath"].apply(lambda fp: os.path.join(self.get_paths().get_dir(), fp))
        df_filtered[C.DF_CLASS_ID_COL] = df["category"].apply(lambda original_classname: class_map[original_classname][C.CLASS_ID])
        df_filtered[C.DF_CLASS_NAME_COL] = df["category"].apply(lambda original_classname: class_map[original_classname][C.CLASS_NAME])
        df_filtered[C.DF_SUB_DS_NAME_COL] = self.name
        df_filtered[C.DF_SUB_DS_ID_COL] = df.index
        
        # Get length
        # df_filtered = df_filtered.apply(get_wave_data_length_2, axis=1)
        
        return df_filtered
        
    def create_meta(self):
        path = write_csv_meta(self.df, self.key + ".filtered")
        return path

def main():
    ds = GAD()
    ds.hell_yeah()
    print(ds.df.head(10))

if __name__ == "__main__":
    main()