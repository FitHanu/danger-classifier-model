import os
import pandas as pd
import constants as C
from dataset import DataSet
from utils.json_utils import get_post_class_mapping
from utils.wav_utils import get_wave_data_length_2
from utils.csv_utils import write_csv_meta



class ESC50(DataSet):
    def __init__(self):
        super().__init__("esc50")

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
    # ds = ESC50()
    # ds.hell_yeah()
    # df = pd.read_csv(ds.get_paths().get_meta_path())
    # print(ds.df) # Columns: [id, file_name, file_path, length, class_id, class_name, sub_ds_name, sub_ds_id]
    # print(df.columns) # Index(['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take'], dtype='object')

    # Set the directory path
    directory = C.FILTERED_DATASET_PATH
    print(len(os.listdir(directory)))
    df = pd.read_csv(os.path.join(C.META_PATH, "esc50.filtered.csv"))
    print(df.shape)



if __name__ == "__main__":
    main()