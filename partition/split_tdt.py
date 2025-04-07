"""
Steps:

1. get ratio from config.json
2. read the full meta csv
3. determine the median of datapoints of all classes
4. for each classname
    (update dataframe (2) in place)
    if number of classname datapoints < median
        sample up
    else
        sample down
5. write the augmented dataframe to the meta directory
6. Spliting (K-Fold Type shit)
    
    Total number of datapoints (after augmented) = n
    Number of folds: k
    Number of classnames: c
    Points per classname/ fold: p = n/k/c
    
    Train: k * 0.8 (folds)
    Dev: k * 0.1 (folds)
    Test: k * 0.1 (folds)
"""

from utils.json_utils import get_config_json
from utils.csv_utils import read_csv_as_dataframe
from utils.file_utils import get_filename_without_extension
from constants import MERGED_META_CSV
from typing import Final
import pandas as pd
import constants as C
from logging_cfg import get_logger
l = get_logger(__name__)

class SplitCfg:
    """_summary_
    Class storing partition config (train, dev, test, folds) for dataset
    """
    
    FOLDS: int = 0

    def __init__(self, tr: float, de: float, ts: float, folds: int = 0):
        
        sum = tr + de + ts
        # train + dev + test should be = 1.0 in any case
        self.train = tr / sum
        self.dev = de / sum
        self.test = ts / sum
        
        if (folds <= 3):
            raise ValueError("SplitCfg.Folds must be greater than 3 when init")
        self.FOLDS: Final[int] = folds 
        
        # tr_f + de_f + ts_f should be = FOLDS in any valid case
        
        ff = float(self.FOLDS)
        
        self.train_f = int(ff * self.train)
        self.dev_f = int(ff * self.dev)
        self.test_f = self.FOLDS - self.train_f - self.dev_f
    
    def get_train(self):
        return self.train

    def get_dev(self):
        return self.dev
    
    def get_test(self):
        return self.test
    
    def throw_f_not_initialized(self):
        """
        Throws ValueError if FOLDS is not initialized
        """
        if self.FOLDS == 0:
            raise ValueError("SplitCfg.Folds not initialized")
        pass
    
    def get_folds(self):
        self.throw_f_not_initialized()
        return self.FOLDS
    
    def get_train_fold(self):
        self.throw_f_not_initialized()
        return self.train_f
    
    def get_dev_fold(self):
        self.throw_f_not_initialized()
        return self.dev_f
    
    def get_test_fold(self):
        self.throw_f_not_initialized()
        return self.test_f
    
    def __str__(self):
        return f"Train: {self.train}, Dev: {self.dev}, Test: {self.test}, Folds: {self.FOLDS}"
    
def init_cfg() -> SplitCfg:
    """
    Initialize SplitCfg from config.json
    """
    l.info("Initializing SplitCfg from config.json")
    cfg = None
    try:
        cfg = get_config_json("partition")
        tr = float(cfg["train"])
        de = float(cfg["dev"])
        ts = float(cfg["test"])
        if "folds" in cfg:
            folds = int(cfg["folds"])
            cfg = SplitCfg(tr, de, ts, folds)
        else:
            cfg = SplitCfg(tr, de, ts)
        return cfg
    except Exception as e:
        l.error(f"Error initializing SplitCfg from config.json: {e}")
        raise e




def get_median(df: pd.DataFrame, rtype: type) -> int:
    """_summary_
    Get the median of classnames datapoints
    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    if rtype not in [int, float]:
        raise ValueError("rtype must be either int or float")
    median = df['class_name'].value_counts().median()
    if rtype == int:
        return int(median)
    else:
        return float(median)


def augment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the dataset df
    Args:
        df (pd.DataFrame): _description_
        median (int): _description_
    """
    augmented_df = pd.DataFrame()
    median = get_median(df, int)

    for _class_name, group in df.groupby(C.DF_CLASS_NAME_COL):
        if len(group) < median:
            # Allow duplicate sampling when group size is limited 
            sampled_group = group.sample(median, replace=True)
        else:
            sampled_group = group.sample(median, replace=False)
        
        augmented_df = pd.concat([augmented_df, sampled_group], axis=0)
    
    return augmented_df

def assign_fold(df: pd.DataFrame, cfg: SplitCfg) -> pd.DataFrame:
    """_summary_
        Assign fold number to each datapoint in the dataframe
    Args:
        df (pd.DataFrame): _description_
        cfg (SplitCfg): _description_

    Returns:
        _type_: _description_
        A copy of original dataframe with fold number assigned
    """
    aug_k_fold_df = df.copy()
    fold_number = cfg.get_folds()
    df[C.DF_FOLD_COL] = -1
    # Asume that the dataframe is already augmented, each class has the same number of datapoints
    for _class_name, group in df.groupby(C.DF_CLASS_NAME_COL):
        # Assign fold 0 -> fold_number foreach points in classname
        class_f_len = len(group) // fold_number
        fi = 0
        for i in range(fold_number):
            fold_start = i * class_f_len
            fold_end = fold_start + class_f_len
            for j in range(fold_start, fold_end):
                aug_k_fold_df.loc[group.index[j], C.DF_FOLD_COL] = int(fi)
            fi += 1
            
    return aug_k_fold_df

def split_tdt(df: pd.DataFrame, cfg: SplitCfg) -> pd.DataFrame:
    """
    6. Spliting (K-Fold Type shit)
    
    Total number of datapoints (after augmented) = n
    Number of folds: k
    Number of classnames: c
    Points per classname/ fold: p = n/k/c
    
    Train: k * 0.8 (folds)
    Dev: k * 0.1 (folds)
    Test: k * 0.1 (folds)
    """
    
    # Get ratio
    # r_tr = cfg.get_train()
    # r_de = cfg.get_dev()
    # r_ts = cfg.get_test()
    
    # # Get folds 
    # f_num = cfg.get_folds()
    
    # Augment the dataframe
    aug_df = augment_df(df)
    
    # Kfold split
    aug_k_df = assign_fold(aug_df, cfg)
    
    return aug_k_df



# def main(): 
#     # Read meta
#     df = read_csv_as_dataframe(FULL_META_CSV)
    
#     # Get split config
#     cfg = init_cfg()
    
#     l.info(f"Spliting with cfg {cfg.__str__()}")
    
#     # Split
#     aug_k_df = split_tdt(df, cfg)
    
#     # Save augmented dataframe
#     org_filename = get_filename_without_extension(FULL_META_CSV)
#     aug_filename = f"{org_filename}.augmented.folded.csv"
#     from constants import META_PATH
#     aug_filepath = os.path.join(META_PATH, aug_filename)
#     aug_k_df.to_csv(aug_filepath, index=False)        
    
    

# if __name__ == "__main__":
#     main()