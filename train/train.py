"""
make a function
Params: pd: pd.DataFrame, SplitCfg: SplitCfg
Steps:

1. Load original df from full.augmented.folded.csv
2. Move files & update path (optional)
    ....
3. Lambda -> format/ resample datapoints
4. Lambda -> Use Yamnet to extract feature embeddings (1, 512)
5. 
"""

from utils.wav_utils import load_wav_16k_mono
import pandas as pd

def main():
    from constants import MERGED_META_CSV
    main_df = pd.read_csv

if __name__ == "__main__":
    main()
