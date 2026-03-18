import pandas as pd
import os
from typing import Tuple
from tqdm import tqdm

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads datasets from a directory with a progress bar.
    """
    files = ["bureau.csv", "application_train.csv", "application_test.csv"]
    dataframes = []

    # Using tqdm to iterate through the files with a visual bar
    pbar = tqdm(files, desc="Loading Datasets")
    for file in pbar:
        pbar.set_description(f"Reading {file}")
        path = os.path.join(data_dir, file)
        dataframes.append(pd.read_csv(path))
    
    return tuple(dataframes)

def merge_left(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Performs a left merge between two dataframes.
    Note: Standard pandas merges are usually too fast for a progress bar 
    unless the datasets are massive (millions of rows).
    """
    print(f"Merging data on column: {on}...")
    return df1.merge(df2, on=on, how="left")