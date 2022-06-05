import os.path

import pandas as pd

from utils import get_class_infors, modify_infors, get_all_infors
from utils import processed_path

if __name__ == '__main__':
    # hotel, scenic, travel, food, news = get_class_infors()
    # hotel, scenic, travel, food, news = modify_infors(hotel, scenic, travel, food, news)
    # all_df = get_all_infors(hotel, scenic, travel, food, news)
    # all_df.to_csv(os.path.join(processed_path, "all.csv"), index=0)
    all_df = pd.read_csv(os.path.join(processed_path, "all.csv"))
    print(all_df.head())
