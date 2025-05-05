import argparse
import json

import pandas as pd

from screening.main import screen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screening")
    parser.add_argument("--data", action="store", dest="data", required=True)
    parser.add_argument("--data_json", action="store", dest="data_json", required=True)
    parser.add_argument("--threshold", action="store", dest="threshold", type=float, required=False)
    args = parser.parse_args()

    data = args.data
    data_json = args.data_json
    threshold = args.threshold

    # data = "testing.parquet"
    # data_json = "testing.json"

    df = pd.read_parquet(data)
    with open(data_json) as file:
        df_json = json.load(file)

    new_df, new_json = screen(df, df_json, threshold)

    # stored in directory name /input/<step name>/<filename>
    new_df.to_parquet("filtered_data.parquet")
    with open("filtered_data.json", "w") as file:
        json.dump(new_json, file)
