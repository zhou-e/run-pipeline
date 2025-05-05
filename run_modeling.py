import argparse
import json
import pickle as pkl

import pandas as pd

from modeling.main import get_model_score

parser = argparse.ArgumentParser(description="Screening")
parser.add_argument("--data", action="store", dest="data", required=True)
parser.add_argument("--data_json", action="store", dest="data_json", required=True)
parser.add_argument("--model", action="store", dest="model", required=True)
args = parser.parse_args()

data = args.data
data_json = args.data_json
model = args.model

df = pd.read_parquet(data)
with open(data_json) as file:
    df_json = json.load(file)

model, score = get_model_score(df, df_json, model)
print(f"cnvrg_tag_accuracy: {score}")

with open("model.sav", "wb") as file:
    pkl.dump(model, file)
