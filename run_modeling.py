import argparse
import json
import pickle as pkl

import pandas as pd

from constants import HYPERPARAMS
from modeling.main import get_model_score

parser = argparse.ArgumentParser(description="Screening")
parser.add_argument("--data", action="store", dest="data", required=True)
parser.add_argument("--data_json", action="store", dest="data_json", required=True)
parser.add_argument("--model", action="store", dest="model", required=True)
parser.add_argument("--param_rf_trees", action="store", type=int, dest="rf_trees", required=False)
parser.add_argument("--param_rf_depth", action="store", type=int, dest="rf_depth", required=False)
parser.add_argument("--param_svc_kernel", action="store", dest="svc_kernel", required=False)
parser.add_argument("--param_lr_penalty", action="store", dest="lr_penalty", required=False)
args = parser.parse_args()

data = args.data
data_json = args.data_json
model = args.model

df = pd.read_parquet(data)
with open(data_json) as file:
    df_json = json.load(file)

# parse hyperparameters
params = HYPERPARAMS.copy()
for key, value in vars(args):
    if value is not None and key.startswith("param_"):
        key = key.replace("param_", "")
        params[key] = value

model, score = get_model_score(df, df_json, model, **params)
print(f"cnvrg_tag_accuracy: {score}")

with open("model.sav", "wb") as file:
    pkl.dump(model, file)
