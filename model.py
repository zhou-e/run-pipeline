import pickle as pkl

from cnvrg import Endpoint
import numpy as np
import pandas as pd
import sklearn

endpoint = Endpoint()

model = pkl.load(open("/input/winning_task/model.sav", "rb"))


def predict(*param):
    param = list(param)

    try:
        res = model.predict_proba([param])
        res = res.to_list()[0]
        endpoint.log_metric("confidence", res[0])
    except Exception as e:
        print(f"Error occurred when running prediction: {e}")
        raise e

    return res[0]
