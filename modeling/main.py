import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from utils.constants import FEATURE_KEY, TARGET_KEY

# Initialize models
models = {
    "RF": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "LR": LogisticRegression(max_iter=1000, random_state=42)
}


def get_model_score(df, df_json, model):
    features = df_json[FEATURE_KEY]
    target = df_json[TARGET_KEY]

    model = model.upper()
    if model not in models:
        raise ValueError

    model = models[model]
    X = df[features]
    y = df[target]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model using cross-validation
    scores = cross_val_score(model, X, y, cv=kf)
    for score in scores:
        print(f"cnvrg_linechart_accuracy value: {score}")

    model.fit(X, y)

    return model, np.mean(scores)
