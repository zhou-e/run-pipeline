import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from utils.constants import FEATURE_KEY, TARGET_KEY


def get_model_score(
    df,
    df_json,
    model,
    rf_trees,
    rf_depth,
    svc_kernel,
    lr_penalty,
):
    features = df_json[FEATURE_KEY]
    target = df_json[TARGET_KEY]

    if model.upper() == "RF":
        model = RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=rf_depth,
            random_state=42
        )
    elif model.upper() == "SVM":
        model = SVC(kernel=svc_kernel, probability=True, random_state=42)
    elif model.upper() == "LR":
        model = LogisticRegression(
            penalty=lr_penalty,
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Model {model} is not supported. Choose from RF, SVM, or LR.")

    print(f"cnvrg_tag_features: {features}")
    X = df[features]
    y = df[target]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate model using cross-validation
    scores = cross_val_score(model, X, y, cv=kf)
    for score in scores:
        print(f"cnvrg_linechart_accuracy value: {score}")

    model.fit(X, y)

    return model, np.mean(scores)
