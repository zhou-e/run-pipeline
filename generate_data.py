# import numpy as np
# import pandas as pd
# from scipy.special import expit  # sigmoid function
#
# # Set random seed for reproducibility
# np.random.seed(42)
#
# # Parameters
# n_samples = 1000
# n_features = 20
# n_informative = 5
#
# # Generate informative features (normally distributed)
# X_informative = np.random.randn(n_samples, n_informative)
#
# # Assign true coefficients to informative features
# true_coefs = np.random.uniform(-3, 3, size=n_informative)
#
# # Generate linear combination + noise
# linear_combination = X_informative @ true_coefs + np.random.normal(0, 1, size=n_samples)
#
# # Apply sigmoid to get probabilities
# probabilities = expit(linear_combination)
#
# # Threshold to get binary target
# y = (probabilities > 0.5).astype(int)
#
# # Generate noise features
# X_noise = np.random.randn(n_samples, n_features - n_informative)
#
# # Combine all features
# X = np.hstack((X_informative, X_noise))
#
# # Optionally shuffle the columns so you can't tell which are informative
# np.random.shuffle(X.T)
#
# # Create a DataFrame
# columns = [f'feature_{i+1}' for i in range(n_features)]
# df = pd.DataFrame(X, columns=columns)
# df['target'] = y
#
# df.to_parquet("testing.parquet")

import json

temp = {"features": [f"feature_{i+1}" for i in range(20)], "target": "target"}
with open("testing.json", "w") as file:
    json.dump(temp, file)
