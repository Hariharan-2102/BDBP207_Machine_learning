 # Implementation of normal equation from scratch
 # Compare results on simulated datasets
 # (disease score fluctuation as target) and the admissions dataset

import numpy as np
import pandas as pds
data_set = pds.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
# print(data_set.columns)
# ------------------ load_data ---------------------------------
x_value = data_set.drop(["disease_score_fluct"],axis=1)
y_value = data_set["disease_score_fluct"]

x_value = x_value.values
y_value = y_value.values

print(x_value.shape)
print(y_value.shape)

# ------------------ normal function ---------------------------

theta = np.dot(np.linalg.inv(np.dot(np.transpose(x_value),x_value)),np.dot(np.transpose(x_value),y_value))
print(theta)
