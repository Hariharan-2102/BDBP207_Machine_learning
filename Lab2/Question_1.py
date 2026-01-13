import numpy as np

X = np.array([
    [1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]
])

covariance = np.cov(X,rowvar=False)
print(covariance)