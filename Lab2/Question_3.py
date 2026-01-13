import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------  Load data -------------------------
X,y = fetch_california_housing(return_X_y=True)
print("------ Initial Shape of the data which is provided ------")
print(X.shape,y.shape)
print()

# ----------  Divide data -----------------------
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.30,random_state=1000)
print("------ After splitting the data shape of the x_train, x_test: ------")
print(x_train.shape,x_test.shape)
print("------ After splitting the data shape of the y_train, y_test: ------")
print(y_train.shape,y_test.shape)
print()

# ----------  Standardise data ------------------
aggregate = StandardScaler()
print(aggregate.fit(x_train))
print(" ------ Standardise data ------ ")
print(aggregate.mean_)
print()
x_trained_scaled = aggregate.transform(x_train)
x_test_scaled = aggregate.transform(x_test)


# ----------  Initializing the model ------------
model = LinearRegression()
model.fit(x_trained_scaled,y_train)


# ---------- Model Training ---------------------
y_pred = model.predict(x_test_scaled)
average_squared_error = mean_squared_error(y_test, y_pred)
root_mean_squared_error = np.sqrt(average_squared_error)
r2 = r2_score(y_test, y_pred)
print()
print("RMSE:", root_mean_squared_error)
print()
print("R2 Score:", r2)

# ---------- Done !!! ----------------------------