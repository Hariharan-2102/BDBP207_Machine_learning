import numpy as np
import pandas as pds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # ----------  Load data -------------------------
    data_frame = pds.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    print("------ Initial Shape of the data which is provided ------")
    X = data_frame.drop(["disease_score",'disease_score_fluct'],axis=1)
    y = data_frame["disease_score_fluct"]
    print()
    print(data_frame.shape)
    print()
    print(X.shape,y.shape)
    print()
    print(data_frame.columns)

    # ----------  Divide data -----------------------
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=999)
    print("------ After splitting the data, shape of the x_train, x_test: ------")
    print(x_train.shape,x_test.shape)
    print()
    print("------ After splitting the data, shape of the y_train, y_test: ------")
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
    y_predict = model.predict(x_test_scaled)
    average_squared_error = mean_squared_error(y_test, y_predict)
    root_mean_squared_error = np.sqrt(average_squared_error)
    r2 = r2_score(y_test, y_predict)
    print()
    print("Root Mean Square Error :", root_mean_squared_error)
    print()
    print("R^2 Score:", r2)

    # ---------- Done !!! ----------------------------

if __name__ == "__main__":
    main()