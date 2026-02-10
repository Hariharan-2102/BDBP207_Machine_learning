import numpy as np
import pandas as pds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    # >>>>>>>>>>>>>>>>> Load data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    data_frame = pds.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = data_frame.drop(["disease_score",'disease_score_fluct'],axis=1)
    y = data_frame["disease_score_fluct"]
    # print(X)
    # print(y)

    # >>>>>>>>>>>>>>>>>> split data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=500)
    X_train = X_train.values
    X_test = X_test.values
    y_test = y_test.values
    y_train = y_train.values
    # print(X_train)
    # print(X_test)
    # print(y_test)
    # print(y_train)

    # >>>>>>>>>>>>>>>>>>> Standardize data <<<<<<<<<<<<<<<<<<<<<<<<<<<<

    aggregate = StandardScaler()
    aggregate.fit(X_train)

    X_train_scaled = aggregate.transform(X_train)
    X_test_scaled = aggregate.transform(X_test)

    X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_test_scaled  = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]



    # print(X_train)
    theta = np.zeros(X_train_scaled.shape[1])
    print(theta)



    def hypothesis_function(x_hypo,theta_hypo):
        return np.dot(x_hypo,theta_hypo)

    def gradient_function(x_value,y_value,theta_gradient):
        m = len(y_value)
        predictions = hypothesis_function(x_value,theta_gradient)
        hypo_for_gradient = predictions - y_value
        gradient = (1/m) * np.dot(x_value.T, hypo_for_gradient)
        return gradient

    alpha = 0.01
    iteration = 1000
    loss_history = []

    for i in range(iteration):

        # compute gradient
        grad = gradient_function(X_train_scaled, y_train, theta)

        # update theta
        theta = theta - alpha * grad

        # compute loss (MSE)
        predictions = hypothesis_function(X_train_scaled, theta)
        loss = np.mean((predictions - y_train)**2)
        loss_history.append(loss)

        if i % 50 == 0:
            print("iteration:", i, "Loss:", loss)


    test_predictions = hypothesis_function(X_test_scaled, theta)

    r2 = r2_score(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    rmse = np.sqrt(mse)

    print("\nR2 Score:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("\nloss_values",loss_history)

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()


    # print(hypothesis_function(X_train_scaled,theta))
    # print(gradient_function(X_train_scaled,y_train,theta))

if __name__ == "__main__":
    main()