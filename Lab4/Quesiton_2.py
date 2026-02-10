import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

def main():
    # >>>>>>>>>>>>>>>>> Load data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    X, y = fetch_california_housing(return_X_y=True)
    # print(X)
    # print(y)

    # >>>>>>>>>>>>>>>>>> split data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=500)
    # X_train = X_train.values
    # X_test = X_test.values
    # y_test = y_test.values
    # y_train = y_train.values
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

def skdata():
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

print(skdata())