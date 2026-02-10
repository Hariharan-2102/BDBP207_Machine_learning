from pandas.core.common import random_state


def load_data():
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns='MedHouseVal')
    y = data.frame['MedHouseVal']
    print(X.shape, y.shape)
    return X, y

def split_data(X, y, test, randomness):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state = randomness)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled_train= scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    return X_scaled_train, X_scaled_test


def hypothesis(X_scaled_train, theta):
    predicted_y = []
    prediction = 0
    for i in range(X_scaled_train.shape[0]):

        for j in range(X_scaled_train.shape[1]):
            prediction += theta[j] * X_scaled_train[i][j]
        predicted_y.append(prediction)
    return predicted_y

def derivative(X_scaled_train,y_train,theta):
    gradient = [0] * X_scaled_train.shape[1]
    for i in range(X_scaled_train.shape[0]):
        y_hat = hypothesis(X_scaled_train,theta)
        for j in range(X_scaled_train.shape[1]):
            gradient[j] += (y_hat[i] - y_train[i]) * X_scaled_train[i][j]
    return gradient

def theta_update(gradients, theta, alpha):
    for j in range(len(theta)):
        theta[j] -= alpha * gradients[j]
    return theta
# def gradient_descent(X_scaled_train, y_train, alpha=0.0001, epsilon=0.001):
#     change = float('inf')
#     theta = [0] * X_scaled_train.shape[1]
#
#     while change > epsilon:
#         gradients = derivative(X_scaled_train, y_train, theta)
#         old_theta = theta.copy()
#
#         for j in range(len(theta)):
#             theta[j] -= alpha * gradients[j]
#
#         change = max(abs(theta[j] - old_theta[j]) for j in range(len(theta)))
#
#     return theta

def gradient_descent(X, y, alpha=0.0001, epsilon=0.001, max_iter=100):
    theta = [0] * X.shape[1]
    change = float('inf')
    iteration = 0

    while change > epsilon and iteration < max_iter:
        gradients = derivative(X, y, theta)
        old_theta = theta.copy()

        for j in range(len(theta)):
            theta[j] -= alpha * gradients[j]
        change = max(abs(theta[j] - old_theta[j]) for j in range(len(theta)))

        if iteration % 5 == 0:
            print(f"iter {iteration}, change = {change}")
        iteration += 1

        # convergence check
        if change < epsilon:
            print("Converged.")
            break

        # safety stop
        if iteration >= max_iter:
            print("Reached max iterations.")
            break

        iteration += 1

    return theta



def r2_score(X_test, y_test, theta):
    from sklearn.metrics import r2_score
    y_pred = hypothesis(X_test, theta)
    score = r2_score(y_test, y_pred)
    return score


def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2, 56)
    y_train = y_train.values #pandas series treats [i] as label and not a position
    y_test = y_test.values #
    X_scaled_train, X_scaled_test = scale_data(X_train, X_test)
    theta = gradient_descent(X_scaled_train, y_train,0.0001, 0.001 )
    y_pred_test = hypothesis(X_scaled_test, theta)
    r2 = r2_score(y_test, y_pred_test)
    print("r2_score: ", r2)

if __name__ == "__main__":
    main()