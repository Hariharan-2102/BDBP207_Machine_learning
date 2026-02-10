# ######################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Data
# Note: Ensure the CSV is in the same directory
data_frame = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = data_frame.drop(["disease_score", "disease_score_fluct"], axis=1)
y = data_frame["disease_score_fluct"]


# 2. Split Data
def train_test_split_data(x_value, y_value):
    return train_test_split(x_value, y_value, test_size=0.30, random_state=999)


x_train_raw, x_test_raw, y_train, y_test = train_test_split_data(X, y)

# 3. Scaling (Correct Implementation)
# You must fit on train and transform both train and test
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_raw)
x_test_scaled = scaler.transform(x_test_raw)

# 4. Add Bias Term (Intercept)
# Linear regression needs a column of 1s to calculate the intercept
x_train_scaled = np.c_[np.ones(x_train_scaled.shape[0]), x_train_scaled]
x_test_scaled = np.c_[np.ones(x_test_scaled.shape[0]), x_test_scaled]


# 5. Hypothesis Function
def hypothesis(X, theta):
    return np.dot(X, theta)


# 6. Gradient Descent Logic
def compute_gradient_descent(X, y, alpha=0.1, iterations=500):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    y_values = y.values  # Convert pandas Series to numpy array

    cost_history = []

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        errors = predictions - y_values

        # Gradient = (1/n) * X_transpose * errors
        gradient = (1 / n_samples) * np.dot(X.T, errors)

        # Update theta
        theta = theta - alpha * gradient

        # Optional: Track cost to check for convergence
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history


# 7. Execution
final_theta, cost_history = compute_gradient_descent(x_train_scaled, y_train)


# 8. Evaluation (R2 Score)
def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


predictions = hypothesis(x_test_scaled, final_theta)
print(f"\nFinal R2 Score: {r2_score(y_test, predictions)}")




























# import numpy as np
# import pandas as pds
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# from BDBP207_Machine_learning.Lab3.Question_1 import x_train
#
# # import matplotlib.pyplot as plt
# # >>>>>>>>>>>>>>>>> Load data <<<<<<<<<<<<<<<<<<<<<<<<<
#
# data_frame = pds.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
# # data_frame_shuffle = data_frame.sample(frac=1, random_state=999).reset_index(drop=True)
# # print(data_frame_shuffle)
# X = data_frame.drop(["disease_score",'disease_score_fluct'],axis=1)
# y = data_frame["disease_score_fluct"]
#
# def train_test_split_data (x_value,y_value):
#     x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=0.30, random_state=999)
#     return x_train, x_test, y_train, y_test
#
# # Standardization of X_values:
# x_std = StandardScaler(train_test_split_data(x_train))
#
#
# # print(X)
# # print(y)
# # print(data_frame.head(5))
# # print(data_frame_shuffle.head(5))
#
# # y_scale = y.to_numpy()
# # print(y_scale)
# # x_scale  = X.to_numpy()
# # print(x_scale)
# # print(X)
#
# # split = train_test_split_data(X,y)
# # print(split)
# # input_theta = np.zeros(X.shape[1])
#
# def hypothesis(x_train_hypo_value,input_theta):
#     x_train = np.array(x_train_hypo_value)
#     hypo_theta_value = np.array(input_theta)
#     hypothesis_value = np.dot(x_train, hypo_theta_value)
#     return hypothesis_value
#
# def compute_gradient(hypothesis_values):
#     alpha_gradient = 0.01
#     iteration = 1000
#     # -alpha_gradient(hypo_theta)
#
#
#
#
# hypo_theta = np.zeros(X.shape[1])
# split_data = train_test_split_data(X,y)
# input_theta = np.zeros(X.shape[1])
#
#
# x_train, x_test, y_train, y_test = train_test_split_data(X,y)
#
# hypothesis_results = hypothesis(x_train,hypo_theta)
# print(hypothesis_results)






























# hypo_value = hypothesis(x_scale,theta_input)
# cost_value = cost_function(hypo_value,y_scale)
# gradient = compute_gradient(hypo_value,x_scale,y_scale)

# print(hypo_value)
# print()
# print(cost_value)
# print()
# print(gradient)






# # print(X)
# X = X.values
# # print(X)
# theta_input = np.zeros(X.shape[1])
# h = hypothesis(X,theta_input)
# # print(h)

# import numpy as np
# import pandas as pd
#
# # ---------------- Load data ----------------
# data_frame = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#
# X = data_frame.drop(["disease_score", "disease_score_fluct"], axis=1)
# y = data_frame["disease_score_fluct"]
#
# # ---------------- Scale X and y ----------------
# X = (X - X.mean(axis=0)) / X.std(axis=0)
# y = (y - y.mean()) / y.std()
#
# # Convert to NumPy
# X = X.to_numpy()
# y = y.to_numpy()
#
# # ---------------- Add bias term ----------------
# X = np.c_[np.ones((X.shape[0], 1)), X]
#
# # ---------------- Functions ----------------
# def hypothesis(X, theta):
#     return np.dot(X, theta)
#
# def cost_function(y_pred, y):
#     m = len(y)
#     return (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
#
# def compute_gradient(y_pred, X, y):
#     m = len(y)
#     return (1 / m) * np.dot(X.T, (y_pred - y))
#
# # ---------------- Training ----------------
# alpha = 0.001
# iterations = 1000
# theta_input = np.zeros(X.shape[1])
#
# for i in range(iterations):
#     y_pred = hypothesis(X, theta_input)
#     cost = cost_function(y_pred, y)
#     gradient = compute_gradient(y_pred, X, y)
#
#     theta_input -= alpha * gradient
#
#     if i % 100 == 0:
#         print(f"Iteration {i} | Cost = {cost:.4f}")
#
# # ---------------- Final output ----------------
# print("\nFinal cost:", cost_function(hypothesis(X, theta_input), y))
# print("Final theta:", theta_input)
#
# def r2_score(y, y_pred):
#     ss_res = np.sum((y - y_pred) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     return 1 - ss_res / ss_tot
#
# print("R²:", r2_score(y, hypothesis(X, theta_input)))







# print(theta)
# print(X.shape)



# import numpy as np
# import pandas as pds
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
#
# # >>>>>>>>>>>  Hypothesis function <<<<<<<<<<<<<<<<<<
#
# def hypothesis(x_values,theta):
#     hypo_element = np.dot(x_values,theta)
#     print(hypo_element)
#     return hypo_element
# # ---------------------------------------------------
#
#
# # >>>>>>>>>>>  Cost function <<<<<<<<<<<<<<<<<<<<<<<<
#
# def cost_function(hypo_values , y_values):
#     m = len(y_values)
#     j_theta = (1/(2 * m)) * np.sum((hypo_values - y_values) ** 2)
#     return j_theta
# # ---------------------------------------------------
#
#
# # >>>>>>>>>>>> Gradient descent <<<<<<<<<<<<<<<<<<<<<
#
# def compute_gradient(hypo_values_gradient,x_value_gradient,y_values_gradient):
#     m = len(y_values_gradient)
#     gradient_values = (1/m) * (np.dot(x_value_gradient.T, (hypo_values_gradient - y_values_gradient)))
#     return gradient_values
# # ---------------------------------------------------
#
# input_x = np.array([[1,1,1],
#                     [1,2,3],
#                     [2,1,3]])
#
# input_theta = np.array([0,0,0])
# input_y = np.array([3,4,5])
#
# h = hypothesis(input_x, input_theta)
# cost = cost_function(h, input_y)
# gradient = compute_gradient(h,input_x,input_y)
#
# print("Hypothesis",h)
# print("cost", cost)
# print("gradient", gradient)
#
# # ----------  Load data -------------------------
# data_frame = pds.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
# print("------ Initial Shape of the data which is provided ------")
# X = data_frame.drop(["disease_score_fluct"],axis=1)
# print(X.columns)
# y = data_frame["disease_score_fluct"]
# print()
# print(data_frame.shape)
# print()
# print(X.shape,y.shape)
# print()
# print(data_frame.columns)
# print()
# plt.plot(X,y)
# plt.xlabel("patient_details_apart_from_'disease_score_fluct'")
# plt.ylabel("disease_score_fluctuation")
# plt.show()
# # ----------  Divide data -----------------------
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=999)
# print("------ After splitting the data, shape of the x_train, x_test: ------")
# print(x_train.shape,x_test.shape)
# print()
# print("------ After splitting the data, shape of the y_train, y_test: ------")
# print(y_train.shape,y_test.shape)
# print()
#
# # ----------  Standardise data ------------------
# aggregate = StandardScaler()
# print(aggregate.fit(x_train))
# print(" ------ Standardise data ------ ")
# print(aggregate.mean_)
# print()
# x_trained_scaled = aggregate.transform(x_train)
# x_test_scaled = aggregate.transform(x_test)
#
#
# # ----------  Initializing the model ------------
# model = LinearRegression()
# model.fit(x_trained_scaled,y_train)
#
#
# # ---------- Model Training ---------------------
# y_predict = model.predict(x_test_scaled)
# average_squared_error = mean_squared_error(y_test, y_predict)
# root_mean_squared_error = np.sqrt(average_squared_error)
# r2 = r2_score(y_test, y_predict)
# print()
# print("Root Mean Square Error :", root_mean_squared_error)
# print()
# print("R^2 Score:", r2)


# ---------- Done !!! ----------------------------


# import numpy as np
# from matplotlib import pyplot as plt   # Uncomment to plot cost
#
# # ---------- Hypothesis ----------
# def hypothesis(x_values, theta):
#     return np.dot(x_values, theta)
#
#
# # ---------- Cost Function ----------
# def cost_function(hypo_values, y_values):
#     m = len(y_values)
#     return (1 / (2 * m)) * np.sum((hypo_values - y_values) ** 2)
#
#
# # ---------- Gradient Computation ----------
# def compute_gradient(hypo_values, x_values, y_values):
#     m = len(y_values)
#     return (1 / m) * np.dot(x_values.T, (hypo_values - y_values))
#
#
# # ---------- Gradient Descent ----------
# def gradient_descent(x_values, y_values, theta, learning_rate=0.01, iterations=1000):
#     cost_history = []
#
#     for i in range(iterations):
#         h = hypothesis(x_values, theta)
#         grad = compute_gradient(h, x_values, y_values)
#         theta = theta - learning_rate * grad
#         cost = cost_function(h, y_values)
#         cost_history.append(cost)
#
#         if i % 100 == 0:  # Print every 100 iterations
#             print(f"Iteration {i}: Cost = {cost:.4f}")
#
#     return theta, cost_history
#
#
# # ---------- Input Data ----------
# input_x = np.array([[1, 1, 1], [1, 2, 3], [2, 1, 3]])  # shape (m,n)
# input_y = np.array([3, 4, 5])
# input_theta = np.zeros(input_x.shape[1])  # initialize theta as zeros
#
# # ---------- Run Gradient Descent ----------
# final_theta, cost_history = gradient_descent(input_x, input_y, input_theta, learning_rate=0.1, iterations=1000)
#
# print("\nFinal Theta:", final_theta)
# print("Final Cost:", cost_history[-1])
#
# # ---------- Plot Cost History ----------
# plt.plot(cost_history)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.title("Cost vs Iterations")
# plt.show()



# import numpy as np
# from matplotlib import pyplot as plt
#
#
# # ---------- Hypothesis ----------
# def hypothesis(X, theta):
#     return np.dot(X, theta)
#
#
# # ---------- Cost Function ----------
# def cost_function(h, y):
#     m = len(y)
#     return (1 / (2 * m)) * np.sum((h - y) ** 2)
#
#
# # ---------- Derivative (Gradient) ----------
# def compute_derivative(X, h, y):
#     m = len(y)
#     return (1 / m) * np.dot(X.T, (h - y))
#
#
# # ---------- R-squared ----------
# def r2_score(y, y_pred):
#     ss_res = np.sum((y - y_pred) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     return 1 - (ss_res / ss_tot)
#
#
# # ---------- Main ----------
# def main():
#     # Input data (bias included)
#     X = np.array([
#         [1, 1],
#         [1, 2],
#         [1, 3]
#     ])
#     y = np.array([2, 3, 4])
#
#     # Initialize parameters
#     theta = np.zeros(X.shape[1])
#     alpha = 0.1
#     iterations = 1000
#
#     cost_history = []
#
#     # Gradient Descent
#     for i in range(iterations):
#         h = hypothesis(X, theta)
#         grad = compute_derivative(X, h, y)
#         theta = theta - alpha * grad
#
#         cost = cost_function(h, y)
#         cost_history.append(cost)
#
#         if i % 100 == 0:
#             print(f"Iteration {i}: Cost = {cost:.4f}")
#
#     # Final results
#     y_pred = hypothesis(X, theta)
#     r2 = r2_score(y, y_pred)
#
#     print("\nFinal Theta:", theta)
#     print("Final Cost:", cost_history[-1])
#     print("R-squared:", r2)
#
#     # Plot cost vs iterations
#     plt.plot(cost_history)
#     plt.xlabel("Iterations")
#     plt.ylabel("Cost")
#     plt.title("Cost vs Iterations")
#     plt.show()
#
#
# # ---------- Run ----------
# main()


# import numpy as np
# from Question_2 import X,y
#
# # >>>>>>>>>>> Hypothesis function <<<<<<<<<<<<<<<<<<
# def hypothesis(x_values, theta):
#     return np.dot(x_values, theta)
#
#
# # >>>>>>>>>>> Cost function <<<<<<<<<<<<<<<<<<<<<<<<
# def cost_function(hypo_values, y_values):
#     m = len(y_values)
#     return (1 / (2 * m)) * np.sum((hypo_values - y_values) ** 2)
#
#
# # >>>>>>>>>>> Gradient computation <<<<<<<<<<<<<<<<<
# def compute_gradient(hypo_values, x_values, y_values):
#     m = len(y_values)
#     return (1 / m) * np.dot(x_values.T, (hypo_values - y_values))
#
#
# # >>>>>>>>>>> R-squared function <<<<<<<<<<<<<<<<<<
# def r2_score(y, y_pred):
#     ss_res = np.sum((y - y_pred) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     return 1 - (ss_res / ss_tot)
#
#
# # ------------------ Input ------------------
# X_mean = X.mean(axis =0)
# X_std = X.std(axis=0)
# X = (X - X_mean) / X_std
#
# input_x = X
# input_y = y
# theta = np.zeros(X.shape[1])
#
# alpha = 0.1
# iterations = 1000
#
# y_pred = hypothesis(input_x, theta)
# r2 = r2_score(input_y, y_pred)
#
# # ------------------ Gradient Descent ------------------
# for i in range(iterations):
#     h = hypothesis(input_x, theta)
#     cost = cost_function(h, input_y)
#     gradient = compute_gradient(h, input_x, input_y)
#
#     theta = theta - alpha * gradient
#
#     if i % 100 == 0:
#         print(f"Iteration {i} | Cost = {cost:.4f}")
#
#
# # ------------------ Final Output ------------------
#
# print("\nFinal theta:", theta)
# print("Final cost:", cost_function(y_pred, input_y))
# print("R-squared:", r2)