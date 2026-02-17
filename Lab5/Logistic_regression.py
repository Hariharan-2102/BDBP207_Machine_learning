import numpy as np
import pandas as pds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib import pyplot as plt


def main():

    # <<<<<<<<<<<< Load data >>>>>>>>>>>>>>
    data = pds.read_csv("data.csv")
    x = data.drop(["diagnosis", "id",'Unnamed: 32'], axis=1)
    y = data["diagnosis"]

    x = x.values
    y = y.values

    # print(x.shape)

    # <<<<<<<<<<<< Feature Encoding >>>>>>>>>>>

    le = LabelEncoder()
    y = le.fit_transform(y)
    # print(y)
    #
    #
    # # data['diagnosis'] = data['diagnosis'].replace('M','1')
    # # data['diagnosis'] = data['diagnosis'].replace('B','0')
    # # print(data["diagnosis"])
    #
    # <<<<<<<<<<<< Train_Test_split >>>>>>>>>

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.70,random_state=999)
    # print(x_train, x_test)

    # # <<<<<<<<<<<< Standardization >>>>>>>>>>>>>

    aggregate = StandardScaler()
    aggregate.fit(x_train)
    x_train_scaled = aggregate.transform(x_train)
    print(type(x_train_scaled))
    # sys.exit(0)
    x_test_scaled = aggregate.transform(x_test)
    # print(x_train_scaled)
    # print(x_test_scaled)

    theta =  np.zeros(x_train_scaled.shape[1])
    # print(theta)

    def hypothesis_logistic_regression(x_hypo,theta_hypo):
        hypothesis = np.dot(x_hypo,np.transpose(theta_hypo))
        return hypothesis

    def sigmoid_function(x_sig,theta_sig):
        sigma = hypothesis_logistic_regression(x_sig,theta_sig)
        sigma_g = 1/(1 + np.exp(-sigma))
        return sigma_g

    def sigmoid_nabla(y_for_nabla,x_train_scaled_nabla,sigma_g):
        nabla = np.dot(np.transpose(x_train_scaled_nabla),y_for_nabla - sigma_g)
        return nabla

    def cost_function(y_train_cost,x_train_scaled_cost,theta_cost):
        print(y_train_cost.shape,x_train_scaled_cost.shape,theta_cost.shape)
        cost = np.sum(y_train_cost * np.log(sigmoid_function(x_train_scaled_cost,theta_cost)) + (1 - y_train) * np.log(1 - sigmoid_function(x_train_scaled,theta)))

        return cost

    alpha = 0.001
    while True:

        sigmoid = sigmoid_function(x_train_scaled,theta)
        old_cf = cost_function(y_train,x_train_scaled,theta)
        old_nabla = sigmoid_nabla(y_train,x_train_scaled,sigmoid)

        theta = theta + alpha * sigmoid_nabla(y_train,x_train_scaled,sigmoid)
        new_cf = cost_function(y_train,x_train_scaled,theta + alpha * sigmoid_nabla(y_train,x_train_scaled,sigmoid))
        new_sigmoid = sigmoid_function(x_train_scaled,theta)
        new_nabla = sigmoid_nabla(y_train,x_train_scaled,new_sigmoid)

        print("Cost Function : ",new_cf)


        if abs(new_cf - old_cf) < 100 and abs(np.linalg.norm(new_nabla) - np.linalg.norm(old_nabla)) < 100:
            break


    # print(hypothesis_logistic_regression(x_train_scaled,theta))
    y_pred = (sigmoid_function(x_test_scaled,theta) >= 0.5).astype(int)
    print(y_pred)
    print(len(x_test_scaled))

    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy of the model is : ",accuracy)
    conf_matrix = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix : \n",conf_matrix)

    plot_x = hypothesis_logistic_regression(x_train_scaled, theta)
    x_axis = []
    for values in plot_x:
        values = float(values)
        x_axis.append(values)
    x_axis.sort()
    # print(x_axis)
    y_axis = [(1/(1+np.exp(-x))) for x in x_axis]


    plt.plot(x_axis,y_axis)
    plt.show()

if __name__ == "__main__":
    main()


