import pandas as pds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def main():
    data = pds.read_csv("../Lab6/data.csv")
    x = data.drop(["diagnosis", "id",'Unnamed: 32'], axis=1)
    y = data["diagnosis"]
    x = x.values
    y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    # <<<<<<<<<<<< Train_Test_split >>>>>>>>>

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.70,random_state=999)

    # <<<<<<<<<<<< Standardization >>>>>>>>>>>>>
    aggregate = StandardScaler()
    aggregate.fit(x_train)
    x_train_scaled = aggregate.transform(x_train)
    print(type(x_train_scaled))
    x_test_scaled = aggregate.transform(x_test)

    # <<<<<<<<<<<< Model Training >>>>>>>>>>>>>>
    model = LogisticRegression()
    model.fit(x_train_scaled,y_train)

    y_predict = model.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy of the model is : ", accuracy)



if __name__ == "__main__":
    main()