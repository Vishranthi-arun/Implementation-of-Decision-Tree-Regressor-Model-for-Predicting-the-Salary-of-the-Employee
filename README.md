# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vishranthi A 
RegisterNumber:  212221230124
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:
![1](https://user-images.githubusercontent.com/93427278/200161294-fd2ca667-ba80-46fd-bd03-9d75408f128f.png)


![2](https://user-images.githubusercontent.com/93427278/200161297-9fada6f3-4e3c-4c5e-bcd6-7b1c83a4cf7f.png)


![3](https://user-images.githubusercontent.com/93427278/200161303-5c5a0d6f-72f1-401e-b3af-7e3984c4f9f0.png)



![4](https://user-images.githubusercontent.com/93427278/200161307-fc42d560-4459-49f9-b924-dc3ae569f4c8.png)



![5](https://user-images.githubusercontent.com/93427278/200161310-0aa24aa4-5f7a-495b-bf76-a010be4678ba.png)



![6](https://user-images.githubusercontent.com/93427278/200161315-b26c3b21-33f3-4855-b735-fa4e0f034cbc.png)


![7](https://user-images.githubusercontent.com/93427278/200161320-79652558-0784-4762-aae3-deb8570a2cb9.png)


![8](https://user-images.githubusercontent.com/93427278/200161322-8c864376-f042-490e-9f4d-2d6197bdced9.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
