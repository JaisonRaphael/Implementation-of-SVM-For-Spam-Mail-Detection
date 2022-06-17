# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the required packages. 
2.Import the dataset to operate on. 
3.Split the dataset. 
4.Predict the required output. 
5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: V.JaisonRaphael
RegisterNumber:  212221230038
*/
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:
Data Head:
(![r1](https://user-images.githubusercontent.com/94165957/174222567-0ed3e454-51f7-448f-ac77-e27da35d382c.png)
Data info:
![r2](https://user-images.githubusercontent.com/94165957/174222660-fc0edd68-5bf7-4b57-b153-671166427ef1.png)
Data isnull():
![r3](https://user-images.githubusercontent.com/94165957/174222737-fad6b01b-94cf-4518-a67f-c714548dfd7c.png)
y_pred:
![r4](https://user-images.githubusercontent.com/94165957/174222783-effb792c-3406-4cec-acaf-86f8396b3c72.png)
Accuracy:
![r5](https://user-images.githubusercontent.com/94165957/174222805-395a466e-e086-491f-96ab-2fec419be0a1.png)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
