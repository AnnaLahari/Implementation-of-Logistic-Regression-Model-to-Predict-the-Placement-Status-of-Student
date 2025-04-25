# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.LAHARI  
RegisterNumber: 212223230111 
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn .preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver= "liblinear")
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

## data.head()

![image](https://github.com/user-attachments/assets/8f39864c-6a96-4e84-9685-02d45a6f91de)

## data1.head()

![image](https://github.com/user-attachments/assets/f4014dc5-709e-49a9-8f51-70b79b183ae2)

## isnull()

![image](https://github.com/user-attachments/assets/7a08c4fd-fd1d-4337-96dd-929e9d19d7f7)

## duplicated()

![image](https://github.com/user-attachments/assets/5117141f-5935-405f-b4e8-212918684637)

## data1

![image](https://github.com/user-attachments/assets/c893eddb-3d5e-415a-b114-1bea73a1620d)

## X

![image](https://github.com/user-attachments/assets/12e98d4c-f7ca-4790-bed7-f488e4620679)


## y

![image](https://github.com/user-attachments/assets/cfe4f2eb-204b-4b25-8c3d-8b01f93d2d4d)

## y_pred

![image](https://github.com/user-attachments/assets/4a446cb6-bf8c-457e-ad9b-bd5182e157c2)

## confusion matrix

![image](https://github.com/user-attachments/assets/2cfaf0d2-9116-45aa-aa7f-0bb27f253290)

## classification report

![image](https://github.com/user-attachments/assets/4874e8c0-310d-48b4-bc08-fc3d55ca648a)

## prediction

![image](https://github.com/user-attachments/assets/5466fef0-bed0-443f-b889-49efb39b14a3)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
