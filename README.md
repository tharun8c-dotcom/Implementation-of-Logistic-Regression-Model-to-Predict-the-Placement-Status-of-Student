# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by removing unwanted columns, encoding categorical values, converting status to 0/1, and scaling features.
2. Split the data into training and testing sets.
3. Train a Logistic Regression model using the training data and predict placement status on the test data.
4. Evaluate the model using accuracy, classification report, and confusion matrix visualization.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R Tharun Rathish
RegisterNumber:  25018411
*/


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

data = data.drop(["sl_no", "salary"], axis=1)


data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})


X = data.drop("status", axis=1)
y = data["status"]

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()


```

## Output:
<img width="726" height="313" alt="image" src="https://github.com/user-attachments/assets/a229a562-bd42-474f-af81-2ad1409c6fb4" />

<img width="710" height="475" alt="image" src="https://github.com/user-attachments/assets/8f5f8e51-3507-45eb-8aaf-15a276b1f306" />

<img width="589" height="284" alt="image" src="https://github.com/user-attachments/assets/bc109864-b11a-4e06-9ae6-a1db249ccaf8" />

<img width="530" height="453" alt="image" src="https://github.com/user-attachments/assets/1ade55fb-207d-47f3-a0a2-b555436ab9fa" />






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
