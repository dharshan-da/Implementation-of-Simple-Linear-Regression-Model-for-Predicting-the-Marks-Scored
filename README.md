# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Step 1: Import NumPy, Pandas, and Matplotlib libraries. Load the Startup.csv dataset and read R&D Spend as input X and Profit as output Y.

Step 2: Normalize the input data X using mean and standard deviation. Initialize slope m = 0, intercept b = 0, learning rate 0.1, epochs 1000, and sample size n.

Step 3: For each epoch, calculate predicted values using Y_predict = mX + b, find gradients dm and db, then update m and b using gradient descent formula.

Step 4: Print the final slope and intercept values. Plot the scatter graph of dataset points and regression line using Matplotlib.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dharshan Sri D A 
RegisterNumber: 212225230055 
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X=np.array([1,2,3,4,5]).reshape(-1,1)
Y=np.array([77,83,87,92,99])
model=LinearRegression()
model.fit(X,Y)
m=model.coef_[0]
b=model.intercept_
print("slope(m):",m)
print("intercept(b):",b)
X_input=float(input("Enter hours studied"))
predicted_marks=model.predict([[X_input]])
print("Predicted Marks:",predicted_marks[0])
Y_pred=model.predict(X)
plt.scatter(X,Y,label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("Hours Studied:")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression (Using sklearn)")
plt.legend()
plt.show()

```

## Output:
<img width="695" height="428" alt="Screenshot 2026-04-21 092838" src="https://github.com/user-attachments/assets/c2bc7a14-fbee-4be5-b81d-f80cd52e6927" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
