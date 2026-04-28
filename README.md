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
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Startup.csv")
X=data['R&D Spend'].values
Y=data['Profit'].values
X=(X-X.mean())/X.std()
m=0
b=0
learning_rate=0.1
epoches=1000
n=len(X)
for i in range(epoches):
    Y_predict=m*X+b
    dm=(-2/n)*np.sum(X*(Y-Y_predict))
    db=(-2/n)*np.sum(Y-Y_predict)
    m=m-learning_rate*dm
    b=b-learning_rate*db
print("Slope(m):",m)
print("Intercept(b):",b)
y_predict=m*X+b
plt.scatter(X,Y)
plt.plot(X,Y_predict)
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()
```

## Output:
<img width="889" height="519" alt="Screenshot 2026-04-27 085452" src="https://github.com/user-attachments/assets/e44fef78-af82-43d2-8b4b-e3993a6f8743" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
