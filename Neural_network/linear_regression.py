import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the features
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
# Initialize parameters
 
# number of datapoints = m
# number of feature = n
m = len(X_train)
n = len(housing.feature_names)

def gradient_descent(w,b, alpha, X, y):
    w_gradient = np.zeros(n)
    b_gradient = 0
    for i in range(m):
        w_gradient += (-1/m) * X[i] * (y[i] - (np.dot(w, X[i]) + b))
        b_gradient += (-1/m) * (y[i] - (np.dot(w, X[i]) + b))
    w -= alpha * w_gradient
    b -= alpha * b_gradient
    return w, b

def cost_function(w, b, X, y):
    cost = 0
    for i in range(m):
        cost += (1/(2*m)) * (y[i] - (np.dot(w, X[i]) + b)) ** 2
    return cost

# initialize paramenters
w = np.zeros(n)
b = 0
alpha = 0.01
epochs = 500
costs = []
for i in range(epochs):
    if (i % 50 == 0):
        print("Epoch:", i)
    w, b = gradient_descent(w, b, alpha, X_train, y_train)
    costs.append(cost_function(w, b, X_train, y_train))
iteration = np.arange(0, epochs, 1)

plt.plot(iteration, costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

# no need of regularisation term as cost function does not oscillate
m = len(X_test)
total_cost =  cost_function(w,b,X_test,y_test)
print(f"Your total cost for testing data is: {total_cost}")

# we can also use polynomial regression in case our data is non linear
# for example f = w1*x1 + w2*x2  + w3*x1*x2+ w4*(x1**2) + w5*(x2**2)
