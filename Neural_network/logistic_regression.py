import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
####################################################### important libraries fetched
# loading the data set 
br = load_breast_cancer()
X = br.data
y = br.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 2)

# noramlisation of features 
X_train = (X_train - np.mean(X_train, axis = 0))/np.std(X_train, axis = 0)
X_test = (X_test - np.mean(X_test, axis = 0))/np.std(X_test, axis = 0)

m = len(X_train) # number of datapoints
n = len(br.feature_names) # number of features

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cost_function(w, b, m, X, y):
    cost = 0
    for i in range(m):
        cost += (1/(2*m))*(-1)*((y[i]*np.log(sigmoid(np.dot(w, X[i]) + b))) + ((1-y[i])*np.log(1-sigmoid(np.dot(w, X[i]) + b))))
    return cost

def gradient_decent(w,b,alpha,X,y):
    w_gradient = np.zeros(n)
    b_gradient = 0
    for i in range(m):
        w_gradient += (-1/m) * X[i] * (y[i] - sigmoid(np.dot(w, X[i]) + b))
        b_gradient += (-1/m) * (y[i] - sigmoid(np.dot(w, X[i]) + b))
    w -= alpha * w_gradient
    b -= alpha * b_gradient
    return w, b

# initialize paramenters
w = np.zeros(n)
b = 0
alpha = 0.01
epochs = 3000
costs = []
for i in range(epochs):
    if (i % 50 == 0):
        print("Epoch:", i)
    w, b = gradient_decent(w, b, alpha, X_train, y_train)
    costs.append(cost_function(w, b,m, X_train, y_train))
iteration = np.arange(0, epochs, 1)

plt.plot(iteration, costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()


# no need of regularisation term as cost function does not oscillate
total_cost =  cost_function(w,b,len(X_test),X_test,y_test)
print(f"Your total error in testing data is: {round(total_cost*100,2)}%")
y_GUESS = []
for i in range(len(X_test)):
    if sigmoid(np.dot(w, X_test[i]) + b) >= 0.5: # using threshold as 0.5
        y_GUESS.append(1)
    else:
        y_GUESS.append(0)
x_new = np.arange(0,len(X_test),1)
plt.scatter(x_new, y_GUESS, color = 'black', marker = '*')
plt.show()
