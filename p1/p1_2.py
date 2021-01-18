'''
HW3 p1_2.py
Written by Kookhoi Kim
'''

import numpy as np
import csv

from sklearn.model_selection import train_test_split


#### load dataset ####
attr_house = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
house = []

with open('p1/housing.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        house.append(row[0].split())

house = np.asarray(house, dtype=np.float)

X_train, X_test, y_train, y_test = train_test_split(house[:,[5, -2]], house[:, -1], test_size=0.1, shuffle=True, random_state=5184)

print('ratio of test num/trainig num : {:.2f}%'.format(X_test.shape[0] / X_train.shape[0] * 100))   # 9:1 -> 11%

def calGradient(w, x, y):
    #import pdb;pdb.set_trace()
    predict_y = np.dot(x, w).flatten()
    
    error = y.flatten() - predict_y
    gradient = -(1.0 / len(x)) * error.dot(x)
    
    return gradient, np.power(error, 2).mean()

W = np.random.randn(X_train.shape[1])
lr = 0.005
tolerance = 1e-5

epoch = 1000

print('Start training {} epochs'.format(epoch))
for n in range(epoch):
    gradient, error = calGradient(W, X_train, y_train)
    new_W = W - lr * gradient
    
    if np.sum(abs(new_W - W)) < tolerance:
        print('converged.')
        break
    
    #print('W is {} and new_W is {}'.format(W, new_W))
    #error = error.sum() / error.shape[0]
    print('Iteration : {:d} \t\t Error : {:.4f}'.format(n, error))

    W = new_W


