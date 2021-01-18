'''
HW3 p1_1.py
Written by Kookhoi Kim
'''

import csv
import numpy as np
import matplotlib.pyplot as plt


'''
 The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
 prices and the demand for clean air', J. Environ. Economics & Management,
 vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
 ...', Wiley, 1980.   N.B. Various transformations are used in the table on
 pages 244-261 of the latter.

 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
'''

#### load dataset ####
attr_house = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
house = []

with open('p1/housing.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        house.append(row[0].split())

house = np.asarray(house, dtype=np.float)
print(attr_house, '\n', house)

#### 1.a) check which attribute is bool type ####
for i in range(house.shape[1]):
    if np.array_equal(house[:, i], house[:, i].astype(bool)):
        print(attr_house[i] + ' is bool type(binary)')

    
#### 1.b) calculate the correlation between attributes and target attribute ####
corr = []
max_idx = 0
for i in range(house.shape[1] - 1):
    corr.append(np.correlate(house[:,i], house[:, -1])[0])
    if corr[i] > corr[max_idx]:
        max_idx = i

print('correalation with target attribute: ', corr)
print('max correlation attritube is ', attr_house[max_idx])


#### 1.c) plot figures of the correlation with target attribute ####

for i in range(house.shape[1] - 1):
    x_data = house[:, i]
    scatter = plt.scatter(x_data, house[:, -1], c='r', s=8)
    plt.xlabel(attr_house[i])
    plt.ylabel(attr_house[-1])
    plt.savefig('p1/' + str(i+1) + '.' + attr_house[i] + '.png', dpi=300)
    plt.clf()
    # if you want to see, use this
    #plt.show() 



#### 1.d) calculate total correlation using corrcoef function ####
total_corr = np.corrcoef(np.transpose(house))
print('total correlate :', total_corr)

np.fill_diagonal(total_corr, 0)
max_idx = np.where(total_corr == total_corr.max())
print('{} and {} have maximum correlation'.format(attr_house[max_idx[0][0]], attr_house[max_idx[1][0]]))


#### 2.a) 