from __future__ import division
import numpy as np
import math
from pylab import plot, show, xlabel, ylabel
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale



def gradientDescent(x, y, theta, alpha, m, numIterations):
    J_history = np.zeros(shape=(numIterations, 1))
    xTrans = x.transpose()
    for i in range(0, numIterations):
    	hypothesis = 1.0 / (1.0 + np.exp((-1) *np.dot(x, theta)))
    	 #calculates the hypothesis by multiplying the parameter theta with input and the value is subjected to the sigmoid function
        loss = hypothesis - y 
        #calculats the difference between the hypothesis and the actual value
        cost = (-1)*np.sum((y*np.log(hypothesis)) + ((1-y)*np.log(1-hypothesis))) / (m)
        #calculates the cost function for logistic regression
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
        J_history[i][0] = cost
    return theta, J_history


a = [
[1,0,0],[1,0,0],
]        #add the number of items in your sample set and number of zeros as the number of variables, this has two zeros for two variables

X1 = [[],[]] #input values for first variable

X2 = [[],[]] #input values for second variable

"""
As many lists are created as the number of variables , here two variables has been set
"""



a[0][1] = X1[0][0]
a[1][1] = X1[1][0] 

a[0][2] = X2[0][0]
a[1][2] = X2[1][0]

""" Here the input values are being stored in the matrix A in the secnd and third columns"""

for x in a:
	print x
	print ""



b = [[],[]] #store results here 

for y in b:
	print y
	print ""

print "X Matrix"
x = np.asarray(a) 
print x
print "" #prints X matrix
print "Y Matrix"
y = np.asarray(b)
print y
print "" #prints Y matrix

print "Matrix A (Input Matrix)"


m, n = np.shape(x)
numIterations= 1000000
alpha = 0.015
theta = np.ones(n)

theta, J_history = gradientDescent(x, y, theta, alpha,m,numIterations)
print "theta"
print(theta)
print ""
print "final cost is"
hx = 1.0 / (1.0 + np.exp((-1) * x.dot(theta))) #Hypothesis is calculated with the theta generated from the previous gradient descent
cost = np.sum((y*np.log(hx)) + ((1-y)*np.log(1-hx))) 
hi = (-1)/(m)
costfin = hi*cost
print costfin

meany = np.mean(y)
print meany

sumsqmeany = np.sum((y - meany)**2)
print sumsqmeany

sumsqmeanysum = np.sum((y - hx)**2)/sumsqmeany

R = 1 - sumsqmeanysum
print "The R value is:"
print R 

zzzz = np.array([1.0,  x1, x2 ]).dot(theta)  #enter x1 and x2 values
predict = 1.0 / (1.0 + np.exp((-1) *zzzz))
print 'Predicted strength of promoter : %f' % (predict)

