import numpy

#Our initial neural net.
def NN(m1, m2, w1, w2, b):
    #Default initial values.
    w1 = Rnd()
    w2 = Rnd()
    b2 = Rnd()
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)



#region Cost.

#In order to adjust the weights, our net has to have a way to gradually correct itself. A common place is a the squared error function.
#There are some nice mathematical implications that lead to squaring the error.
#https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/
#https://youtu.be/c6NBkkKNZXw?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
def squaredError(prediction, target):
    return prediction - target ** 2


#endregion



#region Activators.

#Sigmoid activator, the classical example. Nowadays, ReLU is much more common, mostly due to the high computational cost of sigmoid.
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

#endregion


#region Helpers.

#Just getting a random number.
def Rnd():
    return numpy.random.randn()

#endregion

