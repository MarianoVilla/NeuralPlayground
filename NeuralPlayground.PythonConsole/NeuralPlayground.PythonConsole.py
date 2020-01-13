
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
#The effect is simple: since we have a quadratic function (i.e., a parabola), the further the prediction is from the target, the higher the cost.
#If we were to only try to minimize the difference between the prediction and the target, we would infinitely minimize it.
#That is to say that, if our logic is "the lesser the better", our function would get stucked going down forever. So instead we use the squared error: (n)^2 is always positive.

#A nice article on the mean squared error (MSE) function: https://www.freecodecamp.org/news/machine-learning-mean-squared-error-regression-line-c7dde9a26b93/
#Mainly, we have a line defined by a simple linear equation: y = mx+b.
#The goal is to find the values for m (the slope) and b (the y-intercept) that minimize the mean of the squared error of every data point with respect to our line.

#To find it, we use some equations, that come down to, in pseudocode:
# Slope = m = (xy' x'y') / (x^2)' - (x')^2
# Y-intercept = b = y' - mx'

#Having the next equations for each piece:
# x' = sum_every(x) / n
# y' = sum_every(y) / n
# xy' = sum_every(xy) / n
# (x^2)' = sum_every(x^2) / n
#Note: sum_every(n), as you might infeer, is equivalent to a sigma on the parameter.



#https://youtu.be/c6NBkkKNZXw?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
def cost(prediction, target):
    return (prediction - target) ** 2

#We don't actually use the error, we calculate the slope of the function at the given prediction.

#The slope is (the derivative of cost(b)):
def slope(b):
    return (b-4) * 2

#So our job, when trimming the variables at hand, is to minimize the cost, not of any single error, but of every error.
#Hence, Cost = (C1 + C2 + C3 ... + Cn), with C(x)n = (model(xn) - targetn)^2

#How can we find the best values for our variables, w and b? Gotta find their partial derivatives with respect to each.
#So if the cost function is: cost(w,b) = (w.1 + b - 2)^2 + (w.2 + b - 4)^4 + (w.4 + b - 5)^2, our partial derivatives are:
#dcdw(w,b) = 2.(w.1 + b - 2) . 1 + 2.(w.2 + b - 4).2 + 2.(w.4+b-5).4
#dcdb(w,b) = 2.(w.1 + b - 2) + 2.(w.2 + b - 4) + 2.(w.4 + b - 5)


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

