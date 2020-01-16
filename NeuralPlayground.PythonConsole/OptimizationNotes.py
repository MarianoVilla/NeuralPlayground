import numpy as np

#Main source: http://cs231n.github.io/optimization-1/
#The general idea is to minimize the loss function. Stepping over some initial and quite awful naive attemtps (like randomly trying out many weights), we can follow the gradient.
#To get the gradient, we can use two approaches: numerical or analytic gradient.
#Here's how the numerical version might look like:

def eval_numerical_gradient(f, x):
    """
    A naive implementation of numerical gradient of f at x.
    - f should be a function that takes a single argument.
    - x is the point (numpy array) to evaluate the gradient at.
    """

    fx = f(x) #Evaluate the function at the original point.
    grad = np.zeros(x.shape)
    h = 0.00001

    #Iterate over all indexes in x.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        #Evaluate function at x+h:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h #Increment by h.
        fxh = f(x) #Evaluate f(x + h)
        x[ix] = old_value #Restore to previous value (very important!)

        #Compute the partial derivative:
        grad[ix] = (fxh - fx) / h #The slope.
        it.iternext() #Step to next dimension.

    return grad

#An evident issue with this way of calculating the gradient is the efficiency, since it linearly depends on the number of parameters.
#We can use the analitical way of calculating the gradient to help this.

#The next thing we see in the reference class is Gradient Descent. The escence is quite simple:
def vanilla_gradient_descent():
    while True:
      weights_grad = evaluate_gradient(loss_fun, data, weights)
      weights += - step_size * weights_grad # perform parameter update
#As it's clear, we simply evaluate the gradient of our loss function for the data, with the given weights (which can be random at the first step).
#Then, we adjust the weights using the gradient times the step_size (that can be arbitrary, or have some thought on it).

#There are variations of Gradient Descent, depending on the size of our data set (e.g., Minibatch Gradient Descent). Still, the core idea remains the same.



