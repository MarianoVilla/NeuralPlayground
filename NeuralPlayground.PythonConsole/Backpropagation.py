#A cool way of getting the gradient of our loss function L is through backpropagation.
#Skipping the whole explanation, we can simply remember that:
#--The derivative of each variable in a function tells us the sensitivity of the whole expression on its value.--

#When we have a function that takes multiple parameters, like f(x,y,z) = (x+y)z, we can use the chain rule. First, we have to break the expression in: q = x + y and f = qz.
#Under this simpler conditions, we can compute the derivatives of both operations: so ∂f/∂q=z and ∂f/∂z=q. And since q is addition of x and y, ∂q/∂x=1 and ∂q/∂y=1.
#Since we actually only care for the derivatives of f with respect to its inputs x, y, z, we don't necessarily need to compute q's derivative.
#There's where the chain rule comes in handy: it tells us that  ∂f/∂x=∂f/∂q.∂q/∂x
#An example of this in python:

def derivative_example():
    # set some inputs
    x = -2; y = 5; z = -4

    # perform the forward pass
    q = x + y # q becomes 3
    f = q * z # f becomes -12

    # perform the backward pass (backpropagation) in reverse order:
    # first backprop through f = q * z
    dfdz = q # df/dz = q, so gradient on z becomes 3
    dfdq = z # df/dq = z, so gradient on q becomes -4
    # now backprop through q = x + y
    dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule!
    dfdy = 1.0 * dfdq # dq/dy = 1

    #This output vectos is ∇f!
    return dfdx, dfdy, dfdz

