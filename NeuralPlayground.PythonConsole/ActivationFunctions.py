
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)
def relu(x):
    return 0 if x <= 0 else x
def leaky_relu(x):
    return x/100 if x <= 0 else x

def softmax(x, axis):
    """
    Softmax function. In this implementation, we take the axis as a parameter.
    Credit: https://stackoverflow.com/a/58463851/10713658
    Further reading: http://cs231n.github.io/linear-classify/#softmax
    """
    x -= np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)