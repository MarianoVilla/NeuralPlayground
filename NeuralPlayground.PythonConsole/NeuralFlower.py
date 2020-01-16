import matplotlib.pyplot as plt
import numpy as np

#Most of the code comes from Jon Como's implementation: https://github.com/JonComo/flowers/blob/master/flowers.ipynb
#Amazing explanation in his Youtube series: https://www.youtube.com/watch?v=ZzWaow1Rvho


#region Data.
data = [[3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]]

mystery_flower = [4.5, 1]


w1 = Rnd()
w2 = Rnd()
b = Rnd()
learning_rate = 0.1

#endregion

#region Activation.

#TODO: don't use sigmoid!
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Sigmoid's derivative. Gives us a description of sigmoid's gradient.
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

#endregion



#region Training
def train():

    for i in range(1000):
        ri = np.random.randint(len(data))
        point = data[ri]

        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z)

        target = point[2]

        #The squared error is the cost of our prediction.
        cost = np.square(pred - target)

        #Now we get the derivative of the cost with respect to each parameter.
        #With respect to pred:
        dcost_pred = 2 * (pred - target)
        #With respect to z (gives us the gradient of sigmoid evaluated at z):
        dpred_dz = sigmoid_prime(z)
        #With respect to weights 1 and 2 is just X and Y:
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        #And with respect to the bias is just 1:
        dz_db = 1

        dcost_dz = dcost_pred * dpred_dz
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 -= learning_rate * dcost_dw1
        w2 -= learning_rate * dcost_dw2
        b -=  learning_rate * dcost_db

        return costs, w1, w2, b

costs, w1, w2, b = train()

#endregion






#region Helpers.

#Just getting a random number.
def Rnd():
    return numpy.random.randn()

#endregion