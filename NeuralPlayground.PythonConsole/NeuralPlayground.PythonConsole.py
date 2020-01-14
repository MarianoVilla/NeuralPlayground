#A neural network, I've come to realize after reasearching a bit, is a brute force solution to emulate knowledge-based behavior.
#At first glance, I expected it to be more similar to actual AI (although I haven't quite defined that term). Instead, I found neural networks to be more similar to a correlation machine than anything else.
#The question arises: but how do it know? And the answer, as with electronics, is: it doesn't. NNs emulate the external appeareance of intelligent decision making.
#An introspective research indicates that a semantic training has to be made in order to emulate more complex forms of intelligence, such as ours.
#How do I know that a chair is a chair? Image processing is a step for sure, but that has to be couped with a symbolic appreciation of the input.
#These symbolic connotations seem to pair with a functional, utilitarian perspective. I know that a chair has to be usable in a certain way; i.e., I have to be able to sit on it.
#Hence, besides the interpretation of luminic stimuli, there's a verbal knowledge that stands close to it, putting a "what can this do"; or rather "what can I do with this?".
#That quick fix, tho, would give any object with a fairly plane surface a chair. Even if we combined the image interpretation, which would naturally filter some of the more grotesquely non-chair objects,
#we would still have to bear with a large amount of chair-like objects being incorrectly classified as such.
#That's problematic. Turning to myself again, I ask: how do I know that, for instance, this box is not a chair?
#Well, it looks like I have an alternative category for that box, which seems stronger on it than the chair one.
#I'm unsure wheather we're talking of an effective negative knowledge (i.e., that isn't a chair), or a stronger afirmation (i.e., that is a box).
#Once again, functional semantics come in to rescue: I know that boxes are mainly used for storing/transporting thing rather than sitting.
#There's an evolutonary background to this: storing, saving, the effectiveness of boxes as containers rather than chairs. 
#All of this seems to be pointing towards a mediate knowledge base; that is, a filogenic preprocessing of the object I'm seing.
#That's a weird outcome, actually, because it directs us once again towards a sort of unconscious understanding.
#What I mean is that I don't necessarily have to know every bit of background on the decision to make boxes containers rather than chairs.
#There's a leap of faith there: I've been told that boxes are better containers than chairs, and I blindly trust in that saying, and use it in my decision process as an ethereal database.
#Indeed, this nice article homes.cs.washington.edu/~pedrod/papers/cacm12.pdf states that "Every learner must embody some knowledge or assumptions beyond the data it is given in order to generalize beyond".

#(One thing that has got me thinking is if the functional approach isn't reductionist. But after going around it for a while, I came to realize that a N-to-1 mapping (a function) is effectively what's going on in cognition.
#My reasoning works around the fundamental pilars of onthology: the principles of identity and no contradiction. Simply because a thing is identical to itself, and a thing cannot be itself and its opposite at the same time.
#Hence, although being achieved through a dynamic and complex evaluation process, no matter how many the features or how complex the function of a class may be, a recognition has to end in a value, or at least an uncertainty, both of which go well with our thought principles.
#Still, an important factor to keep in mind is that the more features and odd relations they have in a class's function, the more expensive the computation process of its classification.)



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



#region Activation.

#The activation function is the part of an AN that "decides" wheather the (weighed sum + bias) result of the input should "fire" the neuron.
#The simplest approach is a threshold based function, that returns 1 is the input is higher than a given threshold and 0 otherwise.
#That could be defined as a one liner step function:
def simple_threshhold(input):
    return 1 if(input > threshold) else 0
#This is useful if we're dealing with pure binary decision logic, but in most cases we'll need a soft activation function.
#To achieve that analog-like decision making, we can't simply use a linear function (f(x) = cx),
#'cause that would devoid the purpose of the feedback, being c a constant.
#That's when we arrive to sigmoid, an "S"-shaped, step-like function.

#Sigmoid activator, the classical example, is defined as: f(x) = 1 / 1 + exp(-x).
#Nowadays, ReLU is much more common, mostly due to the high computational cost of sigmoid.
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))
#Sigmoid allows a step-like, constrained set of outputs, but keeping the analog movement to allow gradual responses.

#An evident issue with sigmoid is that the gradient tends to zero when x is both huge or small, which makes our correction rate equally tiny.

#Sigmoid's problems:
# * Vanishing gradient.
# * Non-zero centered output.
# * Sigmoids saturate and kill gradients.
# * Slow convergence.
# * exp(x) is computationally expensive.

#A solution is scaling sigmoid. That gives us the hyperbolic tangent (tahn) function, which is purely: f(x) = 1 - exp(-2x) / 1 + exp(-2x).
#In terms of sigmoid, tahn is: tahn(x) = 2.sigmoid(2x) - 1
#Tahn solves the non-zero centered problem, but still has a vanishing gradient.

#ReLU (Rectified Linear Units) is a much more popular activation function. It's defined as: ReUu(x) = max(0,x),
#that is to say, ReLU(x) = 0 if(x < 0) else x

#This describes a linear-like graph starting at (0,0) (although ReLu is non linear).
#A nice property of ReLu is that is has a [0,inf) range, which means that for values of x smaller than 0, the given neuron won't activate.
#This last bit is a nice optimization we get for free. Still, that very same property derives into the "dying ReLU problem".
#To deal with this, we give the values of ReLU when x < 0 a little gradient; that's called leaky ReLu.

#With respect to sigmoid/tahn, ReLU:
# *Has faster convergence.
# *Doesn't have the vanishing gradient problem.
# *Is cheaper to compute.

#Variants of ReLU.
#As noted, leaky ReLU is an attemp to fix the "dying ReLU" problem by giving it a small slope when x < 0.
#A further improvement is PReLU, which makes the negative slope dynamic.
#Another version is ELU, which follows the same pattern as ReLU, but uses an exponential factor for x < 0.
#Maxout is a generalization of ReLU/leaky ReLU, defined as: max(wT1 + b1, wT2x + b2).
#Maxout is designed to work with the droput regularization technique.


#Cool readings:
#https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
#https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
#https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044


#endregion


#region Helpers.

#Just getting a random number.
def Rnd():
    return numpy.random.randn()

#endregion

