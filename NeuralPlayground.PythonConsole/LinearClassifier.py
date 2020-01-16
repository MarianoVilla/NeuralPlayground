#Main source: http://cs231n.github.io/linear-classify/
#Arguably the simplest possible function we could use is a linear mapping: f(xi,W,b)=Wxi+b
#(Note: due to the "bias trick" (merging W and b, putting b as an additional dimension in the W parameter), this function might appear as f(xi, W) = Wxi.)
#Here, the net makes an attempt to guess the class of a given input, and checks with the ground truth. Then, it computes the error and adjusts its free parameters (W, b) to approach the correct answer (i.e., minimize the error).
#A big advantage of this kind of mapping is that there's no need to remember the training set: once the parameters are learned, all one needs is the configuration achieved.
#Besides, in image recognition, this function implies a matrix multiplication and addition, which makes it way more efficient that, for instance, kNN, that has to memorize every item in the training set AND compare each x with every single one of them.

#As for the loss function, in this case we'll use the Multiclass SVM (Support Vector Machine) loss, which for the i-th example can be defined as ∑j≠yi max(0,sj−syi+Δ). An alternative (L2-SVM) squares the max function. Utimately, the loss function quantifies our unhappiness with predictions.
#(Also, to avoid the ambiguity of multiple solutions to a single problem, we'll use a regularization penalty, L2, that discourages large weights.)
#A different approach is to use a cross-entropy loss function (Softmax) rather than the hinge loss of SVM. I'll make some notes on Software elsewhere.


def L_i(w, y, W):
    """
    Unvectorized version. Compute the multiclass SVM loss for a single example (x, y).
    - x is a column vector representing an image (e.g. 3037 x 1 in CIFAR-10) with an appended bias
    dimension in the 3073-rd position (i.e., bias trick).
    - y is an integer giving index of correct class (e.g., between 0 and 9 in CIFAR-10).
    - W is the weight matrix (e.g., 10 x 3017 in CIFAR-10).
    """
    delta = 1.0
    scores = W.dot(x) #Scores become of size 10 x 1, the scores for each class.
    correct_class_score = scores[y]
    D = W.shape[0] # Number of classes, e.g., 10.
    loss_i = 0.0
    for j in xrange(D): #Iterate over all wrong classes.
        if j == y:
            #Skip for the true class.
            continue
        #Accumulate loss for the i-th example.
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i

def L_i_vectorized(x, y, W):
    """
    A faster, half-vectorized implementation, meaning that for a single example the implementation contains
    no loops, but there is still one loop over th examples (outside this function).
    """
    delta = 1.0
    score = W.dot(x)
    #Compute the margins for all classes in one vector operation.
    margins = np.maximum(0, scores - scores[y] + delta)
    #On y-th position scores[y] - scores[y] canceled and gave delta. We want to ignore the y-th position
    #and only consider margin on max wrong class.
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def L(X, y, W):
    """
    Fully-vectorized implementation.
    - X holds all the training examples as columns (e.g., 3073 x 50,000 in CIFAR-10).
    - y is an array of integers specifying the correct class (e.g., 50,000-D array).
    - W are weights (e.g., 10 x 3073).
    """
    #Evaluate loss over all examples in X without using any for loops.

#Note that in every case, delta can be arbitrarily set to 1.0, since it only measures the tradeoff between the data loss and the regularization loss. 
#Hence, the exact value of this relation is rather meaningless: we only care about the mutual relation of the scores.