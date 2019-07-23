import numpy

# given a dataset F with n features and m data points and labels y, compute optimal linear regression values
# output w* (F values) b (single value)

def regression (trainingSet, trainingLabels):
    # each column should be one data point
    n,m = trainingSet.shape
    onesRow = numpy.ones((1,m))
    Xt = numpy.vstack((trainingSet, onesRow))

    # transpose Fnew to get X
    X = numpy.matrix.transpose(Xt)

    # use matrix multiplication
    XtX = Xt @ X
    inv = numpy.linalg.inv(XtX)

    # THE GOLD STANDARD FOR MATRIX MULTIPLICATION
    Xty = Xt @ trainingLabels

    # retrieve the optimal classification values
    W = inv @ Xty
    w = W[0:n]
    b = W[n]
    return w, b
