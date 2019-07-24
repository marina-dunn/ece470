import numpy

# given vector of optimal w values and single b value, classify testing set using provided F features
def fitness(testingData, w, b, testingLabels):
    # get the dimensions of the matrix
    n,m = w.shape

    # for each datapoint in testing set (each column, IGNORING the label)
    Ytest = (numpy.matrix.transpose(w) @ testingData) +b
    testingLabels = numpy.matrix.transpose(testingLabels)
    # based on closeness of the two classes (0 or 1) sort
    for i in range(numpy.size(Ytest,1)):
        if Ytest[0,i] >= 0:
            Ytest[0,i] = 1
        else:
            Ytest[0,i] = -1

    # misclassifications - result of fitness function plus number of features needed
    result = (testingLabels - Ytest)
    error = numpy.count_nonzero(result)
    error = error + 0.01*n
    return error
