import numpy

#given vector of optimal w values and single b value, classify testing set using provided F features
def fitness(testingData, w, b, testingLabels):
    #temporary for testing purposes
    #w = numpy.matrix('1.2; 1')
    #b = 1.4
    #Y = numpy.matrix('1, -1, -1')
    n,m = w.shape
    #T = numpy.matrix('-0.5, 3, 4; 1,2,3')

    #for each datapoint in testing set (each column, IGNORING the label)
    #wt = numpy.matrix.transpose(w)
    # print('below is w :::::::::::::')
    # print(w)
    Ytest = (numpy.matrix.transpose(w) @ testingData) +b
    testingLabels = numpy.matrix.transpose(testingLabels)
    #based on closeness of the two classes (0 or 1) sort
    #print("testing labels")
    #print(testingLabels)
    #print('Ytest before, after')
    #print(Ytest)

    for i in range(numpy.size(Ytest,1)):
        # print(i)
        if Ytest[0,i] >= 0:
            Ytest[0,i] = 1
        else:
            Ytest[0,i] = -1

    #print (Ytest)
    #misclassifications - result of fitness function plus number of features needed
    result = (testingLabels - Ytest)
    #print(result)
    error = numpy.count_nonzero(result)
    error = error + n
    #print(error)

    return error
