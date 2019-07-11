import numpy

#given vector of optimal w values and single b value, classify testing set using provided F features
def fitness(testingData, w, b, testingLabels):
    #temporary for testing purposes
    #w = numpy.matrix('1.2; 1')
    #b = 1.4
    #Y = numpy.matrix('1, -1, -1')
    #n,m = w.shape
    #T = numpy.matrix('-0.5, 3, 4; 1,2,3')

    #for each datapoint in testing set (each column, IGNORING the label)
    #wt = numpy.matrix.transpose(w)

    Ytest = numpy.matrix.transpose(w)*testingData + b
    #based on closeness of the two classes (0 or 1) sort
    print(Ytest)

    for i in range(len(Ytest)):
        print(i)
        if Ytest[i] >= 0:
            Ytest[i] = 1
        else:
            Ytest[i] = -1

    print (Ytest)

    #misclassifications - result of fitness function
    result = testingLabels - Ytest

    print(result)
    return result
