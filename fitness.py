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
    # print('below is w :::::::::::::')
    # print(w)
    Ytest = numpy.matrix.transpose(w)*testingData +b
    #based on closeness of the two classes (0 or 1) sort
    # print(Ytest)
    # print("testing labels")
    # print(testingLabels)

    for i in range(numpy.size(Ytest,1)):
        # print(i)
        if Ytest[0,i] >= 0:
            Ytest[0,i] = 1
        else:
            Ytest[0,i] = -1

    # print (Ytest)

    #misclassifications - result of fitness function
    result = testingLabels - Ytest

    # print(result)
    error = 0
    for element in range(numpy.size(result,1)):
        if element !=0:
            error = error +1
        
    return error
