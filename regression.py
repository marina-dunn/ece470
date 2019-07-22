import numpy

#given a dataset F with n features and m data points and labels y, compute optimal linear regression values
#output w* (F values) b (single value)

def regression (trainingSet, trainingLabels):
    #each column should be one data point
    #F = numpy.matrix('-2, -1, 0, 1, 2; 2, 3, 4, 2, 1')
    #y = numpy.matrix('-1; 0; 2; 2; 4')

    print("trainginnset")
    print(trainingSet)
    # print("traing labels")
    # print(trainingLabels)

    n,m = trainingSet.shape
    print(n)
    onesRow = numpy.ones((1,m))

    #print(onesRow)

    Xt = numpy.vstack((trainingSet, onesRow))

    #print(Fnew)

    #transpose Fnew to get X

    X = numpy.matrix.transpose(Xt)

    XtX = numpy.matmul(Xt, X)

    # print("Xt*X")
    # print(XtX)

    inv = numpy.linalg.inv(XtX)

    Xty = Xt*trainingLabels
    # print("Xt*y")
    # print(Xty)

    W = numpy.matmul(inv, Xty)

    #debug purposes
    print('W')
    print(W)

    w = W[0:n]
    
    b = W[n]
    #print('w* and b*')
    #print(w)
    #print(b)
    return w, b
