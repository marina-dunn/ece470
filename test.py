import file
import numpy
import fitness_wrapper
#import so we can use the parser!

csvParser = file.csvParser

traindata = numpy.asarray(csvParser('trainingdata.csv'), dtype=numpy.float32)
testdata = numpy.asarray(csvParser('testingdata.csv'), dtype=numpy.float32)
trainlabel = numpy.asarray(csvParser('traininglabels.csv'), dtype=numpy.float32)
testlabel = numpy.asarray(csvParser('testinglabels.csv'), dtype=numpy.float32)

#What are the results of the algorithm if all of the features are included?
start = numpy.matrix('1,1,1,1,1,1,1,1,1')

result = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, start)
print(result)
