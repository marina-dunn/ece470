import numpy

import fitness
import regression

#takes datasets formatted as each column is a data point, each row is a feature
def fitness_wrapper(trainingData, trainingLabels, testingData, testingLabels, individual):
		
	#design decision - make use range of length for future flexibility
	fData = None
	tData = None
	#print(numpy.size(individual,1))
	
	for gene_position in range(numpy.size(individual,1)):
		#print(gene_position)
		if individual[0,gene_position] == 1:
			if fData == None:
				fData = trainingData[gene_position]
				tData = testingData[gene_position]
			else:
				fData = numpy.vstack((fData, trainingData[gene_position]))
				tData = numpy.vstack((tData, testingData[gene_position]))
		
	print("fData:")
	print(fData)
	print("tData:")
	print(tData)
	
	#call regression on new data
	w,b = regression.regression(fData, trainingLabels)
	
	#call fitness function on w and b values
	fitness_result = fitness.fitness(tData, w, b,testingLabels)
	
	
A = numpy.matrix('1,1,4;2,3,5;3,42,6')
#print(numpy.size(A,1))
B = numpy.matrix('1,1,-1')
C = numpy.matrix('8,8,8;7,7,7;6,6,6')
D = numpy.matrix("1,1,1")
individual = numpy.matrix('1,1,0')
A = numpy.matrix.transpose(A)
B = numpy.matrix.transpose(B)
C = numpy.matrix.transpose(C)
#D = numpy.matrix.transpose(D)
fitness_wrapper(A,B,C,D,individual)
