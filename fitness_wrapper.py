import numpy
# importing local files:
import fitness
import regression

#takes datasets formatted having each column as a data point, each row as a feature
def fitness_wrapper(oTrainingData, oTrainingLabels, oTestingData, testingLabels, individual):
	# transposing the data to have the samples become columns and the features become rows
	trainingData = numpy.matrix.transpose(oTrainingData)
	trainingLabels = oTrainingLabels
	testingData  = numpy.matrix.transpose(oTestingData)

	#design decision - make use range of length for future flexibility
	fData = None
	tData = None

	# this cycles through the given individual(binary encoding)
	# and only includes the features that are designated by the individual
	for gene_position in range(numpy.size(individual,1)):
		if individual[0,gene_position] == 1:
			if fData is None:
				fData = trainingData[gene_position]
				tData = testingData[gene_position]
			else:
				fData = numpy.vstack((fData, trainingData[gene_position]))
				tData = numpy.vstack((tData, testingData[gene_position]))

	#call regression on new data to get the optimal classification values
	w,b = regression.regression(fData, trainingLabels)

	#call fitness function to evaluate optimal classification values
	fitness_result = fitness.fitness(tData, w, b,testingLabels)
	return fitness_result
