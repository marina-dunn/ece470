import csv
import random
import fitness_wrapper
import numpy


class Chromosome:
    def __init__(self, binary, fit):
        self.binary = binary
        self.fit = fit

def csvParser(filename):
    with open(filename)as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        vals = []
        for row in csv_reader:
            temp = []
            if row[0] != 'survival':
                for val in row:
                    if val == '?': # Temporarily setting ? values to zero
                        val = 0
                    c = float(val)
                    temp.append(c)
                vals.append(temp)
        return vals

def crossover(parent1, parent2):
    # TODO: make this more efficient (less variables)
    # This function takes in two parents and exchanges random values within them
    len1 = len(parent1)
    len2 = len(parent2)
    rand1 = random.randrange(len1)
    rand2 = random.randrange(len2)
    cross1 = parent1
    cross2 = parent2
    temp = cross1[rand1]
    cross1[rand1] = cross2[rand2]
    cross2[rand2] = temp
    return cross1, cross2

def mutation(individual):
    # In here we are taking in an individual and applying the flip bit mutation
    rand = random.randrange(len(individual))
    val = individual[rand]
    if val == 0:
        individual[rand] = 1
    else:
        individual[rand] = 0
    return individual 

def main():
    traindata = numpy.asarray(csvParser('trainingdata.csv'), dtype=numpy.float32)
    testdata = numpy.asarray(csvParser('testingdata.csv'), dtype=numpy.float32)
    trainlabel = numpy.asarray(csvParser('traininglabels.csv'), dtype=numpy.float32)
    testlabel = numpy.asarray(csvParser('testinglabels.csv'), dtype=numpy.float32)
    # print(traindata)
    matingpool = []
    for q in range(4):
        temp = []
        temp2 = []
        for bina in range(9):
            temp.append(random.randint(0, 1))
        temp2.append(temp)
        individual = numpy.asarray(temp2, dtype=numpy.float32)
        res = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        tempchrom = Chromosome(temp2, res)
        matingpool.append(tempchrom)
    
    for j in range(510): # hard stop by iteration as back up for plateau
        listtemp = []
        # sort the mating pool and pick the two highest
        par1 = matingpool[0][0]
        par2 = matingpool[1][0]
        # run the parents through the GA to get the children
        chi1, chi2 = crossover(par1, par2)
        chi1 = mutation(chi1)
        listtemp.append(chi1)
        individual = numpy.asarray(listtemp, dtype=numpy.float32)
        res1 = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        listtemp = []
        chi2 = mutation(chi2)
        listtemp.append(chi2)
        individual = numpy.asarray(listtemp, dtype=numpy.float32)
        res2 = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        ch1 = Chromosome()
            
if __name__ == "__main__":
    main()