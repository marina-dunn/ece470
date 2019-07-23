import csv
import random
import fitness_wrapper
import numpy

# this is an object that is used to associate the binary encodings and fitness values
class Chromosome:
    def __init__(self, binary, fit):
        self.binary = binary
        self.fit = fit

    def __repr__(self):
        return repr((self.binary, self.fit))

# this parses the csv files
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

# This function takes in two parents and exchanges random values within them to crossover
def crossover(parent1, parent2):
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

# In here we are taking in an individual and applying the flip bit mutation
def mutation(individual):
    rand = random.randrange(len(individual))
    val = individual[rand]
    if val == 0:
        individual[rand] = 1
    else:
        individual[rand] = 0
    return individual

# creates a randomized binary encoding
def makeRandom(n):
    temp = []
    for bina in range(n):
        temp.append(random.randint(0, 1))
    return temp

# checking if the binary encoding includes none of the features or only one
def checkZeros(binary):
    valid = 0
    for val in binary:
        if(val == 1):
            valid += 1
    if valid > 1:
        return binary
    else:
        return checkZeros(makeRandom(9))

def checkPlateau(fit, plateauarr):
    # checks for a plateau of 50 or more of the same fitness values
    if len(plateauarr) >= 50:
        return None
    # adds the newest fitness value to the plateau array if it's the same
    elif plateauarr[-1] == fit:
        plateauarr.append(fit)
    # empties the array and starts a new counter if there is a new fitness value that would not be condusive to a 
    elif plateauarr[-1] != fit:
        plateauarr = [fit]
    return plateauarr

def main():
    # parse all of the csv files to pass to the fitness function
    traindata = numpy.asarray(csvParser('trainingdata.csv'), dtype=numpy.float32)
    testdata = numpy.asarray(csvParser('testingdata.csv'), dtype=numpy.float32)
    trainlabel = numpy.asarray(csvParser('traininglabels.csv'), dtype=numpy.float32)
    testlabel = numpy.asarray(csvParser('testinglabels.csv'), dtype=numpy.float32)
    matingpool = []
    plateauarr = []

    # start out with 4 random individuals in mating pool
    # each individual is a binary encoding that decides which of the features is being included
    for q in range(4):
        temp = checkZeros(makeRandom(9))
        temp2 = []
        temp2.append(temp)
        individual = numpy.asarray(temp2, dtype=numpy.float32)
        res = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        plateauarr.append(res)
        tempchrom = Chromosome(temp2, res)
        matingpool.append(tempchrom)

    for j in range(510): # hard stop by iteration as back up for plateau
        listtemp = []
        # sort the mating pool and pick the two highest
        par1 = matingpool[0].binary[0]
        par2 = matingpool[1].binary[0]
        # run the parents through the GA to get two children
        chi1, chi2 = crossover(par1, par2)
        chi1 = checkZeros(mutation(chi1))
        listtemp.append(chi1)
        individual = numpy.asarray(listtemp, dtype=numpy.float32)
        # get fitness of first child
        res1 = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        # check plateau after first child
        plateauarr = checkPlateau(res1, plateauarr)
        if plateauarr is None:
            break
        ch1 = Chromosome(listtemp, res1)
        listtemp = []
        chi2 = checkZeros(mutation(chi2))
        listtemp.append(chi2)
        individual = numpy.asarray(listtemp, dtype=numpy.float32)
        # get fitness of second child
        res2 = fitness_wrapper.fitness_wrapper(traindata, trainlabel, testdata, testlabel, individual)
        # check plateau after second child
        plateauarr = checkPlateau(res2, plateauarr)
        if plateauarr is None:
            break
        ch2 = Chromosome(listtemp, res2)
        # add children to mating pool - sort mating pool and remove two worst performing individuals
        matingpool.append(ch1)
        matingpool.append(ch2)
        matingpool.sort(key=lambda x: x.fit)
        matingpool.pop()
        matingpool.pop()

    # prints out the results
    print('The binary encoding is: ')
    print(matingpool[0].binary[0])
    print(' , the fitness result is: ')
    print(matingpool[0].fit)
    print(' and the iteration was: ')
    print(j)

if __name__ == "__main__":
    main()
