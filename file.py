import csv
import random

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
    res = csvParser('trainingdata.csv')
    # This next part is the crossover fn
    for val in range(len(res)):
        if val != 0:
            crossed1, crossed2 = crossover(res[val-1], res[val])
            res[val-1] = crossed1
            res[val] = crossed2
    # This is where the mutation would go when we have it in the correct binary format
    # here I am going to run it through the AI and ML 
    test = csvParser('testingdata.csv')
    # here I am going to run it through the AI and test the ML
    
if __name__ == "__main__":
    main()