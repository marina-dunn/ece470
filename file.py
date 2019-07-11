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
    # What I want to do here is take in the two parents
    # use a random value 
    len1 = len(parent1)
    len2 = len(parent2)
    len1 = random.randrange(range(len1))
    print(len1)

def main():
    res = csvParser('trainingdata.csv')
    # here I am going to run it through the AI and ML 
    test = csvParser('testingdata.csv')
    # here I am going to run it through the AI and test the ML
    
if __name__ == "__main__":
    main()