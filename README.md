# ECE 470: AI Project
This is a project programming an Artificial Intelligence using Machine Learning to create a model for our data.

We are using data regarding [echocardiograms](http://archive.ics.uci.edu/ml/datasets/Echocardiogram).

## Machine Learning
The machine learning aspects includes linear regression and the fitness function to evaluate each individual.

## Genetic Algorithm
The GA for this project includes a crossover and mutation from the parents to produce two offspring.

The mating pool consists of the valid possibilities from the population and the parents are selected as the two individuals with the lowest fitness values.

The crossover is a simple switch between two of the features.

The mutation is a simple bit flip of a randomly selected feature.

# Running the code
The code is created using python 3. 

The one included package that is not python standard is the numpy module installation found [here](https://pypi.org/project/numpy/).

The command to run the code is:
`python3 file.py`
