# import numpy
import csv

with open('echocardiogram.csv')as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(', '.join(row))