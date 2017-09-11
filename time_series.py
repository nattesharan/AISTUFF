import pandas
import numpy
import matplotlib.pyplot as pt
numpy.random.seed(7)
dataframe = pandas.read_csv('international-airline-passengers.csv',usecols=[1],engine='python',skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset)

