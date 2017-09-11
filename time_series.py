import pandas
import matplotlib.pyplot as pt
import numpy
numpy.random.seed(7)
dataset = pandas.read_csv('international-airline-passengers.csv',usecols=[1],engine='python',skipfooter=3)
pt.plot(dataset)
pt.show()
