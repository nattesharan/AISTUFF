import pandas
import numpy
import matplotlib.pyplot as pt
import math
from keras.models import Sequential
from keras.layers import Dense
numpy.random.seed(7)
dataframe = pandas.read_csv('international-airline-passengers.csv',usecols=[1],engine='python',skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# split into tran and test dataset
train_size = int(len(dataset)*0.66)
test_size = len(dataset) - train_size
train,test = dataset[:train_size,:],dataset[train_size:,:]
'''
converts the data with single column to t --> t+1 series
X		Y
112		118
118		132
132		129
129		121
121		135
'''
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),0])
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX),numpy.array(dataY)
look_back = 1
trainX,trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)
model = Sequential()
model.add(Dense(8,input_dim=1,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=200,batch_size=2,verbose=2)
# estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
