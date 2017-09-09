import numpy
from keras.models import Sequential
from keras.layers import Dense,Activation
# fix random seed for reproducibility
numpy.random.seed(7)
'''
Data set attibutes
 1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
'''
dataset = numpy.loadtxt('pima-indians-diabetes.data.csv',delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# fit the model
model.fit(X,Y,epochs=150,batch_size=10)
# evaluate the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
