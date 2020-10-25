
from sklearn import datasets 
import numpy as np
from sklearn import svm

# import some data to play with
iris = datasets.load_iris()

train_set = 0.7
nb_classes = 3
x = iris.data
y = iris.target

x_train = x[len(x)-int(len(x)*train_set):]
y_train = y[len(y)-int(len(y)*train_set):]


x_test = x[:len(x)-int(len(x)*train_set)]
y_test = y[:len(y)-int(len(y)*train_set)]


#file to store classifiers 

models = []
#create our classifiers 

for i in range(3):
	#make  classe "i"  positive and the other classes negative in the trainig set 
	y_binary = y_train.copy()
	for k in range(len(x_train)):
		if y_train[k] == i : 
			y_binary[k] = 1
		else:
			y_binary[k] = 0

	#train the model 
	clf = svm.SVC(probability=True)
	clf.fit(x_train, y_binary)

	#put the  classifier in a models array

	models.append(clf)



#use the clissifiers in order to pedict the classes
value = []


for i in range(3):
	value.append(models[i].predict_proba([x_test[0]])[0][1])


print(np.argmax(value))



	

			


