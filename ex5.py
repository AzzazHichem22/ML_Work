from sklearn import datasets 
import numpy as np
from sklearn import svm
import random


def SVM_Enssemble(nb_classifier,x_train,y_train,percentage):
	models = []
	for i in range(nb_classifier):
		#Random sub set
		r = random.sample(range(0,len(x_train)-1), int(len(x_train)*percentage)) 
		x_sub = []
		y_sub = []
		for k in range(len(x_train)):
			if k in r:
				x_sub.append(x_train[k])
				y_sub.append(y_train[k])

		#train the model 
		clf = svm.SVC(probability=True)
		clf.fit(x_sub, y_sub)

		#put the  classifier in a models array
		models.append(clf)


	return models


#Perofrmance test  


# rapport entre le nombre de hits sur le totale 


det Perofrmance_model(x_test,y_test,model):
	percentage = 0
	return percentage


# Test 

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

models = SVM_Enssemble(3,x_train,y_train,0.3)