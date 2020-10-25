
from sklearn import datasets 
import numpy as np
from sklearn import svm
	
def one_vs_all_models(x_train,y_train,nb_classes):
	#file to store classifiers 

	models = []
	#create our classifiers 

	for i in range(nb_classes):
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

	return models



#This function make the classification 
#The in put is the x 
#The out put is a the number of the classe which x belongs to 


def predict(x):
	#use the clissifiers in order to pedict the classes
	values = []

	for i in range(nb_classes):
		values.append(models[i].predict_proba([x])[0][1])


	return np.argmax(values)



# this is an example using Iris data set 

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


models = one_vs_all_models(x_train,y_train,nb_classes)
print(predict(x_train[80]))