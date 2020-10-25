from sklearn import datasets 
import numpy as np
from sklearn import svm
from itertools import combinations 



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

#create the combinations 
comb = combinations(range(nb_classes), 2) 
  
#Create the models associeted to the list of combinations 
for i in list(comb): 
    x_train = []
    y_train = []
    for k in range(len(y)):
    	if y[k] == i[0] :
    		x_train.append(x[k]) 
    		y_train.append(i[0])
    	elif y[k] == i[1]:
    		x_train.append(x[k]) 
    		y_train.append(i[1])


    #train the model 
	clf = svm.SVC(probability=True)
	clf.fit(x_train, y_train)

	#put the  classifier in a models array
	models.append(clf)