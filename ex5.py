from sklearn import datasets 
import numpy as np
from sklearn import svm
import random
from matplotlib import pyplot as plt

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


def prediction(x,models,nb_classes):

	cl = np.zeros(nb_classes)
	for i in range(len(models)):
		cl[models[i].predict([x])] = cl[models[i].predict([x])] + 1

	return np.argmax(cl)


def prdict_test(x_test,models,nb_classes):

	y_test = []
	for i in range(len(x_test)):
		y_test.append(prediction(x_test[i],models,nb_classes))

	return y_test

#Perofrmance test  


# rapport entre le nombre de hits sur le totale 



# Test 

# import some data to play with
digit = datasets.load_breast_cancer()

train_set = 0.7
nb_classes = 3
x = digit.data
y = digit.target

x_train = x[len(x)-int(len(x)*train_set):]
y_train = y[len(y)-int(len(y)*train_set):]


x_test = x[:len(x)-int(len(x)*train_set)]
y_test = y[:len(y)-int(len(y)*train_set)]

#models = SVM_Enssemble(20,x_train,y_train,0.2)
#y_pred = prdict_test(x_test,models,2)



def performance(y_test,y_pred):
	cpt = 0
	for i in range(len(y_test)):

		if y_pred[i] == y_test[i]:
			cpt = cpt +1

	return (cpt/len(y_test))


#Incremental cardinality 

#ox  = []
#oy = []
#for i in range(2,20):
#	models = SVM_Enssemble(i,x_train,y_train,0.3)
#	ox.append(i)
#	y_pred = prdict_test(x_test,models,2)
#	oy.append(performance(y_test,y_pred))


#Same number of SVMs (20)
#ox  = []
#oy = []
#p = 0.1
#while p<0.7:
#	models = SVM_Enssemble(20,x_train,y_train,p)
#	ox.append(p)
#	y_pred = prdict_test(x_test,models,2)
#	oy.append(performance(y_test,y_pred))
#	p = p + 0.1





#plt.plot(ox, oy)
#plt.show()



# the commun point between SVM Ensemble and multi-class SVM is that both of them use a set of
# weak SVMs which are trained to construt a strong model given a better results 


#the dissrence is the way of training our models 
#SVM Ensemble use diffrent random part pf the train-set
# but the one_vs_on_model or one_vs_ all_model uses a diffrent type of classes to distingush for the traiing phase)
