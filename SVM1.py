from sklearn import datasets 
import numpy as np
from sklearn import svm
from itertools import combinations 






def one_vs_one_models(x_train,y_train,nb_classes):
    #file to store classifiers 

    #create the combinations 
    comb = combinations(range(nb_classes), 2) 
    models = []
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
    return models



def  predict(x,nb_classes,models):
    k = 0
    scores = np.zeros(int(nb_classes*(nb_classes-1)/2))
    comb = combinations(range(nb_classes), 2) 
    for el in list(comb):
        if models[k].predict([x]) == el[0]:
            scores[el[0]] = scores[el[0]] + 1
        else:
            scores[el[1]] = scores[el[1]] + 1

        k = k + 1


    return np.argmax(scores)



# import some data to play with
iris = datasets.load_iris()

train_set = 0.7

x = iris.data
y = iris.target
nb_classes = 3
x_train = x[len(x)-int(len(x)*train_set):]
y_train = y[len(y)-int(len(y)*train_set):]


x_test = x[:len(x)-int(len(x)*train_set)]
y_test = y[:len(y)-int(len(y)*train_set)]


models = one_vs_one_models(x_test,y_train,nb_classes)
print(predict(x_test[0],nb_classes,models))