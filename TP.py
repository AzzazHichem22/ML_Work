import numpy as np
from matplotlib import pyplot as plt
import random
import math

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#data = zip(X, y)
#points_aleatoire = np.random.choices(data, 10)
#print(points_aleatoire)
#plt.scatter(X, y)
#plt.show()

def model(x,theta):
	return x.dot(theta)



def RANSAC(x, y,taille, n, k, t, d):
	iterator = 0
	meilleur_modele = None
	meilleur_ensemble = None
	meilleur_ensemble_points = None
	meilleur_erreur = 1000

	while iterator < k:
		# Definir l'enssemble de points aléatoires 
		points_aleatoires = random.sample(range(0,taille-1), 20)
		x_alea = np.zeros(shape=(n,1))
		y_alea = np.zeros(shape=(n,1))
		
		k = 0
		for i in points_aleatoires:
			x_alea[k] = x[i]
			y_alea[k] = y[i]
			k = k + 1
		
		#Parametres du modèle
		
		X = np.hstack((x_alea,np.ones(x_alea.shape)))
		reg = LinearRegression().fit(X,y_alea)
		modele_possible = reg.coef_
		ensemble_points = points_aleatoires

		#Construction de l'enssemble de points 

		#Le reste des points de notre dataset
		x_reste  =  np.zeros(shape=(taille-len(points_aleatoires),1))
		y_reste =  np.zeros(shape=(taille-len(points_aleatoires),1))
		point_rest = np.zeros(shape=(taille-len(points_aleatoires),1))
		k = 0
		for i in range(taille): 
			if i not in points_aleatoires:
				x_reste[k] = x[i]
				y_reste[k] = y[i]
				point_rest[k] = i
				k = k +1



		X_reste = np.hstack((x_reste,np.ones(x_reste.shape)))
		y_pred = reg.predict(X_reste)

		# Rajouter les points qui repondent au critère à l'enssemble de points 

		for  point in range(taille-n):				
			
			if (math.sqrt((y_reste[point] - y_pred[point] )**2)) < t:
					ensemble_points.append(point)


		

		# la cardinalité de l'ensemble de poits est superieure à d 
		if len(ensemble_points) > d:

			x_points   =  np.zeros(shape=(len(ensemble_points) ,1))
			y_points =  np.zeros(shape=(len(ensemble_points) ,1))

			k = 0
			for i in ensemble_points :
				x_points[k] = x[i] 
				y_points[k] = y[i]
				k  = k + 1 


			X_points= np.hstack((x_points,np.ones(x_points.shape)))

			x_test = X_points[:len(X_points)-int(len(X_points)*0.7)]
			y_test = y_points[:len(X_points)-int(len(y_points)*0.7)]

			x_train =X_points[len(X_points)-int(len(X_points)*0.7):]
			y_train = y_points[len(X_points)-int(len(y_points)*0.7):]
			

			
			reg = LinearRegression().fit(x_train,y_train)

			y_predict = reg.predict(x_test)

			modele_possible = reg.coef_
			erreur = mean_squared_error(y_test, y_predict)

			print(erreur)
			if erreur < meilleur_erreur:
				meilleur_modele = modele_possible
				meilleur_ensemble_points = ensemble_points
				meilleur_erreur = erreur
		
		iterator = iterator + 1


	return meilleur_modele, meilleur_ensemble_points, meilleur_erreur





n_samples = 100
n_outliers = 15


x, y = make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10)


# Add outlier data
np.random.seed(0)

x = np.append(x,3 + 0.5 * np.random.normal(size=(n_outliers, 1)))

y = np.append(y,-3 + 0.5 * np.random.normal(size=n_outliers ))

x = np.append(x,-3 + 0.5 * np.random.normal(size=(n_outliers, 1)))

y = np.append(y,3 + 0.5 * np.random.normal(size=n_outliers ))


x = x.reshape(x.shape[0],1)
y = y.reshape(y.shape[0],1)


#X = np.hstack((x,np.ones(x.shape)))


#reg = LinearRegression().fit(X, y)


#print(reg.predict(X)[0])

#theta  = np.random.randn(2,1)











#reg = LinearRegression().fit(X, y)
#modele_possible = reg.get_params()
#Y_predict  = reg.predict(X)


taille =2*n_outliers + n_samples
n = 40
k = 40
t = 5
d = 50

meilleur_modele, meilleur_ensemble_points, meilleur_erreur = RANSAC(x, y,taille, n, k, t, d)

#print("meilleur_modele is :")
#print(meilleur_modele)
#print("meilleur_ensemble_points :")
#print(meilleur_ensemble_points)
#print("meilleur_erreur :")
#print(meilleur_erreur)
#plt.scatter(X, y)
#plt.show()

#plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
 #           label='Inliers')
#plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
 #           label='Outliers')
#plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
#plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
 #        label='RANSAC regressor')
#plt.legend(loc='lower right')
#plt.xlabel("Input")
#plt.ylabel("Response")
#plt.show()


