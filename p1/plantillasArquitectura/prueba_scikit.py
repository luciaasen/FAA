from Datos import Datos
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import OneHotEncoder

# Variables al principio para facilitar ejecucion
porcentajeValidacionSimple = 0.2
seed = 1
alphaNB = 1.0
folds = 4

print("***Parametros Utilizados***")
print("Porcentaje del conjunto entrenamiento ValidacionSimple: ", porcentajeValidacionSimple)
print("Semilla: ", seed)
print("Correccion de Laplace (alpha): ", alphaNB)
print("Numero de Folds para ValidacionCruzada ", folds)

# dataset=Datos('datos/tic-tac-toe.data')
# dataset=Datos('datos/conjunto_datos_lentillas.txt')
dataset=Datos('datos/german.data')

# Para que funcione con scikit hay que discretizar los atributos continuos
# utilizaremos KBinsDiscretizer
matrixAux = []
# iteramos la matriz por columnas excepto la columna de clases
for col, att in zip(dataset.datos.T[:-1], dataset.nominalAtributos[:-1]):
    # Si el atributo es discreto no hace falta hacer nada
    if att == True:
        # matrixAux.append(col.tolist())
        matrixAux.append(col)
    else:
        # hay que discretizar los valores continuos
        # asignamos el mismo numero de intervalos que clases distintas
        numBins = len(dataset.diccionarios[-1])
        enc = KBinsDiscretizer(n_bins=numBins, encode='ordinal', strategy='kmeans')
        # transformamos en un array de arrays (cada numero dentro de una lista individual)
        colReshaped = col.reshape(-1,1)
        colBinned = enc.fit_transform(colReshaped)
        # matrixAux.append(colBinned.ravel().tolist())
        matrixAux.append(colBinned.ravel())

# ahora la matriz esta traspuesta y debemos devolverla y anadir las clases
matrixAux.append(dataset.datos.T[-1:].ravel())
matrixAux = np.array(matrixAux)
matrixAux = matrixAux.T

# utilizamos clasificador multinomial porque los valores son discretos
# alpha --> correccion laplace
# fit_prior: si False utiliza distribucion uniforme
clf = MultinomialNB(alpha=alphaNB, fit_prior=False)

# print("Original Matrix\n", dataset.datos)
# print("MatrixAux\n", matrixAux)

###############################################################################
###############################################################################
# ValidacionSimpleScikit
###############################################################################
###############################################################################

# Particion de los datos
trainMat, testMat = train_test_split(matrixAux, test_size=porcentajeValidacionSimple, random_state=seed, shuffle=True)
train = (trainMat.T[:-1]).T
test = (testMat.T[:-1]).T
trainClasses = trainMat.T[-1:].ravel()
testClasses = testMat.T[-1:].ravel()

# print("Train: \n", train)
# print("TrainClasses: \n", trainClasses)
# print("Test: \n", test)
# print("TestClasses: \n", testClasses)


# Entrenamos con el conjunto de entrenamiento
clf.fit(train, trainClasses)
# Validamos con el conjunto de test
score = clf.score(test, testClasses)
print("\nValidacionSimple Score = " + str(score*100) + " %")


# ValidacionCruzadaScikit
###############################################################################
###############################################################################

# Particion de los datos
# foldsList = []
# n_splits es el numero de folds que queremos
# shuffle=True => shuffle datos antes de realizar division en folds
kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
# for trainK, testK in kf.split(matrixAux):
#     print("TRAIN:", trainK)
#     print("TEST:", testK)
#     foldsList.append((trainK.tolist(), testK.tolist()))

# clf is the classifier, in this case MultinomialNB
# (matrixAux.T[:-1]).T is the data matrix without the classes
# y=matrixAux.T[-1:].ravel() is the list of the classes
# cv is the object that creates the folds in the data
cvs = cross_val_score(clf, (matrixAux.T[:-1]).T, y=matrixAux.T[-1:].ravel(), cv=kf)
avgCvs = sum(cvs)/folds
# cvs = cross_val_score(clf, (matrixAux.T[:-1]).T, y=matrixAux.T[-1:].ravel(), cv=folds)
print("\nValidacionCruzada")
print("Scores of each fold: ", cvs)
print("Average score of Cross Validation: ", avgCvs)
