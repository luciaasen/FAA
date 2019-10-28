from Datos import Datos
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import OneHotEncoder

# Variables al principio para facilitar ejecucion
porcentajeValidacionSimple = 0.2
seed = 0
alphaNB = 1.0

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

# ahora la matriz esta traspuesta y debemos devolverla
matrixAux = np.array(matrixAux).T
clases = dataset.datos.T[-1:].ravel()


print("Original Matrix\n", dataset.datos)
# print("New Matrix\n", matrixAux[:-4][:3])
###############################################################################
###############################################################################
# ValidacionSimpleScikit
###############################################################################
###############################################################################
# Particion de los datos

train, test = train_test_split(matrixAux, test_size=porcentajeValidacionSimple, random_state=seed, shuffle=True)
print("Train: \n", train)
print("Test: \n", test)
print("Clases: \n", clases)

print("LEN MAT = ", len(matrixAux))
print("LEN clases = ", len(clases))

# utilizamos multinomial porque los valores son discretos
clf = MultinomialNB()
clf.fit(matrixAux, clases)
# con el conjunto de entrenamiento, obtenemos las predicciones??
pred = clf.predict(train)

# p
print(pred)
