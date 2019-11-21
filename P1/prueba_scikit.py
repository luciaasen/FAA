from plantillasArquitectura import Datos as d
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB


def discretiza(fileName):
    dataset=d.Datos(fileName)
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
    return matrixAux

def pruebaScikit():
    nRepeticiones = 25
    dicc = dict()
    dicc['LENTILLAS'] = 'plantillasArquitectura/datos/conjunto_datos_lentillas.txt'
    dicc['TIC-TAC-TOE'] = 'plantillasArquitectura/datos/tic-tac-toe.data'
    dicc['GERMAN'] = 'plantillasArquitectura/datos/german.data'
    for fileName in dicc:
        matrixAux = discretiza(dicc[fileName])
        print('\n', fileName)
        seed = 2
        for j in range(2,4):
            porcentaje = j/10
            alphaNB = 1.0
            for i in range(2):
                scores = []
                for k in range(nRepeticiones):
                    clf = MultinomialNB(alpha=alphaNB, fit_prior=False)
                    trainMat, testMat = train_test_split(matrixAux, test_size=porcentaje, random_state=seed, shuffle=True)
                    train = (trainMat.T[:-1]).T
                    test = (testMat.T[:-1]).T
                    trainClasses = trainMat.T[-1:].ravel()
                    testClasses = testMat.T[-1:].ravel()
                    clf.fit(train, trainClasses)
                    scores.append(1-clf.score(test, testClasses))
                    seed += 1
                scoresnp = np.array(scores)
                mean, std = np.mean(scoresnp), np.std(scoresnp)
                print('\nv. simple,',porcentaje*100,'%, Laplace a ',alphaNB,'\nError medio:', mean, '\nDesviacion tipica:', std)
                alphaNB = 1e-10

        seed = 2
        for i in range(5,25,5):
            alphaNB = 1.0
            for j in range(2):
                clf = MultinomialNB(alpha=alphaNB, fit_prior=False)
                scores = []
                for k in range(nRepeticiones):
                    # cv is the object that creates the folds in the data
                    kf = KFold(n_splits=i, random_state=seed, shuffle=True)
                    scores.append(1-np.mean(np.array(cross_val_score(clf, (matrixAux.T[:-1]).T, y=matrixAux.T[-1:].ravel(), cv=kf))))
                    seed += 1
                cvs = np.array(scores)
                mean, std = np.mean(cvs), np.std(cvs)
                print('\nv.  cruzada, K =',i,', Laplace a ',alphaNB,'\nError medio:', mean, '\nDesviacion tipica:', std)
                alphaNB = 1e-10


## utilizamos clasificador multinomial porque los valores son discretos
## alpha --> correccion laplace
## fit_prior: si False utiliza distribucion uniforme
#clf = MultinomialNB(alpha=alphaNB, fit_prior=False)
#
################################################################################
## ValidacionSimpleScikit
################################################################################
## Particion de los datos
#trainMat, testMat = train_test_split(matrixAux, test_size=porcentajeValidacionSimple, random_state=seed, shuffle=True)
#train = (trainMat.T[:-1]).T
#test = (testMat.T[:-1]).T
#trainClasses = trainMat.T[-1:].ravel()
#testClasses = testMat.T[-1:].ravel()
#
## Entrenamos con el conjunto de entrenamiento
#clf.fit(train, trainClasses)
## Validamos con el conjunto de test
#score = clf.score(test, testClasses)
#
################################################################################
## ValidacionCruzadaScikit
################################################################################
## Particion de los datos
## n_splits es el numero de folds que queremos
## shuffle=True => shuffle datos antes de realizar division en folds
#kf = KFold(n_splits=folds, random_state=seed, shuffle=True)
## clf is the classifier, in this case MultinomialNB
## (matrixAux.T[:-1]).T is the data matrix without the classes
## y=matrixAux.T[-1:].ravel() is the list of the classes
## cv is the object that creates the folds in the data
#cvs = cross_val_score(clf, (matrixAux.T[:-1]).T, y=matrixAux.T[-1:].ravel(), cv=kf)
#avgCvs = sum(cvs)/folds
