import Datos as d
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import warnings

warnings.filterwarnings("ignore")
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

def logReg(fileName, epocas = [1,2,4,6,8] , ctesApr = [(i+1)/5 for i in range(5)]  , nRepeticionesCruzada = 5, nRepeticionesSimple = 20):

    porcentajes = [0.25,0.30]
    ks = [5,10,15]

    matrixAux = discretiza(fileName)
    print('\n', fileName, '\nSGD')
    # Validacion simple
    seed = 1
    for porcentaje in porcentajes:
        for cteApr in ctesApr:
            for nEpocas in epocas:
                errors = []
                clf = SGDClassifier( max_iter = nEpocas, learning_rate= "constant", eta0 = cteApr)
                for i in range(nRepeticionesSimple):
                    trainMat, testMat = train_test_split(matrixAux, test_size=porcentaje, random_state=seed, shuffle=True)
                    train = (trainMat.T[:-1]).T
                    test = (testMat.T[:-1]).T
                    trainClasses = trainMat.T[-1:].ravel()
                    testClasses = testMat.T[-1:].ravel()
                    
                    clf.fit(train, trainClasses)
                    clf.score(test, testClasses)
                    errors.append(1-clf.score(test,testClasses))
                    seed += 1
                errorsnp = np.array(errors)
                mean, std = np.mean(errors), np.std(errors)
                print('\nVSimple,', porcentaje,'%, CteApr ', cteApr, 'Epocas ', nEpocas, '\nError medio:', mean, '\nDesviacion tipica:', std)
    for k in ks:
        for cteApr in ctesApr:
            for nEpocas in epocas:
                errors = []
                clf = SGDClassifier( max_iter = nEpocas, learning_rate= "constant", eta0 = cteApr)
                for i in range(nRepeticionesCruzada):  
                    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
                    errors.append(1-np.mean(np.array(cross_val_score(clf, (matrixAux.T[:-1]).T, y=matrixAux.T[-1:].ravel(), cv=kf))))
                    seed += 1
                errorsnp = np.array(errors)
                mean, std = np.mean(errors), np.std(errors)
                print('\nVCruzada K =,', k, 'CteApr ', cteApr, 'Epocas ', nEpocas, '\nError medio:', mean, '\nDesviacion tipica:', std)




