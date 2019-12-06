from abc import ABCMeta,abstractmethod
import math
import numpy as np
from scipy.stats import norm
import Cromosoma as cr
import random as r
from copy import deepcopy
class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # Entendemos que 'datos' y 'pred' son de la misma longitud para poder
    # realizar esta funcion sin controles
    # Asumiendo que tenemos dos clases (que tendran valores 0 y 1 despues de
    # haber sido 'traducidas' por el diccionario), damos al usuario la opcion
    # de solicitar la matriz de confusion para el analisis ROC
    # Denotamos a la clase 0 como negative, y 1 como positive
    def error(self,datos,pred, ROC = False):
        numErr = 0
        numOk = 0
        rowLen = len(datos[0])
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        # Suponiendo que pred es una lista con las predicciones, y que por defecto
        # será de la misma longitud que el numero de filas de 'datos'
        # comparamos cada entrada de 'pred' con la ultima entrada de cada fila
        # de 'datos' (que es la clase real)
        for i in range(0,len(pred)):
            if pred[i] == datos[i][rowLen-1]:
                if pred[i]==0 and ROC: # True Negative
                    TN += 1
                elif ROC: # True Positive
                    TP += 1
                numOk = numOk + 1
            else:
                if pred[i] == 0 and ROC: # False Negative
                    FN +=1
                elif ROC: # False positive
                    FP +=1
                numErr = numErr + 1

        confMatrix = np.array([[TP, FP],[FN, TN]])
        # devuelve un numero entre 0,1 que representa el error
        error = numErr/(numErr + numOk)
        return (error, confMatrix)



    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self,particionado,dataset,clasificador,seed=None, ROC = False):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        listaParticiones = particionado.creaParticiones(dataset.datos, seed)
        # Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # Inicializamos error a 0
        error = 0.0
        mConf = np.array([[0,0],[0,0]])
        for particion in listaParticiones:
            # Extraemos datos para el entrenamiento, y los de test
            trainData = dataset.extraeDatos(particion.indicesTrain)
            testData = dataset.extraeDatos(particion.indicesTest)
            # Entrenamos datos (es decir, generamos tablas de Naive Bayes)
            clasificador.entrenamiento(trainData, dataset.nominalAtributos, dataset.diccionarios)
            # Clasificamos usando las tablas que ya han sido asignadas
            pred = clasificador.clasifica(testData, dataset.nominalAtributos, dataset.diccionarios)
            # Sumamos el error de esta iteracion al error total
            err, mConfusion = self.error(testData, pred, ROC = True)
            error += err
            if ROC:
                mConf += mConfusion
        # Calculamos error medio a lo largo de todas las particiones.
        # En el caso de validacion simple, esto sera solo una particion
        error /= len(listaParticiones)

        if not ROC:
            return error
        else:
            mConf = mConf/len(listaParticiones)
            return error, mConf

    def validacionROC(self,particionado,dataset,clasificador,seed=None, alpha = 0.5):
        listaParticiones = particionado.creaParticiones(dataset.datos, seed)
        # Inicializamos error a 0
        error = 0.0
        mConf = np.array([[0,0],[0,0]])
        for particion in listaParticiones:
            # Extraemos datos para el entrenamiento, y los de test
            trainData = dataset.extraeDatos(particion.indicesTrain)
            testData = dataset.extraeDatos(particion.indicesTest)
            # Entrenamos datos (es decir, generamos tablas de Naive Bayes)
            clasificador.entrenamiento(trainData, dataset.nominalAtributos, dataset.diccionarios, self.laplace)
            # Clasificamos usando las tablas que ya han sido asignadas
            pred = clasificador.clasificaROC(testData, dataset.nominalAtributos, dataset.diccionarios, alpha)
            # Sumamos el error de esta iteracion al error total
            err, mConfusion = self.error(testData, pred, ROC = True)
            error += err
            mConf += mConfusion
        # Calculamos error medio a lo largo de todas las particiones.
        # En el caso de validacion simple, esto sera solo una particion
        error /= len(listaParticiones)
        mConf = mConf/len(listaParticiones)
        return error, mConf


##############################################################################
##############################################################################

##############################################################################

class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, laplace = False):
        self.laplace = laplace

    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        # There are len - 1 attributes, since last element in atributosDiscretos, the class, does not correspond to a proper attribute
        nAtributos = len(atributosDiscretos)-1
        # We extract a set with all classes in the data
        classes = diccionario[-1].values()
        # The following loop will build the NB tables
        # We create a list where the i-th element of the list is associated to the i-th attribute,
        # and for each attribute a dictionary with classes as keys is created
        # If the attribute is continuous, then for each class key the appropriate normal distribution is assigned as a value
        # Otherwise, for each class key a new dictionary is created, which has the attribute values as keys,
        #            and for each value key,the number of data elements with the current class and current value is assigned as value


        self.prioris = prioris(datosTrain)
        attrTables = []
        for i in range(nAtributos):
                # For each attribute, a dictionary with all classes as values
                classesTable = dict()

                laplaceNeedsToBeApplied = False
                for clase in classes:
                    # We extract a set with all attribute values for the i-th attribute
                    attrValues = diccionario[i].values()
                    # If i-th attribute discrete:
                    # For each class, a dictionary with the i-th attribute values as keys
                    classesTable[clase] = dict()
                    if(atributosDiscretos[i]):
                        for value in attrValues:
                            # We count the number of elements of class clase out of the elements
                            # where the i-th attribute has value value
                            count = np.sum((datosTrain[:,i]==value ) & (datosTrain[:,-1]==clase))
                            if count == 0 and self.laplace == True:
                                laplaceNeedsToBeApplied = True
                                #print('Laplace correction will be applied because no data with class ', clase, ' and ', i, 'th attribute with value ', value, ' was encountered in ', datosTrain)
                            classesTable[clase][value] = count

                    # If i-th attribute continuous:
                    else:
                        # We create an array with the i-th attribute values of data where class == clase

                        filteredColumn = datosTrain[datosTrain[:,-1]==clase][:, i]
                        # We extract mean and variance of the i-th column
                        mean = np.mean(filteredColumn)
                        std = np.std(filteredColumn)
                        if math.isnan(std) or std==0:
                            raise ZeroDivisionError('The standard deviation is 0 or NaN for the data ', filteredColumn)
                        classesTable[clase] = norm(mean, std)
                # If needed, we apply Laplace correction to the table: classesTable[clase][value] needs to be incremented for all clase, value
                # Ineficiente a tope pero al menos no es larguisimo, nValuesxNclasses no deberia ser un valor muy grande
                if laplaceNeedsToBeApplied:
                    for clase in classes:
                        for value in attrValues:
                            classesTable[clase][value] += 1

                attrTables.append(classesTable)
        self.NBTables = attrTables
        return


    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pred = []
        classes = diccionario[-1].values()
        for data in datosTest:
            maxClass = ['Initial maximum class', 0]
            for clase in classes:
                # Initialize the posteriori numerator as the priori probability for clase
                try:
                    verodotpriori = self.prioris[clase]
                except:
                    print('Clases:', classes)
                    print('Prioris:', self.prioris)
                # Now we multiply by each P(attrN == valueofattrNinourdataelement | clase)
                nAtributos = len(atributosDiscretos)-1
                for i in range(nAtributos):
                    # Value of the i-th attribute in the given datosTest element
                    value = data[i]
                    # We search in NBTables:
                    # take the i-th position of the array, corresponding to the dictionary of the i-th attribute
                    if atributosDiscretos[i]:
                        # inside, if the attribute is discrete, the dictionary corresponding to the 'clase' key
                        # and from there, the key 'value' (which is number of occurrences of class = clase and ithattribute = value)
                        nOccurrences = self.NBTables[i][clase][value]
                        # And divide by the number of occurrences of the other values given class = clase
                        vero = nOccurrences/sum(self.NBTables[i][clase].values())
                    else :
                        # if the attribute is continuous, take the distribution stored in the 'clase' key
                        # and calculate the pdf of the ith attribute being the value that our datosTest element has
                        vero = self.NBTables[i][clase].pdf(value)
                    verodotpriori *= vero

                # If the last calculated numerator is greater than the previous max, update the class and its numerator
                if verodotpriori > maxClass[1]:
                    maxClass = [clase, verodotpriori]
            # We append to the pred array the class predicted for the datosTest element we are testing
            pred.append(maxClass[0])
        return pred

    def clasificaROC(self, datosTest, atributosDiscretos, diccionario, alpha):
        # A set with al classes
        classes = diccionario[-1].values()
        pr = dict()
        # For each data
        i = 0
        for dato in datosTest:
            pr[i] = dict()
            # And for each class
            for clase in classes:
                vero = 1
                j = 0
                # We calculate the product or all veros
                # of all attribute values in data, given the class
                for value in dato[:-1]:
                    if atributosDiscretos[j]: #Nominal
                        nOccurrences = self.NBTables[j][clase][value]
                        vero *= nOccurrences/sum(self.NBTables[j][clase].values())
                    else:#Discreto
                        vero *= self.NBTables[j][clase].pdf(value)
                    j+=1
                pr[i][clase] = vero
            i+=1
        # Positive class = 1: we get the probability for all data
        # given the positive class, and normalize the vector
        positiveProbs = np.array([pr[i][1] for i in range(len(datosTest))])
        positiveProbs /= np.linalg.norm(positiveProbs)
        pred = [1 if prob > alpha else 0 for prob in positiveProbs]
        return pred
##############################################################################

class ClasificadorVecinosProximos(Clasificador):
    def __init__(self, k = 1, weight = False, max_weight = 100):
        self.k = k
        self.weight = weight
        self.max_weight = max_weight

    # KNN no requiere entrenamiento realmente.
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        self.trainData = datostrain

    # Funcion que clasifica un vector
    # v:    vector a clasificar (fila de la matriz)
    # datos:    matriz sobre la que clasificar
    # atributosDiscretos:   array bool con la indicatriz de los atributos nominales
    # k:    numero de vecinos proximos
    # descripcion:  1 - calcular distancia del vector v con respecto a cada
    # return:   clase predicha
    #                   fila de la matriz datos
    #               2 - ordenar por distancias (menor a mayor)
    #               3 - obtener las k distancias menores
    #               4 - obtener las clases que aparecen
    #               5 - sin pesos:  predecir la clase que más aparece
    #                   con pesos:  para cada distancia obtener el inverso
    #                               para cada clase sumar esos inversos
    #                               predecir la mayor
    #                               (sirve para corregir en caso de que haya
    #                               mas vecinos proximos a grandes distancias)

    def clasificaFila(self, v, datos, atributosDiscretos, k, weight=False):
        dists = [(row[-1], distancia(v, row[:-1], atributosDiscretos[:-1])) for row in datos]
        dists = sorted(dists, key=lambda elem: elem[1])
        dists = dists[:k]
        classes = [x[0] for x in dists]
        if weight == False:
            clase = max(set(classes), key=classes.count)
        else:
            inv_dists = [(x[0], 1/x[1]) if x[1] != 0 else (x[0], self.max_weight*k) for x in dists]
            weighted_classes = []
            for x in set(classes):
                weights = [d[1] if x == d[0] else 0 for d in inv_dists]
                weighted_classes.append((x, sum(weights)))
            clase = max(weighted_classes, key=lambda x: x[1])[0]
        return clase


    def clasifica(self,datostest,atributosDiscretos,diccionario):
        datos_n = self.normalizarDatos(datostest, atributosDiscretos)
        pred = [self.clasificaFila(row[:-1], self.trainData, atributosDiscretos, self.k, self.weight) for row in datostest]
        return pred


    # Funcion que calcula las medias y las desviaciones tipicas de los
    # atributos Continuos de la matriz datos
    # datos:    matriz de datos continuos y discretos
    # atributosDiscretos:   array bool con la indicatriz de los atributos nominales
    def calcularMediasDesv(self,datos,atributosDiscretos):
        return [True if atr == True else (np.mean(col), np.std(col)) for col, atr in zip(datos.T,atributosDiscretos)]

    # Funcion que normaliza los atributos continuos de la matriz datos
    # datos:    matriz de datos continuos y discretos
    # atributosDiscretos:   array bool con la indicatriz de los atributos nominales
    def normalizarDatos(self,datos,atributosDiscretos):
        avg = 0
        std = 1
        med_desv = self.calcularMediasDesv(datos,atributosDiscretos)
        datos_normalizados = np.array([x if atr == True else (x - atr[avg])/atr[std] for x,atr in zip(datos.T, med_desv)])
        return datos_normalizados.T




##############################################################################
##############################################################################

class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, cteApr=1, nEpocas=15):
        self.cteApr = cteApr
        self.nEpocas = nEpocas

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        self.w = np.random.rand(len(atributosDiscretos)) - 0.5
        for epoca in range(self.nEpocas):
            for dato in datosTrain:
                x = np.append([1], dato[:-1])
                clase = 1 if dato[-1] == 1 else 0
                # w = w - nu*(sigm(wx)-clase)x
                # If dot product is too low, 1/ e^inf could incur in numerical values
                # So we replace 1/e^inf with 0
                sigarg = np.dot(self.w,x)
                sig = sigmoidal(sigarg) if sigarg > -600 else 0
                coefficient = self.cteApr * (sig - clase)
                self.w -= coefficient*x

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pred = []
        for data in datosTest:
            x = np.append([1], data[:-1])
            sigarg = np.dot(self.w,x)
            vero = sigmoidal(sigarg) if sigarg > -600 else 0
            pred.append( 1 if vero > 0.5 else 0)
        return pred



##############################################################################
class ClasificadorGenetico(Clasificador):
    def __init__(self, tamPoblacion=100, nEpocas=100, pCruce=0.85, pElit=0.05, maxReglas=10, usePrior=True):
        self.tamPoblacion = tamPoblacion
        self.nEpocas = nEpocas
        self.pCruce = pCruce
        self.pElit = pElit
        self.maxReglas = maxReglas
        if usePrior is True:
            self.prior = 0
        else: self.prior = -1


        # Aseguramos que no varia el tamano de la poblacion
        self.numElitismo = round(self.tamPoblacion * self.pElit)
        self.numProgenitores = round(self.tamPoblacion * self.pCruce)
        self.numProgenitores += self.numProgenitores%2
        self.numMutar = self.tamPoblacion - self.numElitismo - self.numProgenitores

    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        # Paso 1: Crear una poblacion aleatoria
        #   tamano tamPoblacion (numero de cromosomas)
        #   numero de reglas --> aleatorio entre
        #   longitud de una regla depende de los valores que puedan tomar
        #   los atributos si un atr toma n valores -> n bits
        nAtributos = len(diccionario) - 1
        lensAtributos = [len(diccionario[i]) for i in range(0, nAtributos)]
        self.poblacion = [[cr.Cromosoma(r.randint(1,self.maxReglas), lensAtributos), None] for i in range(0,self.tamPoblacion)]
        # Codificamos los datos de entrenamiento para ahorrar tiempo
        temp_cr = self.poblacion[0][0]
        if datosTrain.dtype is not int:
            datosTrain = datosTrain.astype(dtype=int)
        encoded_train = np.array([np.append(temp_cr.encode(row), 0) for row in datosTrain[:,:-1]])
        clases = datosTrain[:,-1]
        if self.prior == 0:
            self.prior = round(sum(clases)/len(clases))
        encoded_train[:,-1] = clases
        self.train = encoded_train
        self.currentGen = 1
        self.avgFitness = 0
        self.currentGen = 0
        self.topFitnessHistory = []
        self.avgFitnessHistory = []

        for i in range(0, self.nEpocas):
            self.calcularFitnessPoblacion()
            self.poblacion.sort(key=lambda elem: elem[1])
            self.topFitnessHistory.append(self.poblacion[-1][1])
            self.avgFitnessHistory.append(self.avgFitness)
            print("Gen ",i+1,"/",self.nEpocas, "- BEST FITNESS: ", self.poblacion[-1][1], " AVG FITNESS: ", self.avgFitness)
            # Si ya tenemos fitness del 100% no tiene sentido seguir
            if(self.poblacion[-1][1] == 1.0):
                print("Stop -> Max Fitness Achieved")
                break
            self.evolucionaPoblacion()
            self.currentGen += 1

        self.calcularFitnessPoblacion()
        self.poblacion.sort(key=lambda elem: elem[1])


    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        classifier = self.poblacion[-1][0]
        if datosTest.dtype is not int:
            datosTest = datosTest.astype(dtype=int)
        encoded_test = np.array([classifier.encode(row) for row in datosTest[:,:-1]])
        pred = [classifier.predict(row, default=self.prior) for row in encoded_test]
        return pred



    # calcula el fitness de cada individuo de la poblacion
    # devuelve una lista de tuplas con (cromosoma, fitness)
    def calcularFitnessPoblacion(self, poblacion=None):
        if poblacion is None:
            pob = self.poblacion
        else: pob = poblacion
        self.avgFitness = 0
        for elem in pob:
            if elem[1] is None:
                cr = elem[0]
                elem[1] = cr.calcularFitness(datos=self.train, default=self.prior)
            self.avgFitness += elem[1]
        self.avgFitness = self.avgFitness/self.tamPoblacion

    # asume que la poblacion esta ordenada de menor a mayor
    def ruletaIndices(self):
        total = sum([0 if fit[1] is None else fit[1] for fit in self.poblacion])
        choice = r.uniform(0,total)
        suma = 0
        for i in range(0, self.tamPoblacion):
            fit = self.poblacion[i][1]
            suma += fit
            if suma >= choice:
                return deepcopy(self.poblacion[i])

        # Puede que por errores minimos de float, salga del bucle
        # devolvemos el ultimo individuo en este caso
        return self.poblacion[-1]


    # selecciona a los mejores cromosomas
    def elitismo(self):
        if(0 < self.numElitismo):
            supervivientes = [deepcopy(el) for el in self.poblacion[-self.numElitismo:]]
            return supervivientes
        else: return []



    # Selecciona individuos para cruzar mediante una ruleta basada en pesos
    # par es una lista [[[cr1, 0.3],[cr2,0.4]], [[cr3, 0.1],[cr4,0.4]]...]
    #   cada elemento de par es una lista de 2 elementos [[cr1, 0.3],[cr2,0.4]]
    #       cada elemento de estos es un elemento de la poblacion [cr1, 0.3]
    def seleccionCruce(self):
        pares = [[self.ruletaIndices(), self.ruletaIndices()] for i in range(0, self.numProgenitores//2)]
        return pares

    # Selecciona individuos para mutar mediante una ruleta basada en pesos
    def seleccionMutar(self):
        paraMutar = [self.ruletaIndices() for i in range(0, self.numMutar)]
        return paraMutar


    # Cruza los cromosomas de los progenitores de dos en dos
    # con probabilidad pCruce
    # pares == [[[cr1, 0.3],[cr2,0.4]], [[cr3, 0.1],[cr4,0.4]]...]
    # par in pares == [[cr1, 0.3],[cr2,0.4]]
    # par1 == [cr1, 0.3]
    # par1[0] == cr1
    # par1[1] == 0.3
    def cruzarProgenitores(self, pares):
        cruces = []
        for par in pares:
            par1 = par[0]
            par2 = par[1]
            par1[0].cruzar(par2[0])
            # Modificamos su fitness porque hay que volver a calcular
            par1[1] = None
            par2[1] = None
            cruces.append(par1)
            cruces.append(par2)

        return cruces


    # cambia 1 bit de los cromosomas de los progenitores
    def mutarProgenitores(self, progenitores):
        for p in progenitores:
            p[0].mutar()
            # Modificamos su fitness porque hay que volver a calcular
            p[1] = None
        return progenitores


    # Funcion que crea la siguiente generacion a partir de la anterior
    # a base de 1 - Elitismo
    #           2 - Cruce
    #           3 - Mutacion
    def evolucionaPoblacion(self):
        # Paso 1 ELITISIMO: elegir a los mejores y add a la siguiente generacion
        nextGen = self.elitismo()
        # Paso 2 CRUCE: elegir para cruzar proporcional al fitness
        # y add a la siguiente generacion
        pares = self.seleccionCruce()
        cruzados = self.cruzarProgenitores(pares)
        nextGen.extend(cruzados)
        # Paso 3 MUTACION: elegir para mutar proporcional al fitness
        # y add a la siguiente generacion
        paraMutar = self.seleccionMutar()
        mutados = self.mutarProgenitores(paraMutar)
        nextGen.extend(mutados)
        # Sustituir la poblacion con la siguiente generacion
        self.poblacion = nextGen

def testPob(lista=None):
    pob = [c[0] for c in lista]
    test = set(pob)
    if(len(test) != len(pob)):
        print("\nERROR")
        for el in test:
            pob.remove(el)
        print("CULPRITs == ", pob)









##############################################################################

# Funciones auxiliares

# Funcion para calcular los prioris del conjunto de datos de train
# Recibe una matriz np con datos como filas. Ultima columna corresponde a la clase
# Devuelve un diccionario que tiene las clases como keys, y el priori de la clase como valor

def prioris(datosTrain):
    prioris = dict()
    nDatos = len(datosTrain)
    classIdx = np.size(datosTrain,1)-1
    for dato in datosTrain:
        clase = dato[classIdx]
        if clase not in prioris.keys():
            prioris[clase] = 1/nDatos
        else :
            prioris[clase] += 1/nDatos
    return prioris

# Funcion para calcular la sigmoidal
def sigmoidal(t):
  return 1 / (1 + math.exp(-t))

# Funcion para calcular distancia entre dos vectores
# v1,v2: vectores de igual longitud
# atributes: vector de atributos
# si el atributo es nominal, se utiliza la distancia de Manhattan
def distancia(v1, v2, atributes):
  dst = []
  for x,y,atr in zip(v1,v2,atributes):
      if atr == True:
          if x == y:
              dst.append(0)
          else:
              dst.append(1)
      else:
          dst.append((x - y)**2)
  return math.sqrt(sum(dst))
