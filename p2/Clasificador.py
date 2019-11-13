from abc import ABCMeta,abstractmethod
import math 
import numpy as np
from scipy.stats import norm
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
    def validacion(self,particionado,dataset,clasificador,seed=None, laplace = False, ROC = False):

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
            clasificador.entrenamiento(trainData, dataset.nominalAtributos, dataset.diccionarios, laplace)
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

    def validacionROC(self,particionado,dataset,clasificador,seed=None, laplace = False, alpha = 0.5):
        listaParticiones = particionado.creaParticiones(dataset.datos, seed)
        # Inicializamos error a 0
        error = 0.0
        mConf = np.array([[0,0],[0,0]])
        for particion in listaParticiones:
            # Extraemos datos para el entrenamiento, y los de test
            trainData = dataset.extraeDatos(particion.indicesTrain)
            testData = dataset.extraeDatos(particion.indicesTest)
            # Entrenamos datos (es decir, generamos tablas de Naive Bayes)
            clasificador.entrenamiento(trainData, dataset.nominalAtributos, dataset.diccionarios, laplace)
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


    
    pass
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario, laplace=False):
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
                            if count == 0 and laplace == True:
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

class ClasificadorKNN(Clasificador):



    # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        pass



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        pass




##############################################################################
##############################################################################

class ClasificadorRegresionLogistica:
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario, nEpocas, cteApr):
        self.w = np.random.rand(len(atributosDiscretos)) - 0.5
        for epoca in range(nEpocas):
            for dato in datosTrain:
                x = np._append_const([1], dato[:-1])
                clase = 1 if dato[-1] == 1 else 0
                # w = w - nu*(sigm(wx)-clase)x
                # If dot product is too low, 1/ e^inf could incur in numerical values
                # So we replace 1/e^inf with 0
                sigarg = np.dot(w,x)
                sig = sigmoidal(sigarg) if sigarg > -600 else 0
                coefficient = cteApr * (sig - clase)
                self.w -= coefficient*x

    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pred = []
        for data in datosTest:
            x = np.append([1], data[:-1])
            vero = sigmoidal(np.dot(self.w,x))
            pred.append( 1 if vero > 0.5 else 0)
        return pred  



##############################################################################


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
