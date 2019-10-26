from abc import ABCMeta,abstractmethod


class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # Entendemos que 'datos' y 'pred' son de la misma longitud para poder
    # realizar esta funcion sin controles
    def error(self,datos,pred):
        numErr = 0
        numOk = 0
        rowLen = len(datos[0])

        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        # Suponiendo que pred es una lista con las predicciones, y que por defecto
        # será de la misma longitud que el numero de filas de 'datos'
        # comparamos cada entrada de 'pred' con la ultima entrada de cada fila
        # de 'datos' (que es la clase real)
        for i in range(0,len(pred)):
            if pred[i] == datos[i][rowLen]:
                numOk = numOk + 1
            else:
                numErr = numErr + 1

        # devuelve un numero entre 0,1 que representa el error
        error = numErr/(numErr + numOk)
        return error



    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self,particionado,dataset,clasificador,seed=None):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particiones = particionado.creaParticiones(dataset.datos, seed)
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        for i in particiones.indicesTrain:
            clasificador.entrenamiento(i, dataset.nominalAtributos, dataset.diccionarios)
        #TODO Aqui no faltaria calcular el error en casa iteracion, guardar los valores y al final devolver la media?


        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opcion es repetir la validacion simple un numero especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcularia la media.
        pred = clasificador.clasifica(particiones.indicesTest, dataset.nominalAtributos, dataset.diccionarios)
        error = error(dataset.datos, pred)
        return error

##############################################################################
##############################################################################

##############################################################################

class ClasificadorNaiveBayes(Clasificador):



        pass
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario, laplace=False):
        prioris = prioris(datostrain)
        # There are len - 1 attributes, since last element in atributosDiscretos, the class, does not correspond to a proper attribute
        nAtributos = len(atributosDiscretos)-1
        # We extract a set with all classes in the data
        classes = set(datosTrain[:, -1])
        # The following loop will build the NB tables
        # We create a list where the i-th element of the list is associated to the i-th attribute,
        # and for each attribute a dictionary with classes as keys is created
        # If the attribute is continuous, then for each class key the appropriate normal distribution is assigned as a value
        # Otherwise, for each class key a new dictionary is created, which has the attribute values as keys,
        #            and for each value key,the number of data elements with the current class and current value is assigned as value

        

        attrTables = []
        for i in range(nAtributos-1):
                # For each attribute, a dictionary with all classes as values
                classesTable = dict()
                    
                for clase in classes:
                    # We extract a set with all attribute values for the i-th attribute
                    attrValues = set(datosTrain[:,i])
                    # If i-th attribute discrete:
                    # For each class, a dictionary with the i-th attribute values as keys
                    classesTable[clase] = dict()            
                    if(atributosDiscretos[i]):
                        for value in attrValues:
                            # We count the number of elements of class clase out of the elements 
                            # where the i-th attribute has value value
                            count = np.sum((datosTrain[:,i]==value ) & (datosTrain[:,-1]==clase))
                            #if count == 0 and laplace == True: 
                            #    AQUI SE HARIA LAPLACE: activar una flag que, una vez recorridos todos los values, sume uno a todos los dics
                            #    Ineficiente a tope pero al menos no es larguisimo, nValuesxNclasses no deberia ser un valor muy grande    
                            classesTable[clase][value] = count

                    # If i-th attribute continuous:
                    else:
                        # We create an array with the i-th attribute values of data where class == clase
                        filteredColumn = datosTrain[datosTrain[:,-1]==clase][:, i]
                        # We extract mean and variance of the i-th column
                        mean = np.mean(filteredColumn)
                        std = np.std(filteredColumn)
                        classesTable[clase] = norm(mean, std)
                        
                attrTables.append(classesTable)   
        return attrTables            



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        pass


##############################################################################

class ClasificadorKNN(Clasificador):



    # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        pass



    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        pass




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
