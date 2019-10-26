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
    # TODO: implementar esta funcion
    def validacion(self,particionado,dataset,clasificador,seed=None):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particiones = particionado.creaParticiones(dataset.datos, seed)
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        for i in particiones.indicesTrain:
            clasificador.entrenamiento(i, dataset.nominalAtributos, dataset.diccionarios)
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado
        # de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        pred = clasificador.clasifica(particiones.indicesTest, dataset.nominalAtributos, dataset.diccionarios)
        error = error(dataset.datos, pred)
        return error

##############################################################################
##############################################################################

##############################################################################

class ClasificadorNaiveBayes(Clasificador):



    # TODO: implementar
    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        pass



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
