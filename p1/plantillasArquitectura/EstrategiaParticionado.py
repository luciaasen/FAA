from abc import ABCMeta,abstractmethod
import random


class Particion():

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
    # Constructor
    def __init__(self,nombreEstrategia, numeroParticiones):
        # Clase abstracta
        __metaclass__ = ABCMeta
        self.nombreEstrategia = nombreEstrategia
        self.numeroParticiones = numeroParticiones
        self.listaParticiones = []

      # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor


    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self,datos,seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    # Constructor
    def __init__(self, nombreEstrategia, porcentajeDeseado):
        # super().__init__(nombreEstrategia, 2, porcentajeDeseado)
        # 2 es el numero de particiones en validacion simple
        super().__init__(nombreEstrategia, 2)
        # Porcentaje deseado es una propiedad especifica de la validacion simple
        self.porcentajeDeseado = porcentajeDeseado


    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        # Assuming porcentajeDeseado refers to the percentage of the data we want to save for testing
        # we obtain the number of rows (data) to use for testing
        testRows = int( (self.porcentajeDeseado * totalRows)/100 )
        rows = list(range(0, totalRows))
        random.shuffle(rows)
        particionSimple = Particion()
        # array size of totalRows - testRows
        particionSimple.indicesTrain = rows[testRows :]
        # array size of testRows
        particionSimple.indicesTest = rows[: testRows]

        self.listaParticiones.append(particionSimple)


        return self.listaParticiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, nombreEstrategia, numParticiones):
        super().__init__(nombreEstrategia, numParticiones)

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        rows = list(range(0, totalRows))
        random.shuffle(rows)
        # number of rows in each partition (we take the integer part for the size of the test data)
        partitionSize = int(totalRows / self.numeroParticiones)
        # Decision de diseno: en el caso de que no se divida perfectamente el conjunto en el
        # numero de particiones, los datos extra se utilizaran en entrenamiento. Esto da lugar
        # a que en una iteracion, el mismo conjunto de datos (los del final) se utilicen
        # exclusivamente para entrenar y jamas entren en el conjunto de testing
        for i in range(0, self.numeroParticiones):
            particionCruzada = Particion()
            test = rows[i * partitionSize : (i + 1) * partitionSize]
            train = list(set(rows) - set(test))
            particionCruzada.indicesTrain = train
            particionCruzada.indicesTest = test
            self.listaParticiones.append(particionCruzada)

        return self.listaParticiones
