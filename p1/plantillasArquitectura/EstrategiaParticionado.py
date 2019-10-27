from abc import ABCMeta,abstractmethod
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


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
    def __init__(self, porcentajeDeseado):
        # 2 es el numero de particiones en validacion simple
        super().__init__("ValidacionSimple", 2)
        # Porcentaje deseado es una propiedad especifica de la validacion simple
        self.porcentajeDeseado = porcentajeDeseado


    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        # Assuming porcentajeDeseado refers to the percentage of the data we want to save for testing
        # we obtain the number of rows (data) to use for testing
        numTestRows = int( (self.porcentajeDeseado * totalRows)/100 )
        rows = list(range(0, totalRows))
        random.shuffle(rows)
        particionSimple = Particion()
        # array size of totalRows - numTestRows
        particionSimple.indicesTrain = rows[numTestRows :]
        # array size of numTestRows
        particionSimple.indicesTest = rows[: numTestRows]

        self.listaParticiones.append(particionSimple)


        return self.listaParticiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, numParticiones):
        super().__init__("ValidacionCruzada", numParticiones)

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
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

#####################################################################################################

class ValidacionSimpleScikit(EstrategiaParticionado):
    # Constructor
    def __init__(self, porcentajeDeseado):
        # 2 es el numero de particiones en validacion simple
        super().__init__("ValidacionSimpleScikit", 2)
        # Porcentaje deseado es una propiedad especifica de la validacion simple
        self.porcentajeDeseado = porcentajeDeseado


    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        # Assuming porcentajeDeseado refers to the percentage of the data we want to save for testing
        # we obtain the number of rows (data) to use for testing
        percent = self.porcentajeDeseado/100
        rows = list(range(0, totalRows))
        # instanciacion de un objeto de clase ShuffleSplit
        # n_splits es el num de particiones (train, test), en el caso de ValidacionSimple, 1
        splitter = ShuffleSplit(n_splits=1, test_size=percent, random_state=seed)
        self.listaParticiones = createSplits(splitter, rows)

        return self.listaParticiones

#####################################################################################################
class ValidacionCruzadaScikit(EstrategiaParticionado):
    def __init__(self, numParticiones):
        super().__init__("ValidacionCruzadaScikit", numParticiones)

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        rows = list(range(0, totalRows))
        # number of rows in each partition (we take the integer part for the size of the test data)
        partitionSize = int(totalRows / self.numeroParticiones)
        # instanciacion de un objeto de clase ShuffleSplit
        # n_splits es el num de particiones (train, test), en el caso de ValidacionCruzada, n
        splitter = ShuffleSplit(n_splits=self.numeroParticiones, test_size=partitionSize, random_state=seed)
        self.listaParticiones = createSplits(splitter, rows)

        return self.listaParticiones



##############################################################################

# Funciones auxiliares

# Funcion comun que extrae la lista de particiones a partir de los porcentajes y
# numero de particiones a partir de un objeto ShuffleSplit
def createSplits(splitter, rows):
    lista = []
    for train, test in splitter.split(rows):
        particion = Particion()
        particion.indicesTrain = list(train)
        particion.indicesTest = list(test)
        lista.append(particion)

    return lista
