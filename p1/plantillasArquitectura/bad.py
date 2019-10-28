from abc import ABCMeta,abstractmethod


class Particion():

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
    # Constructor
    def __init__(nombreEstrategia, numeroParticiones):
        self.nombreEstrategia = nombreEstrategia
        self.numeroParticiones = numeroParticiones
        self.listaParticiones = []
      # Clase abstracta
      __metaclass__ = ABCMeta

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
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        # we assign a new value to seed only if it is None already
        if seed == None:
            random.seed(seed)

        # number of rows of the datos Matrix (number of inidvidual data)
        totalRows = len(datos)
        # Assuming porcentajeDeseado refers to the percentage of the data we want to save for testing
        testRows = int( (self.porcentajeDeseado * totalRows)/100 )


        trainRows = totalRows - testRows
        # list of row indices
        rows = list(range(0, totalRows))
        particionSimple = Particion()
        # fill the indicesTest list by choosing randomly from the list of all the indices
        # by the end of the loop, the indices left in the 'rows' list can simply be assigned to our indicesTrain list
        for i in range(0, testRows):
            chosen = random.choice(rows)
            particionSimple.indicesTest.append(chosen)
            rows.remove(chosen)

        particionSimple.indicesTrain = rows

        return particionSimple


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, nombreEstrategia, numParticiones):
        super().__init__(nombreEstrategia, numParticiones)

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
        # number of rows in each partition
        partitionSize = int(totalRows / self.numeroParticiones)
        rows = list(range(0, totalRows))
        random.shuffle(rows)
        particionCruzada = Particion()
        #
        # for i in range(0, self.numeroParticiones - 1):
        #     for j in range(0, partitionSize):
        #         chosen = rows[j]
        #         particionCruzada.indicesTrain.append(chosen)
        #         rows.remove(chosen)
        #
        # particionCruzada.indicesTest = rows

        # Es raro hacer el bucle de arriba, si indices solo guarda indices.

        return particionCruzada



from abc import ABCMeta,abstractmethod
import random


class Particion():

    # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones
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

        return particionSimple


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
        # number of rows in each partition
        partitionSize = int(totalRows / self.numeroParticiones)
        rows = list(range(0, totalRows))
        random.shuffle(rows)
        particionCruzada = Particion()

        # lim es el tamano del cjto de datos hasta que sea test data
        lim = partitionSize * (self.numeroParticiones - 1)
        particionCruzada.indicesTrain = rows[:lim]
        particionCruzada.indicesTest = rows[lim:]

        # insertamos listas de tamano partitionSize
        for i in range(0, self.numeroParticiones - 1):
            self.listaParticiones.append(particionCruzada.indicesTrain[i*partitionSize: (i+1)*partitionSize])

        self.listaParticiones.append(particionCruzada.indicesTest)

        return particionCruzada



# TRY 1: USING train_test_split function
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
        numTestRows = int(percent * totalRows))
        rows = list(range(0, totalRows))
        particionSimple = Particion()
        # this time we use the scikit learn function which automatically shuffles
        # and splits the data
        train, test = train_test_split(rows, test_size=percent, random_state=seed, shuffle=True)
        particionSimple.indicesTrain = train
        particionSimple.indicesTest = test

        self.listaParticiones.append(particionSimple)


        return self.listaParticiones


# TRY 2: USING ShuffleSplit for both
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
