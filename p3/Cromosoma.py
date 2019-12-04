# Clase cromosoma

# Supongamos que tenemos un dato con 2 atributos y clase
# por ejemplo: +,-,1
# codificamos atr1/2 en binario (+: 0, -: 1)
# codificamos la clase en binario (0: 0, 1: 1)
# atr1 puede tomar valores +,- {0,1}
# atr2 puede tomar valores +,- {0,1}
# clase puede tomar valores 0,1 {0,1}
#
# codificamos atr1 con 2 bits:  00 => no puede tener ni un +, ni un -
#                               01 => no puede tener un + pero si un -
#                               10 => puede tener un + pero no un -
#                               11 => puede tener un + y un -
# Parecido a lo de los esquemas que vimos en teoria:
# Sea la regla 01111:
#       Dato -,+ -> (10) lo clasifica con un 1
#       Dato -,- -> (11) lo clasifica con un 1
#       Dato +,- -> (01) no lo clasifica
#
# Eg: cromosoma tendra 3 reglas de longitud 5: C = {01111, 01011, 01100}
#   En verdad C = 011110101101100
import random as r
import numpy as np



class Cromosoma:

    # numReglas --> numero de reglas que contiene el cromosoma
    # lenRegla --> numero de bits en cada regla
    # lensAtrs --> numero de bits de para codificar cada atributo
    # reglas    --> array numpy de bits numReglas*lenRegla
    #           --> si no existe genera las propias aleatoriamente
    def __init__(self, numReglas, lensAtributos, reglas=None):
        self.nReglas = numReglas
        self.lenRegla = sum(lensAtributos) + 1
        self.lensAtributos = lensAtributos
        if reglas is None:
            # regla = [r.randint(0,1) for i in range(0, self.lenRegla)]
            self.reglas = np.array([r.randint(0,1) for i in range(0, self.nReglas*self.lenRegla)])
        else:
            self.reglas = reglas

    # Funcion que cruza 2 cromosomas en un punto p
    # Eg C1 = 00|0111, C2 = 11|1000
    # Cruce en p=2: C1' = 001000 C2' = 110111
    # 0 <= p < longitud del cromosoma mas pequeno
    #   se pueden cruzar 2 cromosomas de long distinta
    #   C1 = 00|0111 C2 = 11|0011001100
    #   C1' = 000011001100 C2' = 110111
    def cruzar(self, c2, p=-1):
        if p == -1:
            max = min(len(self.reglas), len(c2.reglas)) - 1
            p = r.randint(0, max)

        c11 = np.copy(self.reglas[:p])
        c21 = np.copy(c2.reglas[:p])
        self.reglas[:p] = c21
        c2.reglas[:p] = c11


    # funcion que cambia 1 bit aleatorio de un cromosoma
    def mutar(self):
        bit = r.randint(0, self.nReglas*self.lenRegla - 1)
        if self.reglas[bit] == 0:
            self.reglas[bit] = 1
        else: self.reglas[bit] = 0

    # usando las reglas un cromosoma clasifica un dato
    # dato --> dato a clasificar
    # error --> clasificacion en caso de no ser reconocido por una regla
    # eg1: 2 atributos binarios y clase 0,1
    # c1 = 11100, dato = 00 -> cromosoma predice 0
    #             dato = 01 -> error -> clasifica como un 0 (porque asi vamos a hacerlo)
    # c2 = 10100 01101, dato 00 -> c21 predice 0 c22 error -> clasifica 0
    #                   dato 11 -> c21 error, c22 error -> clasifica 0 (porque asi vamos a hacerlo)
    # c3 = 11110 11100 11101,   dato 11 -> 0
    #                           dato 10 -> 0,0,1 --> 0
    def predict(self, dato, default=-1):
        nAtributos = len(self.lensAtributos)
        predicciones = {0: 0, 1: 0}
        for n in range(0, self.nReglas):
            regla = self.reglas[n*self.lenRegla: (n + 1)*self.lenRegla]
            # print("REGLA == ", regla)
            # print("REGLA[:-1] == ", regla[:-1])
            # print("Dato == ", dato)
            # print("DOT == ", np.dot(dato, regla[:-1]))
            # para que una regla 'reconozca' el dato
            # el producto escalar tiene que dar el numero de atributos
            if np.dot(dato, regla[:-1]) == nAtributos:
                predicciones[regla[self.lenRegla - 1]] += 1
        if predicciones[0] == predicciones[1] and default != -1:
            return default
        else:
            return max(predicciones, key=predicciones.get)

    # Funcion para calcular el fitness de un cromosoma
    # func --> funcion con la que calcular el fitness
    # datos --> matriz de datos codificada donde la ultima columna es la clase de los datos
    # por defecto suma los bits del cromosoma y devuelve eso
    def calcularFitness(self, func=None, datos=None):
        if datos is not None:
            # ultima columna de datos --> clases
            clases = datos[:,-1]
            # resto de las columnas de datos
            data = datos[:,:-1]
            pred = [self.predict(row) for row in data]
            aciertos = 0
            for prediccion, clase in zip(pred, clases):
                if prediccion == clase:
                    aciertos +=1
            fitness = aciertos/len(clases)
            return fitness
        elif func is not None:
            return func(self)
        else:
            return func(self)

    # Codifificación que se va a utilizar:
    # supongamos que un atributo puede tomar 3 valores (0,1,2)
    #       Ese atributo se codifica con 3 bits
    #       atr = 0 --> 100
    #       atr = 1 --> 010
    #       atr = 2 --> 001
    def encode(self, dato, atribs=None):
        if atribs is None:
            atribs = self.lensAtributos
        encoded = []
        for i in range(0, len(dato)):
            enc = [0]*atribs[i]
            enc[dato[i]] = 1
            encoded.extend(enc)
        return np.array(encoded)


    def __str__(self):
        return str(self.reglas)
