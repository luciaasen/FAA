import Cromosoma as cr
from Datos import Datos
import numpy as np
import Clasificador as cl
import EstrategiaParticionado as ep

################################################################################################
# Prueba de la clase Cromosoma
################################################################################################
# numReglas = 1
# len atrib1 = 3 (puede tomar vals 0,1,2)
# len atrib2 = 2 (puede tomar vals 0,1)
# len atrib3 = 2 (puede tomar vals 0,1)
# c1 = [0,0,0,0,0,0,0,0]
# c2 = [1,1,1,1,1,1,1,1]

# numReglas = 1
# seed = 1
# lensAtributos = [3,2,2]
# lenRegla = sum(lensAtributos) + 1
# p = 2
# r1 = np.array([0 for i in range(0,lenRegla * numReglas)])
# r2 = np.array([1 for i in range(0,lenRegla * numReglas)])
# c1 = cr.Cromosoma(numReglas, lensAtributos, reglas=r1)
# c2 = cr.Cromosoma(numReglas, lensAtributos, reglas=r2)
#
# # c1 = [0,0,0,0,0,0,0,0]
# # c2 = [1,1,1,1,1,1,1,1]
# print("Inicio")
# print("C1 == ", c1.reglas)
# print("C2 == ", c2.reglas)
#
# # c1 = [1,1,0,0,0,0,0,0]
# # c2 = [0,0,1,1,1,1,1,1]
# c1.cruzar(c2, p)
# print("Despues Cruce")
# print("C1 == ", c1.reglas)
# print("C2 == ", c2.reglas)
#
# c1.mutar()
# print("Despues Mutar C1")
# print("C1 == ", c1.reglas)
# print("C2 == ", c2.reglas)
#
# c2.mutar()
# print("Despues Mutar C2")
# print("C1 == ", c1.reglas)
# print("C2 == ", c2.reglas)
#
# c3 = cr.Cromosoma(numReglas*2, lensAtributos)
# print("Despues Crear C3")
# print("C1 == ", c1.reglas)
# print("C3 == ", c3.reglas)
# c1.cruzar(c3, 2)
#
# print("Despues Cruzar C3 C1")
# print("C1 == ", c1.reglas)
# print("C3 == ", c3.reglas)
#
#
# dato = np.array([2,0,1])
# print("dato == ", dato)
# encoded = c1.encode(dato)
# # encoded = [0 0 1 1 0 0 1]
# print("encoded == ", encoded)
# datos = np.array([[0]*(lenRegla), [1]*(lenRegla)])
# # datos = [[0 0 0 0 0 0 0 0] [1 1 1 1 1 1 1 1]]
# print("datos == ", datos)
# # c1 = [1,1,0,0,0,0,0,0] con 1 bit mutado
# print("reglas C1 == ", c1.reglas)
# # deberia predecir ambos a 0 (porq c1 no 'reconoce' a ningun dato)
# print("\nBEGIN FITNESS CALCULATIONS\n")
# fc11 = c1.calcularFitness(datos=datos)
# print("Fitness c1 == ", fc11)
# r4 = np.append(c1.reglas, datos[0])
# r4 = np.append(r4, datos[1])
# c4 = cr.Cromosoma(3, lensAtributos, reglas=r4)
# print("reglas C4 == ", c4.reglas)
# fc41 = c4.calcularFitness(datos=datos)
#
# # Datos mas reales
# newdatos = np.array([[2, 0, 1], [2, 1, 0], [1, 1, 1]])
# print("new datos ==\n ", newdatos)
# encDatos = np.array([np.append(c4.encode(row), 0) for row in newdatos[:,:-1]])
# clases = newdatos[:,-1]
# encDatos[:,-1] = clases
# lensAtributos = [3,2]
# print("Encoded matrix ==\n ", encDatos)
# r5 = np.append(encDatos[0], encDatos[1])
# c5 = cr.Cromosoma(2, lensAtributos, reglas=r5)
# fc51 = c5.calcularFitness(datos=encDatos)
# print("Fitness c5 == ", fc51)


# Ejemplo para que se vea del todo lo que esta pasando
# 3 atributos
#   atrib1 toma valores 0,1,2
#   atrib2 toma valores 0,1
#   atrib3 toma valores 0,1
#
# datos entrenamiento (matriz)
#   datos = [[0, 0, 0, 1], [1, 1, 1, 1]]
#       codificados = [[10010101][01001011]]
# 3 cromosomas
#  cBueno = [11111111] siempre predice 1 (por suerte)
#  cMedio = [11111111 10010100] predice 1 (y 0 para dato 2)
#  cMalo = [11111110] predice siempre 0
# lensAtributos = [3, 2, 2]
# rBueno = np.array([1]*(sum(lensAtributos) + 1))
# rMedio = np.append(np.array([1]*(sum(lensAtributos) + 1)), np.array([1,0,0,1,0,1,0,0]))
# rMalo = np.array([1]*(sum(lensAtributos) + 1))
# rMalo[sum(lensAtributos)] = 0
# datos = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
# print("Datos:\n ", datos)
#
#
# cBueno = cr.Cromosoma(1, lensAtributos, reglas=rBueno)
# cMedio = cr.Cromosoma(2, lensAtributos, reglas=rMedio)
# cMalo = cr.Cromosoma(1, lensAtributos, reglas=rMalo)
# encDatos = np.array([np.append(cBueno.encode(row), 0) for row in datos[:,:-1]])
# clases = datos[:,-1]
# encDatos[:,-1] = clases
# print("Datos Codificados:\n ", encDatos)
# print("cBueno: ", cBueno.reglas)
# print("cMedio: ", cMedio.reglas)
# print("cMalo: ", cMalo.reglas)
# fBueno = cBueno.calcularFitness(datos=encDatos)
# fMedio = cMedio.calcularFitness(datos=encDatos)
# fMalo = cMalo.calcularFitness(datos=encDatos)
# print("Fitness cBueno: ", fBueno)
# print("Fitness cMedio: ", fMedio)
# print("Fitness cMalo: ", fMalo)
#
#
# # prueba crear cromosoma aleatorio:
# cAl = cr.Cromosoma(1, lensAtributos)
# print("reglas al == ", cAl.reglas)



################################################################################################
# dataset=Datos('DatasetEjemplo/ejemplo1.data')
# dataset=Datos('DatasetEjemplo/tic-tac-toe.data')
# diccionario = dataset.diccionarios
# print(diccionario)
# nReglas = 3
# nAtributos = len(diccionario) - 1
# lensAtributos = [len(diccionario[i]) for i in range(0, nAtributos)]
# lenRegla = sum(lensAtributos) + 1
#
# print("nAtributos = 9 --> ", nAtributos)
# print("lenAtributos = [3, 3, 3...3] --> ", lensAtributos)
# print("lenRegla = 3 + 3 ... + 3== 28 --> ", lenRegla)

dataset=Datos('DatasetEjemplo/tic-tac-toe.data')
# dataset=Datos('DatasetEjemplo/ejemplo1.data')
# dataset=Datos('DatasetEjemplo/ejemplo2.data')
dicc = dataset.diccionarios

# datosTrain = dataset.datos[:50]
# print(datosTrain)
gen = cl.ClasificadorGenetico(tamPoblacion=100, nEpocas=10, seed=3)
errors = []
for i in range(2):
    es = ep.ValidacionSimple(20)
    errors.append(gen.validacion(es,dataset,gen))
errorsnp = np.array(errors)
mean,std = np.mean(errors), np.std(errors)
print("MEAN ERRORS == ", mean, " STD == ", std)
# # gen.entrenamiento(datosTrain, dicc, dicc)
# datosTest = dataset.datos[50:]
